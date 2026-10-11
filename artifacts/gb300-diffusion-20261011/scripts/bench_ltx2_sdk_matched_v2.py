"""Matched LTX-2 40+3-step SDK video/audio comparison; native runtimes only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from PIL import Image

# Imported in spawned SGLang workers too. First-warmup observations are not timed.
import ltx2_decode_config_contract
import ltx2_benchmark_contract as contract

PROMPTS = [
    'A red fox walks slowly through a snowy forest. The camera follows at eye level. Soft morning light filters through pine trees. Snow crunches softly beneath its paws.',
    'Ocean waves roll onto a sandy beach at sunset. A slow steady camera pan follows the shoreline. The sound of gentle waves and distant seabirds.',
    'A ceramic teapot pours hot tea into a cup on a wooden table near a sunlit window. Close-up, steam rising, gentle pouring sound, natural colors.',
]


def cpu_array(value):
    if hasattr(value, 'detach'):
        assert value.device.type == 'cpu', 'SDK returned an unfinished GPU payload'
        value = value.detach()
        # NumPy has no BF16 dtype; normalizing the returned CPU audio is
        # outside the SDK timer, just like image/video persistence.
        if value.is_floating_point():
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def normalize_video(value):
    if isinstance(value, list):
        if len(value) == 1:
            return normalize_video(value[0])
        value = np.stack([np.asarray(x) for x in value])
    value = cpu_array(value)
    while value.ndim > 4 and value.shape[0] == 1:
        value = value[0]
    assert value.shape == (121, 1024, 1536, 3), value.shape
    if value.dtype != np.uint8:
        assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
        value = (value * 255).round().astype(np.uint8)
    return np.ascontiguousarray(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', required=True, choices=['sglang', 'vllm'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sglang-reference', default=None)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prompt-limit', type=int, default=3)
    parser.add_argument('--bcg', action='store_true')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'records.jsonl').exists()
    assert str(root) == os.environ['LTX2_BENCH_OUTPUT']
    import torch
    (root / 'runtime.json').write_text(json.dumps(dict(
        framework=args.framework, torch=torch.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)), recipe=contract.RECIPE,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv'], text=True),
    ), indent=2))
    # The exact same negative prompt is stored alongside the pinned recipe.
    negative = (contract.ROOT / 'ltx2-negative-prompt.txt').read_text()
    if args.framework == 'sglang':
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
        engine = DiffGenerator.from_pretrained(
            model_path=args.model, backend='sglang', pipeline_class_name='LTX2TwoStagePipeline',
            num_gpus=1, tp_size=1, enable_torch_compile=False,
            enable_breakable_cuda_graph=args.bcg, disable_conditioning_cache=True,
            component_residency={'all': 'resident'}, disable_autocast=True, vae_tiling=False, vae_sp=False,
            **(dict(warmup_mode='server', warmup_resolutions=['1536x1024'],
                    warmup_num_frames=121, warmup_steps=1) if args.bcg else {}),
        )
        def generate(prompt, seed):
            output = engine.generate(sampling_params_kwargs=dict(
                prompt=prompt, negative_prompt=negative, width=1536, height=1024,
                num_frames=121, fps=24, num_inference_steps=40, guidance_scale=3.0,
                max_sequence_length=1024, seed=seed, generator_device='cpu',
                generate_audio=True, save_output=False, quality='lossless',
            ))
            assert output is not None and output.audio is not None
            return output.frames, output.audio
        close = engine.shutdown
    else:
        assert args.sglang_reference
        reference = Path(args.sglang_reference)
        schedules = {phase:json.loads((reference / f'schedule-{phase}.json').read_text())['sigmas']
                     for phase in ('stage1', 'stage2')}
        assert len(schedules['stage1']) == 41 and len(schedules['stage2']) == 4
        import vllm_omni.diffusion.worker.diffusion_model_runner as runner
        original_compile = runner.regionally_compile
        def observe_compile(model, *positional, **keywords):
            result = original_compile(model, *positional, **keywords)
            names = [name for name, module in result.named_modules()
                     if hasattr(getattr(module, '_omni_original_forward', module.forward), '_torchdynamo_orig_callable')]
            (root / 'compile-activation.json').write_text(json.dumps(dict(compiled_modules=names, compile_kwargs=keywords), indent=2))
            return result
        runner.regionally_compile = observe_compile
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        engine = Omni(model=args.model, model_class_name='LTX2TwoStagePipeline',
                      tensor_parallel_size=1, dtype='bfloat16', enforce_eager=False,
                      diffusion_compile_granularity='regional', enable_cpu_offload=False,
                      vae_use_tiling=False, vae_use_slicing=False,
                      enable_layerwise_offload=False, stage_init_timeout=1800, init_timeout=1800)
        def generate(prompt, seed):
            params = OmniDiffusionSamplingParams(
                height=1024, width=1536, num_frames=121, seed=seed, generator_device='cpu',
                num_inference_steps=40, num_outputs_per_prompt=1,
                extra_args=dict(contract.GUIDANCE, frame_rate=24, max_sequence_length=1024,
                                stage_1_sigmas=schedules['stage1'], stage_2_sigmas=schedules['stage2']),
            )
            outputs = engine.generate({'prompt':prompt, 'negative_prompt':negative}, sampling_params_list=[params])
            assert len(outputs) == 1
            output = outputs[0]
            assert output.images and output.multimodal_output.get('audio') is not None
            return output.images, output.multimodal_output['audio']
        close = engine.close
    records = []
    try:
        sequence = [('warmup', i) for _ in range(args.warmup) for i in range(args.prompt_limit)]
        sequence += [('measurement', i) for _ in range(args.repeats) for i in range(args.prompt_limit)]
        saved = set()
        with (root / 'records.jsonl').open('w') as ledger:
            for number, (phase, prompt_id) in enumerate(sequence):
                started = time.perf_counter()
                video, audio = generate(PROMPTS[prompt_id], 42 + prompt_id)
                seconds = time.perf_counter() - started
                video = normalize_video(video)
                audio = cpu_array(audio)
                assert audio.size and np.isfinite(audio).all()
                record = dict(number=number, phase=phase, prompt_id=prompt_id, seed=42+prompt_id,
                              prompt=PROMPTS[prompt_id], sdk_e2e_s=seconds,
                              video_sha256=hashlib.sha256(video.tobytes()).hexdigest(),
                              audio_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                              video_shape=list(video.shape), audio_shape=list(audio.shape))
                records.append(record)
                ledger.write(json.dumps(record)+'\n'); ledger.flush()
                print('BENCH_RECORD '+json.dumps(record), flush=True)
                if phase == 'measurement' and prompt_id not in saved:
                    np.save(root/f'prompt{prompt_id}-frames.npy', video)
                    np.save(root/f'prompt{prompt_id}-audio.npy', audio)
                    for index in (0, 60, 120):
                        Image.fromarray(video[index]).save(root/f'prompt{prompt_id}-frame{index}.png')
                    saved.add(prompt_id)
        measured = [r for r in records if r['phase'] == 'measurement']
        if args.framework == 'vllm':
            assert json.loads((root/'compile-activation.json').read_text())['compiled_modules']
            from torch._dynamo.utils import counters
            (root/'dynamo-counters.json').write_text(json.dumps({k:dict(v) for k,v in counters.items()}, indent=2))
            for phase in ('stage1', 'stage2'):
                assert json.loads((root/f'schedule-{phase}.json').read_text())['sigmas'] == schedules[phase]
        summary = dict(framework=args.framework, bcg=args.bcg, conditioning_cache='disabled', metric='native SDK complete CPU video and audio; file saving excluded',
                       requests=len(measured), median_s=statistics.median(r['sdk_e2e_s'] for r in measured))
        (root/'summary.json').write_text(json.dumps(summary, indent=2))
        print('BENCH_SUMMARY '+json.dumps(summary), flush=True)
    finally:
        print('BENCH_SHUTDOWN_BEGIN', flush=True)
        close()
        print('BENCH_SHUTDOWN_END', flush=True)


if __name__ == '__main__':
    main()
