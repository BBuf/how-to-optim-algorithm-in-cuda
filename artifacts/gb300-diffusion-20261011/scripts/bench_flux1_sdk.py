"""Matched FLUX.1-dev SDK latency; uncached text encoders, 50 steps, guidance 3.5."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import image_decode_config_contract


PROMPTS = [
    'A futuristic cyberpunk city at night, neon lights reflecting on wet streets',
    'A red fox sitting in a snowy forest, soft morning light, detailed wildlife photograph',
    'A ceramic teapot and two cups on a wooden table next to a sunlit window, still life',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['sglang', 'vllm'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--bcg', action='store_true')
    args = parser.parse_args()
    assert not args.bcg or args.framework == 'sglang'
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    assert str(root) == os.environ['IMAGE_BENCH_OUTPUT']
    assert not (root/'records.jsonl').exists()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,pci.bus_id', '--format=csv'], text=True)
    (root / 'gpu.txt').write_text(gpu)
    import torch
    (root / 'runtime.json').write_text(json.dumps({'torch':torch.__version__, 'cpu_affinity':sorted(os.sched_getaffinity(0))}))
    if args.framework == 'vllm':
        import vllm_omni.diffusion.worker.diffusion_model_runner as runner_module
        original_compile = runner_module.regionally_compile
        def observe_compile(model, *positional, **keywords):
            block_classes = getattr(model, '_repeated_blocks', ())
            block_groups = getattr(model, '_layerwise_offload_blocks_attrs', ())
            expected = [name for name, module in model.named_modules()
                        if type(module).__name__ in block_classes
                        or (len(name.split('.')) >= 2
                            and name.split('.')[-2] in block_groups
                            and name.split('.')[-1].isdigit())]
            assert expected, 'No native regional compile targets found'
            result = original_compile(model, *positional, **keywords)
            names = [name for name, module in result.named_modules()
                     if hasattr(getattr(module, '_omni_original_forward', module.forward),
                                '_torchdynamo_orig_callable')]
            assert set(names) == set(expected), (names, expected)
            evidence = dict(model_class=type(model).__name__, compiled_modules=names,
                            expected_modules=expected, compile_kwargs=keywords)
            (root / 'compile-activation.json').write_text(json.dumps(evidence, indent=2))
            print('BENCH_COMPILE_ACTIVATION '+json.dumps(evidence), flush=True)
            return result
        runner_module.regionally_compile = observe_compile
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
        engine = Omni(model=args.model, tensor_parallel_size=1, dtype='bfloat16',
                      enforce_eager=False, diffusion_compile_granularity='regional',
                      enable_cpu_offload=False, enable_layerwise_offload=False,
                      vae_use_tiling=False, vae_use_slicing=False,
                      stage_init_timeout=900, init_timeout=900)
        compile_evidence = json.loads((root / 'compile-activation.json').read_text())
        assert compile_evidence['compiled_modules'], 'Regional compile did not activate'
        def generate(prompt, seed):
            params = OmniDiffusionSamplingParams(height=1024, width=1024, seed=seed,
                generator_device='cuda', num_inference_steps=50, guidance_scale=3.5, true_cfg_scale=1.0, max_sequence_length=512,
                num_outputs_per_prompt=1)
            outputs = engine.generate({'prompt':prompt, 'negative_prompt':None, 'modalities':['image']},
                                      sampling_params_list=[params])
            images = extract_images_from_outputs(outputs)
            assert len(images) == 1, len(images)
            return images[0]
        close = engine.close
    else:
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
        engine = DiffGenerator.from_pretrained(model_path=args.model, backend='sglang',
                    model_id='FLUX.1-dev',
                    num_gpus=1, tp_size=1, enable_torch_compile=False,
                    disable_conditioning_cache=True, disable_autocast=True,
                    vae_tiling=False, vae_sp=False,
                    enable_breakable_cuda_graph=args.bcg,
                    warmup_mode='server', warmup_resolutions=['1024x1024'],
                    component_residency={'all':'resident'})
        def generate(prompt, seed):
            result = engine.generate(sampling_params_kwargs=dict(prompt=prompt,
                negative_prompt=None, width=1024, height=1024, num_inference_steps=50,
                guidance_scale=3.5, true_cfg_scale=1.0, max_sequence_length=512, seed=seed, generator_device='cuda',
                quality='lossless', save_output=False))
            assert result is not None
            return result.frames[0]
        close = engine.shutdown
    records = []
    try:
        sequence = [('warmup', i) for _ in range(args.warmup) for i in range(3)]
        sequence += [('measurement', i) for _ in range(args.repeats) for i in range(3)]
        with (root / 'records.jsonl').open('w') as ledger:
            for number, (phase, prompt_id) in enumerate(sequence):
                seed = 42 + prompt_id
                started = time.perf_counter()
                picture = generate(PROMPTS[prompt_id], seed)
                seconds = time.perf_counter() - started
                if not isinstance(picture, Image.Image):
                    array = np.asarray(picture)
                    if array.dtype != np.uint8:
                        array = (array * 255).round().clip(0, 255).astype(np.uint8)
                    picture = Image.fromarray(array)
                assert picture.size == (1024, 1024), picture.size
                path = root / f'{number:03}-{phase}-prompt{prompt_id}.png'
                picture.save(path)
                record = dict(number=number, phase=phase, prompt_id=prompt_id,
                    prompt=PROMPTS[prompt_id], seed=seed, sdk_e2e_s=seconds,
                    rgb_sha256=hashlib.sha256(picture.convert('RGB').tobytes()).hexdigest(),
                    output=str(path))
                records.append(record)
                ledger.write(json.dumps(record)+'\n'); ledger.flush()
                print('BENCH_RECORD '+json.dumps(record), flush=True)
        assert (root/'decode-config.json').exists(), 'Actual VAE decode was not observed'
        assert (root/'denoise-config.json').exists(), 'Actual CFG/steps were not observed'
        measured = [r for r in records if r['phase'] == 'measurement']
        summary = dict(framework=args.framework, bcg=args.bcg, conditioning_cache='disabled', metric='native SDK complete CPU image; PNG save excluded',
            warmup=args.warmup*3, requests=len(measured), gpu=gpu,
            median_s=statistics.median(r['sdk_e2e_s'] for r in measured),
            per_prompt={str(i):statistics.median(r['sdk_e2e_s'] for r in measured if r['prompt_id']==i) for i in range(3)})
        (root / 'summary.json').write_text(json.dumps(summary,indent=2))
        print('BENCH_SUMMARY '+json.dumps(summary),flush=True)
        if args.framework == 'vllm':
            from torch._dynamo.utils import counters
            evidence = {key:dict(value) for key,value in counters.items()}
            (root / 'dynamo-counters.json').write_text(json.dumps(evidence,indent=2))
            assert evidence.get('stats', {}).get('unique_graphs', 0) > 0, evidence
            print('BENCH_DYNAMO_COUNTERS '+json.dumps(evidence),flush=True)
    finally:
        print('BENCH_SHUTDOWN_BEGIN',flush=True)
        close()
        print('BENCH_SHUTDOWN_END',flush=True)


if __name__ == '__main__':
    main()
