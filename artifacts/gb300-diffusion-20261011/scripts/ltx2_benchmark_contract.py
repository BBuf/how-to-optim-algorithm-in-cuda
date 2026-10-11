"""Benchmark-only request alignment and first-warmup schedule observations.

Does not change model arithmetic. SGLang's LTX-2 request adapter predates its
per-modality public fields, so select its existing guider through request extra.
"""
import importlib.abc
import importlib.machinery
import sys
import json
import os
from pathlib import Path

ROOT = Path('/scratch/bbuf-diffusion-20261011')
RECIPE = json.loads((ROOT / 'ltx2-matched-recipe.json').read_text())
OUTPUT = Path(os.environ['LTX2_BENCH_OUTPUT'])
GUIDANCE = RECIPE['stage1_guidance']


def install_sglang_sampling(module):
    LTX2SamplingParams = module.LTX2SamplingParams
    original_extra = LTX2SamplingParams.build_request_extra

    def request_extra(self):
        extra = original_extra(self)
        extra['ltx2_stage1_guider_params'] = dict(GUIDANCE)
        extra['ltx2_distilled_lora_strength_stage_1'] = 0.0
        extra['ltx2_distilled_lora_strength_stage_2'] = 1.0
        return extra

    LTX2SamplingParams.build_request_extra = request_extra


def install_sglang_denoising(module):
    LTX2DenoisingStage = module.LTX2DenoisingStage
    original_prepare = LTX2DenoisingStage._prepare_denoising_loop
    seen = set()

    def observe_prepare(self, batch, server_args):
        ctx = original_prepare(self, batch, server_args)
        phase = batch.extra.get('ltx2_phase', 'unknown')
        if phase not in seen and not batch.is_warmup:
            evidence = dict(
                phase=phase,
                sigmas=ctx.scheduler.sigmas.detach().cpu().float().tolist(),
                steps=int(batch.num_inference_steps),
                width=int(batch.width), height=int(batch.height),
                num_frames=int(batch.num_frames),
                guidance=batch.extra.get('ltx2_stage1_guider_params'),
                latent_shape=list(ctx.latents.shape),
                cpu_affinity=sorted(os.sched_getaffinity(0)),
            )
            OUTPUT.mkdir(parents=True, exist_ok=True)
            (OUTPUT / f'schedule-{phase}.json').write_text(json.dumps(evidence, indent=2))
            print('BENCH_LTX2_SCHEDULE ' + json.dumps(evidence), flush=True)
            seen.add(phase)
        return ctx

    LTX2DenoisingStage._prepare_denoising_loop = observe_prepare


def install_vllm(ltx2_denoise):
    import dataclasses
    original = ltx2_denoise.prepare_scheduler_stage
    seen = set()

    def observe(pipeline, request_inputs, **kwargs):
        result = original(pipeline, request_inputs, **kwargs)
        steps = len(pipeline.scheduler.sigmas) - 1
        phase = 'stage1' if steps == 40 else 'stage2'
        if phase not in seen:
            evidence = dict(
                phase=phase,
                sigmas=pipeline.scheduler.sigmas.detach().cpu().float().tolist(),
                steps=steps, width=request_inputs.width, height=request_inputs.height,
                num_frames=request_inputs.num_frames,
                guidance=dataclasses.asdict(request_inputs.guidance),
                cpu_affinity=sorted(os.sched_getaffinity(0)),
            )
            OUTPUT.mkdir(parents=True, exist_ok=True)
            (OUTPUT / f'schedule-{phase}.json').write_text(json.dumps(evidence, indent=2))
            print('BENCH_LTX2_SCHEDULE ' + json.dumps(evidence), flush=True)
            seen.add(phase)
        return result

    ltx2_denoise.prepare_scheduler_stage = observe


# Install observations only after each module's ordinary runtime import. This
# preserves worker platform initialization and does not import model code early.
TARGETS = {
    'sglang': {
        'sglang.multimodal_gen.configs.sample.ltx_2': install_sglang_sampling,
        'sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising': install_sglang_denoising,
    },
    'vllm': {'vllm_omni.diffusion.models.ltx2.ltx2_denoise': install_vllm},
}[os.environ['LTX2_BENCH_FRAMEWORK']]


class ObservingLoader(importlib.abc.Loader):
    def __init__(self, original, callback):
        self.original, self.callback = original, callback

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        self.callback(module)


class ObservingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        assert spec is not None and spec.loader is not None
        spec.loader = ObservingLoader(spec.loader, TARGETS[fullname])
        return spec


assert not any(name in sys.modules for name in TARGETS)
sys.meta_path.insert(0, ObservingFinder())
