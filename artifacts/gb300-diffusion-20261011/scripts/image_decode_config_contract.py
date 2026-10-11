"""First-warmup observation of actual image VAE precision and tiling; no arithmetic edits."""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

OUTPUT=Path(os.environ['IMAGE_BENCH_OUTPUT'])
FRAMEWORK=os.environ['IMAGE_BENCH_FRAMEWORK']
MODEL=os.environ['IMAGE_BENCH_MODEL']


def observe_vae(vae):
    if getattr(vae,'_bench_decode_observed',False):return
    import torch
    original=vae.decode
    checked=False
    @functools.wraps(original)
    def observed(latents,*args,**kwargs):
        nonlocal checked
        if not checked and not torch.cuda.is_current_stream_capturing():
            record=dict(model=MODEL,framework=FRAMEWORK,vae_class=type(vae).__name__,input_dtype=str(latents.dtype),input_shape=list(latents.shape),parameter_dtype=str(next(vae.parameters()).dtype),autocast_active=torch.is_autocast_enabled('cuda'),use_tiling=getattr(vae,'use_tiling',None),use_slicing=getattr(vae,'use_slicing',None))
            assert latents.dtype==next(vae.parameters()).dtype==torch.bfloat16,record
            assert not record['autocast_active'] and record['use_tiling'] is False,record
            assert record['use_slicing'] in (False,None),record
            OUTPUT.mkdir(parents=True,exist_ok=True)
            (OUTPUT/'decode-config.json').write_text(json.dumps(record,indent=2))
            checked=True
        return original(latents,*args,**kwargs)
    vae.decode=observed
    vae._bench_decode_observed=True


def install_sglang(module):
    cls=module.DecodingStage;original=cls.decode_raw
    original_forward=cls.forward
    active_real_request=False
    @functools.wraps(original_forward)
    def forward_observed(self,batch,*args,**kwargs):
        nonlocal active_real_request
        active_real_request=not batch.is_warmup
        try:
            return original_forward(self,batch,*args,**kwargs)
        finally:
            active_real_request=False
    cls.forward=forward_observed
    @functools.wraps(original)
    def observed(self,*args,**kwargs):
        if active_real_request:
            observe_vae(self.vae)
        return original(self,*args,**kwargs)
    cls.decode_raw=observed


def install_omni(module):
    cls=getattr(module,{'flux1':'FluxPipeline','qwen21':'QwenImage21Pipeline','zimage':'ZImagePipeline'}[MODEL]);original=cls.forward
    observed_zimage=False
    active_real_request=False
    @functools.wraps(original)
    def observed(self,*args,**kwargs):
        nonlocal observed_zimage,active_real_request
        req=args[0] if args else kwargs['req']
        if req.is_dummy_run():
            OUTPUT.mkdir(parents=True,exist_ok=True)
            (OUTPUT/'startup-observation.json').write_text(json.dumps(dict(request_ids=req.request_ids,excluded_from_contract=True)))
            return original(self,*args,**kwargs)
        observe_vae(self.vae)
        active_real_request=True
        try:
            result=original(self,*args,**kwargs)
        finally:
            active_real_request=False
        if MODEL=='zimage' and not observed_zimage:
            record=dict(true_cfg=self.do_classifier_free_guidance,steps=self.num_timesteps,guidance=self.guidance_scale,sigmas=self.scheduler.sigmas.detach().cpu().float().tolist())
            assert record['true_cfg'] is False and record['steps']==9 and record['guidance']==0.0,record
            (OUTPUT/'denoise-config.json').write_text(json.dumps(record,indent=2));observed_zimage=True
        return result
    cls.forward=observed
    if MODEL in ('flux1','qwen21'):
        import inspect
        original_diffuse=cls.diffuse
        signature=inspect.signature(original_diffuse)
        checked=False
        @functools.wraps(original_diffuse)
        def diffuse_observed(self,*args,**kwargs):
            nonlocal checked
            if active_real_request and not checked:
                values=signature.bind(self,*args,**kwargs).arguments
                record=dict(true_cfg=values['do_true_cfg'],steps=len(values['timesteps']),sigmas=self.scheduler.sigmas.detach().cpu().float().tolist())
                assert record['true_cfg'] is False and record['steps']>0,record
                if MODEL=='flux1':
                    record['guidance']=values['guidance'].detach().cpu().float().tolist()
                    assert record['steps']==50 and record['guidance']==[3.5],record
                else:
                    record['true_cfg_scale']=values['true_cfg_scale']
                    assert record['true_cfg_scale']==1.0,record
                OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'denoise-config.json').write_text(json.dumps(record,indent=2));checked=True
            return original_diffuse(self,*args,**kwargs)
        cls.diffuse=diffuse_observed


def install_sglang_denoising(module):
    import torch
    cls=module.DenoisingStage;original=cls._prepare_denoising_loop
    checked=False
    @functools.wraps(original)
    def observed(self,batch,server_args):
        nonlocal checked
        ctx=original(self,batch,server_args)
        if not checked and not batch.is_warmup:
            record=dict(true_cfg=batch.do_classifier_free_guidance,steps=len(ctx.timesteps),requested_steps=batch.num_inference_steps,sigmas=ctx.scheduler.sigmas.detach().cpu().float().tolist())
            assert record['true_cfg'] is False and record['steps']>0,record
            if MODEL=='flux1':
                record.update(embedded_guidance=server_args.pipeline_config.embedded_cfg_scale,guidance=ctx.guidance.detach().cpu().float().tolist())
                assert record['steps']==50 and record['embedded_guidance']==3.5,record
                assert torch.equal(ctx.guidance,torch.full_like(ctx.guidance,3500.0)),record
            elif MODEL=='zimage':
                record['guidance']=batch.guidance_scale
                assert record['steps']==9 and record['guidance']==0.0,record
            else:
                record['true_cfg_scale']=batch.true_cfg_scale
                assert record['true_cfg_scale']==1.0,record
            OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'denoise-config.json').write_text(json.dumps(record,indent=2));checked=True
        return ctx
    cls._prepare_denoising_loop=observed


if FRAMEWORK=='sglang':
    TARGET='sglang.multimodal_gen.runtime.pipelines_core.stages.decoding';CALLBACK=install_sglang
else:
    TARGET='vllm_omni.diffusion.models.'+{'flux1':'flux.pipeline_flux','qwen21':'qwen_image_21.pipeline_qwen_image_21','zimage':'z_image.pipeline_z_image'}[MODEL];CALLBACK=install_omni


TARGETS={TARGET:CALLBACK}
if FRAMEWORK=='sglang':
    TARGETS['sglang.multimodal_gen.runtime.pipelines_core.stages.denoising']=install_sglang_denoising


class ObservingLoader(importlib.abc.Loader):
    def __init__(self,original,callback):self.original=original;self.callback=callback
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module);self.callback(module)


class ObservingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname not in TARGETS:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        assert spec is not None and spec.loader is not None
        spec.loader=ObservingLoader(spec.loader,TARGETS[fullname]);return spec


assert not any(name in sys.modules for name in TARGETS)
sys.meta_path.insert(0,ObservingFinder())
