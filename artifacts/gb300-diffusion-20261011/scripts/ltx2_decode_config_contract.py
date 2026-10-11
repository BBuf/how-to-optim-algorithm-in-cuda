"""Record actual decode precision/tiling, without changing arithmetic."""
import json
import ltx2_benchmark_contract as contract


def install(module):
    import torch
    cls=module.LTX2AVDecodingStage
    forward=cls.forward
    def observed(self,batch,server_args):
        if not getattr(self,'_bench_decode_config_installed',False):
            decode=self.vae.decode
            def checked(*args,**kwargs):
                record=dict(disable_autocast=server_args.disable_autocast,autocast_active=torch.is_autocast_enabled('cuda'),input_dtype=str(args[0].dtype),input_shape=list(args[0].shape),parameter_dtype=str(next(self.vae.parameters()).dtype),use_tiling=self.vae.use_tiling,use_framewise_decoding=self.vae.use_framewise_decoding)
                assert record['disable_autocast'] and not record['autocast_active'] and not record['use_tiling'] and not record['use_framewise_decoding'],record
                contract.OUTPUT.mkdir(parents=True,exist_ok=True)
                (contract.OUTPUT/'decode-config.json').write_text(json.dumps(record,indent=2))
                return decode(*args,**kwargs)
            self.vae.decode=checked
            self._bench_decode_config_installed=True
        return forward(self,batch,server_args)
    cls.forward=observed


contract.RECIPE=dict(contract.RECIPE,decode_tiling=False,sglang_disable_autocast=True,status='v2 matched full-volume BF16 decode; earlier defaults mixed tiling/autocast')
contract.TARGETS['sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.decoding_av']=install


def install_omni(module):
    import torch
    cls=module.LTXRuntime
    forward=cls._decode_output
    def observed(self,*args,**kwargs):
        if not getattr(self,'_bench_decode_config_installed',False):
            decode=self.vae.decode
            def checked(latents,*a,**kw):
                record=dict(autocast_active=torch.is_autocast_enabled('cuda'),input_dtype=str(latents.dtype),input_shape=list(latents.shape),parameter_dtype=str(next(self.vae.parameters()).dtype),use_tiling=self.vae.use_tiling,use_framewise_decoding=self.vae.use_framewise_decoding)
                assert not record['autocast_active'] and not record['use_tiling'] and not record['use_framewise_decoding'],record
                assert latents.dtype==next(self.vae.parameters()).dtype==torch.bfloat16,record
                contract.OUTPUT.mkdir(parents=True,exist_ok=True)
                (contract.OUTPUT/'decode-config.json').write_text(json.dumps(record,indent=2))
                return decode(latents,*a,**kw)
            self.vae.decode=checked
            self._bench_decode_config_installed=True
        return forward(self,*args,**kwargs)
    cls._decode_output=observed


if contract.os.environ['LTX2_BENCH_FRAMEWORK']=='vllm':
    contract.TARGETS['vllm_omni.diffusion.models.ltx2.ltx2_runtime']=install_omni
