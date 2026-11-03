"""Evidence-only prepared-input and L25 seam capture; unchanged model/graph math."""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

BLOCK = 'rtp_llm.models_py.modules.dsv41.block'
ATTN = 'rtp_llm.models_py.modules.dsv41.fp8.attention_v41'
FRONT = 'rtp_llm.models_py.modules.dsv41.fp8.attention'
MOE = 'rtp_llm.models_py.modules.dsv41.moe.moe_layer'
COMP = 'rtp_llm.models_py.modules.dsv41.fp8.compressor'
GLOBAL = 'rtp_llm.models_py.modules.dsv41.fp8._v41_decode_global'
HC = 'rtp_llm.models_py.modules.dsv41.hc.delayed'
NORM = 'rtp_llm.models_py.modules.base.cuda.norm'
TARGETS = (BLOCK, ATTN, MOE, HC, NORM)
_capture = None
_params = {}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), SHA256=hashlib.sha256(raw).hexdigest())


def rows(value, positions):
    need(value.numel() % positions.numel() == 0, 'actual tensor/position geometry differs')
    return value.reshape(positions.numel(), -1)


def record(label, value, positions, **metadata):
    if hasattr(value, 'device'):
        _capture._capture(label, rows(value, positions), positions, **metadata)


def parameter(name, value):
    if not hasattr(value, 'device'):
        return
    identity = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype))
    old = _params.setdefault(name, (value, identity))
    need(old[0] is value and old[1] == identity, 'actual parameter reference changed: ' + name)


def bind_parameters(attn):
    layer = attn.layer_id
    for name in ['global_wkv', 'global_wgate', 'global_norm', 'kv_norm', 'q_norm']:
        parameter('L%d.%s' % (layer, name), getattr(attn, name, None))
    for name in ['wkv', 'wq_a_wkv']:
        linear = getattr(attn, name, None)
        for field in ['weight', 'weight_scales']:
            parameter('L%d.%s.%s' % (layer, name, field), getattr(linear, field, None))


def domain(block, args, kwargs, mode):
    import torch
    hidden = args[0] if args else kwargs['x']
    if mode == 'D':
        meta = args[1] if len(args) > 1 else kwargs['attn_metadata']
        count = hidden.shape[0] * hidden.shape[1]
        positions = meta.position_ids[:count].reshape(-1)
        valid = getattr(meta, 'valid_requests', None)
        if valid is not None:
            mask = valid[:hidden.shape[0]].reshape(-1, 1).expand(-1, hidden.shape[1]).reshape(-1)
            positions = torch.where(mask, positions, torch.full_like(positions, -1))
        ids = args[2] if len(args) > 2 else kwargs.get('input_ids')
    else:
        cp = getattr(block.attn, '_cp_ctx', None)
        need(cp is not None and cp.cp_size == 4 and cp.kv_cache_sharded,
             'history P requires actual CP4 sharded domain')
        positions = torch.where(cp.local_is_real, cp.global_positions,
                                torch.full_like(cp.global_positions, -1))
        ids = args[1] if len(args) > 1 else kwargs.get('input_ids')
    need(positions.is_cuda and hidden.numel() == positions.numel()*4*5120,
         'actual historical block rows/positions differ')
    return hidden, ids, positions


def block_wrapper(original,mode):
    @functools.wraps(original)
    def call(block,*args,**kwargs):
        if block.layer_id not in _capture._config['layers'] or mode!=_capture._config['role']:
            return original(block,*args,**kwargs)
        previous=getattr(_capture._context,'value',None)
        if previous is not None and previous.get('block') is block:
            return original(block,*args,**kwargs)
        if mode=='P':
            cp=getattr(block.attn,'_cp_ctx',None)
            if cp is None or cp.cp_size!=4 or not cp.kv_cache_sharded:return original(block,*args,**kwargs)
        hidden,ids,positions=domain(block,args,kwargs,mode)
        _capture._context.value=dict(role=mode,layer=block.layer_id,attn=block.attn,block=block,positions=positions)
        try:
            record('block_entry4HC',hidden,positions)
            if block.layer_id==0:
                need(ids is not None and ids.numel()==positions.numel(),'real token geometry required')
                record('actual_input_token',ids,positions)
            if mode=='D':
                import torch
                prepared=kwargs.get('_prepared_attn')
                record('prepared_attn_present',torch.full((positions.numel(),1),int(prepared is not None),device=positions.device,dtype=torch.int8),positions)
                if prepared is not None:
                    residual,x_pre,post,comb=prepared[:4]
                    # forward_decode_layer passes prepared[0] as x. Capture the
                    # other natural values too; residual equality is insufficient.
                    record('prepared_normalized_input5120',x_pre,positions)
                    coeff=torch.cat([rows(block.attn_hc.pre_mix_out,positions),rows(post,positions),rows(comb,positions)],dim=1)
                    need(coeff.shape[1]==24 and coeff.dtype==torch.float32,'actual HC pre/post/comb packing differs')
                    record('prepared_HC_coeff24',coeff,positions)
                if block.layer_id==25:bind_parameters(block.attn)
            return original(block,*args,**kwargs)
        finally:_capture._context.value=previous
    return call


def attention_wrapper(original,method):
    @functools.wraps(original)
    def call(attn,*args,**kwargs):
        context=getattr(_capture._context,'value',None)
        if context is None or context['role']!='D' or attn is not context['attn']:
            return original(attn,*args,**kwargs)
        positions=context['positions']
        if method=='_forward_decode_body':
            record('actual_attention_input5120',args[0] if args else kwargs['x'],positions)
            if attn.layer_id==25:
                import torch
                quant=kwargs.get('shared_input_quant')
                if quant is None and len(args)>2:quant=args[2]
                record('shared_quant_present',torch.full((positions.numel(),1),int(quant is not None),device=positions.device,dtype=torch.int8),positions)
                if quant is not None:
                    for i,value in enumerate(quant):record('actual_shared_input_quant_%d'%i,value,positions)
        result=original(attn,*args,**kwargs)
        if attn.layer_id==25 and method=='_forward_decode_body':record('attn_output5120',result,positions)
        return result
    return call


def seam_context(block=None):
    context = getattr(_capture._context, 'value', None)
    return context if context is not None and context['role']=='D' and context['layer']==25 and (block is None or context['block'] is block) else None


def seam_parameters(block):
    for label, unit in [('attnHC', block.attn_hc), ('FFNHC', block.ffn_hc)]:
        for field in ['fn', 'base', 'scale']:
            parameter('L25.'+label+'.'+field, getattr(unit, field))
    parameter('L25.FFN_norm.weight', block.ffn_norm.weight)


def patch_seam_block(module):
    original_inject = module.Block._inject_engram
    @functools.wraps(original_inject)
    def inject(block, hidden):
        result = original_inject(block, hidden)
        context = seam_context(block)
        if context is not None:
            record('after_Engram4HC', result, context['positions'])
        return result
    module.Block._inject_engram = inject
    original_mega = module.Block._try_mega_mhc
    @functools.wraps(original_mega)
    def mega(block, attn_out, residual, post, comb, **kwargs):
        context = seam_context(block)
        if context is not None:
            seam_parameters(block)
            positions = context['positions']
            import torch
            record('seam_actual_rows',torch.full((positions.numel(),1),residual.numel()//20480,device=positions.device,dtype=torch.int32),positions)
            for label, value in [('seam_residual4HC',residual), ('seam_attn_out5120',attn_out),
                                 ('attnHC_pre4',block.attn_hc.pre_mix_out), ('attnHC_post4',post), ('attnHC_comb16',comb)]:
                record(label, value, positions, registration_shape=list(value.shape))
        result = original_mega(block, attn_out, residual, post, comb, **kwargs)
        if context is not None:
            import torch
            record('seam_fused_route', torch.full((positions.numel(),1),int(result is not None),device=positions.device,dtype=torch.int8), positions)
            if result is not None:
                for label, value in zip(['seam_post4HC','seam_FFN_norm5120','FFNHC_post4','FFNHC_comb16'],result):
                    record(label,value,positions)
                record('FFNHC_pre4',block.ffn_hc.pre_mix_out,positions)
        return result
    module.Block._try_mega_mhc = mega


def patch_seam_hc(module):
    original = module.DelayedHCUnit._pre_impl
    @functools.wraps(original)
    def pre(unit, x, *args, **kwargs):
        context = seam_context()
        label = None
        if context is not None:
            block = context['block']
            label = 'natural_attnHC' if unit is block.attn_hc else 'natural_FFNHC' if unit is block.ffn_hc else None
            if label is not None:
                seam_parameters(block)
                record(label+'_residual4HC',x,context['positions'],registration_shape=list(x.shape))
        result = original(unit,x,*args,**kwargs)
        if label is not None:
            for suffix,value in zip(['output5120','post4','comb16'],result):
                record(label+'_'+suffix,value,context['positions'])
            record(label+'_pre4',unit.pre_mix_out,context['positions'])
        return result
    module.DelayedHCUnit._pre_impl = pre


def patch_seam_norm(module):
    original = module.RMSNorm.forward
    @functools.wraps(original)
    def norm(unit,x,*args,**kwargs):
        context = seam_context()
        active = context is not None and unit is context['block'].ffn_norm
        if active:
            record('natural_FFN_norm_input5120',x,context['positions'])
        result = original(unit,x,*args,**kwargs)
        if active:
            record('natural_FFN_norm_output5120',result,context['positions'])
        return result
    module.RMSNorm.forward = norm

def patch(name,module):
    need(pin(module.__file__)['SHA256']==_capture._config['hook_source_sha256'][name],'actual hook source drift '+name)
    _capture._source_identity[name]=pin(module.__file__)
    if name==BLOCK:
        patch_seam_block(module)
        for method,mode in [('forward','P'),('_forward_prefill_fast_fp8','P'),('forward_decode','D')]:
            setattr(module.Block,method,block_wrapper(getattr(module.Block,method),mode))
    elif name==HC:patch_seam_hc(module)
    elif name==NORM:patch_seam_norm(module)
    elif name==ATTN:
        production=module.AttentionV41FP8._prefill_produce
        @functools.wraps(production)
        def prefill_input(attn,x,positions,*args,**kwargs):
            context=getattr(_capture._context,'value',None)
            if context is not None and context['role']=='P' and attn is context['attn']:
                record('actual_attention_input5120',x,context['positions'])
                if context['layer']==25:bind_parameters(attn)
            return production(attn,x,positions,*args,**kwargs)
        module.AttentionV41FP8._prefill_produce=prefill_input
        module.AttentionV41FP8._forward_decode_body=attention_wrapper(module.AttentionV41FP8._forward_decode_body,'_forward_decode_body')
    elif name==MOE:
        original=module.MoE.forward
        @functools.wraps(original)
        def moe(unit,*args,**kwargs):
            context=seam_context();active=context is not None and unit is context['block'].ffn
            if active:record('FFN_input5120',args[0] if args else kwargs['x'],context['positions'])
            result=original(unit,*args,**kwargs)
            if active:record('FFN_output5120',result,context['positions'])
            return result
        module.MoE.forward=moe

def install(config):
    global _capture
    need(config['schema']=='history41-inner-graph-latches-v1' and config['role'] in ['P','D']
         and config['layers']==list(range(26))
         and config['positions']==[7045]
         and config['artifact_kind']=='inner_python_overlay' and config['wheel_sha256'] is None
         and config['gpu_budget_bytes']==4194304,'explicit prepared-input trajectory profile required')
    need(set(config['hook_source_sha256']) == set(TARGETS), 'all historical source fingerprints required')
    path = Path(__file__).with_name('history38_latches_v1.py')
    need(pin(path)['SHA256']==config['latches_SHA256'], 'existing adapted latch source drift')
    spec = importlib.util.spec_from_file_location('_history41_existing_graph_latches', path)
    _capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_capture)
    _capture._config = config
    save = _capture._save_json
    def save_with_parameters(path, value):
        if value.get('state')=='CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE':
            import torch
            parameters = {}
            total = 0
            for name, (tensor, before) in _params.items():
                after = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
                need(before==after, 'parameter source identity changed at export')
                raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                total += len(raw)
                need(total <= 128*1024**2, 'bounded actual parameter CPU hashes')
                parameters[name] = dict(SHA256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                                        shape=list(tensor.shape), dtype=str(tensor.dtype))
            value['actual_parameters_after_EOF'] = parameters
            value['all_registered_hit_counters'] = {key:hit.cpu().tolist()
                                                    for key,(_,hit) in _capture._buffers.items()}
            value['input_RID_binding_scope'] = 'controller-exclusive sole RPC + actual GPU positions/token samples; no GPU RID latch'
        save(path, value)
    _capture._save_json = save_with_parameters
    need(not any(name in sys.modules for name in TARGETS), 'no hot installation')
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in TARGETS:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            need(spec is not None and spec.loader is not None, 'natural module loader required')
            original = spec.loader.exec_module
            def execute(module):
                original(module)
                patch(fullname, module)
            spec.loader.exec_module = execute
            return spec
    sys.meta_path.insert(0, Finder())
    return _capture
