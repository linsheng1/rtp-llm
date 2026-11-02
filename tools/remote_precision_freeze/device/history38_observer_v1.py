"""D-only real block inputs, all40 layers, two actual GPU positions.

The old latch kernels, request-drained private control and budgets are retained.
P has no hooks or buffers. No model/KV mutation, no tensor CPU read in forward.
"""
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
TARGETS = (BLOCK, ATTN, FRONT, MOE, COMP, GLOBAL, HC, NORM)
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


def block_wrapper(original, mode):
    @functools.wraps(original)
    def call(block, *args, **kwargs):
        if mode != 'D' or block.layer_id not in _capture._config['layers']:
            return original(block,*args,**kwargs)
        previous=getattr(_capture._context,'value',None)
        if previous is not None and previous.get('block') is block:
            return original(block,*args,**kwargs)
        hidden,ids,positions=domain(block,args,kwargs,'D')
        _capture._context.value=dict(role='D',layer=block.layer_id,block=block,attn=block.attn,positions=positions)
        try:
            record('block_entry4HC',hidden,positions)
            if block.layer_id==0:
                need(ids is not None and ids.numel()==positions.numel(),'actual decode token geometry required')
                record('actual_input_token',ids,positions)
            return original(block,*args,**kwargs)
        finally:_capture._context.value=previous
    return call


def install(config):
    global _capture
    need(config['schema']=='history38-inner-graph-latches-v1' and config['layers']==list(range(40))
         and config['positions']==[7169,7200] and config['gpu_budget_bytes']==4194304
         and config['artifact_kind']=='inner_python_overlay' and config['wheel_sha256'] is None,
         'explicit bounded40-layer continuation profile required')
    need(set(config['hook_source_sha256'])=={BLOCK},'only actual block source is hooked')
    if config['role']=='P':return None
    need(config['role']=='D','D-only trajectory profile')
    path=Path(__file__).with_name('history38_latches_v1.py')
    need(pin(path)['SHA256']==config['latches_SHA256'],'latch pin drift')
    spec=importlib.util.spec_from_file_location('_history38_latches',path)
    _capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(_capture)
    _capture._config=config
    save=_capture._save_json
    def save_counters(path,value):
        if value.get('state')=='CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE':
            value['all_registered_hit_counters']={k:hit.cpu().tolist() for k,(_,hit) in _capture._buffers.items()}
            value['input_RID_binding_scope']='controller-exclusive sole RPC + actual GPU positions/token samples; no GPU RID latch'
        save(path,value)
    _capture._save_json=save_counters
    need(BLOCK not in sys.modules,'no hot installation')
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname!=BLOCK:return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            need(spec is not None and spec.loader is not None,'natural source loader required')
            original=spec.loader.exec_module
            def execute(module):
                original(module)
                need(pin(module.__file__)['SHA256']==config['hook_source_sha256'][BLOCK],'actual block source drift')
                _capture._source_identity[BLOCK]=pin(module.__file__)
                module.Block.forward_decode=block_wrapper(module.Block.forward_decode,'D')
            spec.loader.exec_module=execute
            return spec
    sys.meta_path.insert(0,Finder())
    return _capture
