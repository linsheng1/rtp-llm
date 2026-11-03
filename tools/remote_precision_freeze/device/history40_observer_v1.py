"""Natural L25 decode Q/core/selected-KV samples; no numerical replacement.

Actual topology L25/TP1/64heads/ratio1/KVsource20/indexsource24 is mandatory.
Only row7169 is retained. Original graph-safe4MiB/192object latches are reused.
"""
import functools,hashlib,importlib.abc,importlib.machinery,importlib.util,json,os,sys
from pathlib import Path
BLOCK='rtp_llm.models_py.modules.dsv41.block'
ATTN='rtp_llm.models_py.modules.dsv41.fp8.attention_v41'
FRONT='rtp_llm.models_py.modules.dsv41.fp8.attention'
QKV='rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv'
MEGA='rtp_llm.models_py.modules.dsv41.fp8._v41_mega_attention'
SPLIT='rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_decode_attn'
TARGETS=(BLOCK,ATTN,FRONT,QKV,MEGA,SPLIT)
_capture=None
_params={}
def need(ok,why):
 if not ok:raise RuntimeError(why)
def pin(p):p=Path(p);return dict(path=str(p.resolve()),SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
def context():return getattr(_capture._context,'value',None)
def record(label,value,**metadata):
 c=context()
 if c is None or not hasattr(value,'device'):return
 p=c['positions'];need(value.numel()%p.numel()==0,'actual sample geometry')
 _capture._capture(label,value.reshape(p.numel(),-1),p,registration_shape=list(value.shape),**metadata)
def parameter(label,t):
 if not hasattr(t,'device'):return
 ident=(t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype));old=_params.setdefault(label,(t,ident));need(old[0] is t and old[1]==ident,'parameter reference changed '+label)
def selected_pool(label,pool,slots,table,logical_entries):
 import torch
 c=context();count=c['positions'].numel();indices=slots.reshape(count,-1).long();need(pool.dtype==torch.uint8 and pool.ndim==3 and indices.shape[1]<=640,'bounded actual byte pool/readset')
 valid=indices>=0;inside=(indices>=0)&(indices<pool.shape[0]*pool.shape[1]);safe=indices.clamp(0,pool.shape[0]*pool.shape[1]-1)
 pages=safe//pool.shape[1];offset=safe%pool.shape[1]
 req=c['metadata'].req_id_per_token[:count].long().clamp_min(0);rows=table.index_select(0,req).long();record(label+'_request_block_table',rows)
 matches=(pages[:,:,None]==rows[:,None,:])&(rows[:,None,:]>=0);owners=matches.sum(-1);logical=matches.long().argmax(-1)*logical_entries+offset;unique=(owners==1)&(offset<logical_entries)&inside
 raw=pool[pages,offset];raw=torch.where((valid&inside)[:,:,None],raw,torch.zeros_like(raw))
 record(label+'_raw',raw,pool_shape=list(pool.shape),pool_stride=list(pool.stride()),logical_entries_per_block=logical_entries,record_bytes=pool.shape[-1])
 record(label+'_physical_slots',indices);record(label+'_logical_indices',torch.where(valid&unique,logical,torch.full_like(logical,-1)))
 record(label+'_valid',valid);record(label+'_mapping_valid',(~valid)|(unique&inside))
def block_wrapper(original):
 @functools.wraps(original)
 def call(block,*args,**kwargs):
  if block.layer_id!=25:return original(block,*args,**kwargs)
  import torch
  x=args[0];m=args[1];ids=args[2];attn=block.attn
  need((attn.tp_size,attn.n_heads,attn.compress_ratio,attn.kv_source_layer_id,attn.index_source_layer_id)==(1,64,1,20,24),'actual target topology differs')
  count=x.shape[0]*x.shape[1];p=m.position_ids[:count].reshape(-1);valid=m.valid_requests[:x.shape[0]].reshape(-1,1).expand(-1,x.shape[1]).reshape(-1);p=torch.where(valid,p,torch.full_like(p,-1))
  previous=context();_capture._context.value=dict(role='D',layer=25,block=block,attn=attn,metadata=m,positions=p)
  try:
   record('actual_input_token',ids);record('block_entry4HC',x);record('request_start_pos',m.start_pos[:x.shape[0]].repeat_interleave(x.shape[1]));lengths=getattr(m,'input_lengths',None)
   if hasattr(lengths,'device'):record('request_input_lengths',lengths[:x.shape[0]].repeat_interleave(x.shape[1]))
   for name in ['wq_a','wq_a_wkv','wq_b','wo_b']:
    unit=getattr(attn,name,None)
    for field in ['weight','weight_scales']:parameter(name+'.'+field,getattr(unit,field,None))
   return original(block,*args,**kwargs)
  finally:_capture._context.value=previous
 return call

def mega_wrapper(original,method):
 @functools.wraps(original)
 def call(mega,*args,**kwargs):
  c=context();active=c is not None and mega is getattr(c['attn'],'_mega_attention',None)
  if not active:return original(mega,*args,**kwargs)
  if method=='project_q':
   record('Qr_before_Q_projection',args[1]);result=original(mega,*args,**kwargs);record('Q_after_projection',result);return result
  if method=='decode':
   import torch
   from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
   from rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils import require_pool_tokens_per_block
   q,pool,slots,positions=args[:4];attn=c['attn'];meta=c['metadata'];record('core_branch',torch.ones_like(c['positions'],dtype=torch.int8));record('Q_before_core',q);record('core_positions',positions)
   record('swa_lengths',kwargs.get('swa_lengths'));record('global_lengths',kwargs.get('extra_lengths'))
   selected_pool('SWA',pool,slots,meta.pool_block_tables[SWA_KV],require_pool_tokens_per_block(attn._kv_cache,region=SWA_KV))
   region=attn._global_region();extra=kwargs.get('extra_pool');ei=kwargs.get('extra_indices');need(extra is not None and ei is not None,'actual L25 global input missing')
   selected_pool('GLOBAL',extra,ei,attn._block_tables_by_type[region],require_pool_tokens_per_block(attn._kv_cache,region=region)//attn.compress_ratio)
   for name in ['query_weight','query_scale','output_weight','output_scale','cos_sin','sink']:parameter('mega.'+name,getattr(mega,name))
   record('sink',mega.sink.reshape(1,-1).expand(q.shape[0],-1));record('cos_sin_position_row',mega.cos_sin.index_select(0,positions.reshape(-1).long().clamp_min(0)))
   result=original(mega,*args,**kwargs);record('core_output',result[0]);record('core_scales',result[1]);return result
  result=original(mega,*args,**kwargs)
  if method=='project_output':record('output_projection5120',result)
  return result
 return call

def patch(name,module):
 need(pin(module.__file__)['SHA256']==_capture._config['hook_source_sha256'][name],'actual source drift '+name);_capture._source_identity[name]=pin(module.__file__)
 if name==BLOCK:module.Block.forward_decode=block_wrapper(module.Block.forward_decode)
 elif name==QKV:
  original=module.decode_compute_qkv
  @functools.wraps(original)
  def qkv(attn,*a,**kw):
   c=context();active=c is not None and attn is c['attn'];result=original(attn,*a,**kw)
   if active:
    record('Qr_from_qkv',result.qr);record('Q_from_qkv',result.q);record('new_KV_from_qkv',result.kv)
    if result.qr_quantized is not None:
     for i,v in enumerate(result.qr_quantized):record('Qr_quantized_%d'%i,v)
   return result
  module.decode_compute_qkv=qkv
 elif name==MEGA:
  for method in ['project_q','decode','project_output']:setattr(module.MegaAttention,method,mega_wrapper(getattr(module.MegaAttention,method),method))
 elif name==ATTN:
  original=module.AttentionV41FP8._forward_decode_body
  @functools.wraps(original)
  def body(attn,*a,**kw):
   c=context();active=c is not None and attn is c['attn']
   if active:
    record('actual_attention_input5120',a[0]);q=kw.get('shared_input_quant',a[2] if len(a)>2 else None)
    if q is not None:
     for i,v in enumerate(q):record('actual_shared_input_quant_%d'%i,v)
   result=original(attn,*a,**kw)
   if active:record('attention_output5120',result)
   return result
  module.AttentionV41FP8._forward_decode_body=body
  select=module.AttentionV41FP8._select_indices_decode
  @functools.wraps(select)
  def indices(attn,*a,**kw):
   result=select(attn,*a,**kw)
   if context() is not None and attn is context()['attn']:record('selected_local_indices',result)
   return result
  module.AttentionV41FP8._select_indices_decode=indices
  ar=module.AttentionV41FP8._prefill_output_all_reduce
  need(pin(ar.__code__.co_filename)['SHA256']==_capture._config['hook_source_sha256'][FRONT],'actual inherited reduction source differs')
  @functools.wraps(ar)
  def reduce(attn,out):
   active=context() is not None and attn is context()['attn']
   if active:record('pre_AR5120',out)
   result=ar(attn,out)
   if active:record('post_AR5120',out)
   return result
  module.AttentionV41FP8._prefill_output_all_reduce=reduce
 # SPLIT is fingerprinted and left unchanged. Missing mega core evidence blocks
 # analysis rather than silently claiming a different branch was sampled.

def install(config):
 global _capture
 need(config['schema']=='history40-inner-graph-latches-v1' and config['layers']==[25] and config['positions']==[7169] and config['gpu_budget_bytes']==4194304 and config['artifact_kind']=='inner_python_overlay' and config['wheel_sha256'] is None,'explicit L25 profile only');need(set(config['hook_source_sha256'])==set(TARGETS),'complete source pins')
 if config['role']=='P':return None
 need(config['role']=='D','D-only core profile')
 p=Path(__file__).with_name('history38_latches_v1.py');need(pin(p)['SHA256']==config['latches_SHA256'],'latch source drift');spec=importlib.util.spec_from_file_location('_history40_latches',p);_capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(_capture);_capture._config=config;save=_capture._save_json
 def save_parameters(p,v):
  if v.get('state')=='CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE':
   import torch
   out={};total=0
   for n,(t,before) in _params.items():
    after=(t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype));need(before==after,'parameter identity changed at EOF');raw=t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();total+=len(raw);need(total<=128*1024**2,'original parameter hash budget');out[n]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),shape=list(t.shape),dtype=str(t.dtype))
   v['actual_parameters_after_EOF']=out;v['all_registered_hit_counters']={k:h.cpu().tolist() for k,(_,h) in _capture._buffers.items()};v['input_RID_binding_scope']='controller-exclusive sole native RID plus actual GPU token/position; no GPU RID latch'
  save(p,v)
 _capture._save_json=save_parameters;need(not any(n in sys.modules for n in TARGETS),'no hot installation')
 class Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname not in TARGETS:return None
   s=importlib.machinery.PathFinder.find_spec(fullname,path);need(s is not None and s.loader is not None,'natural loader');original=s.loader.exec_module
   def execute(m):original(m);patch(fullname,m)
   s.loader.exec_module=execute;return s
 sys.meta_path.insert(0,Finder());return _capture
