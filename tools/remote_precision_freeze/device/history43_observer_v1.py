"""Natural L24/P+D Q/core/canonical selected-KV sampling at7045.

Existing graph-safe latches/bounds; P's actual single-request workspace metadata
and D's actual physical page tables are observed, never replaced.
"""
import functools,hashlib,importlib.abc,importlib.machinery,importlib.util,sys
from pathlib import Path
BLOCK='rtp_llm.models_py.modules.dsv41.block';ATTN='rtp_llm.models_py.modules.dsv41.fp8.attention_v41';FRONT='rtp_llm.models_py.modules.dsv41.fp8.attention';QKV='rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv';MEGA='rtp_llm.models_py.modules.dsv41.fp8._v41_mega_attention';UTILS='rtp_llm.models_py.modules.dsv41.utils'
TARGETS=(BLOCK,ATTN,FRONT,QKV,MEGA,UTILS);_capture=None;_params={}
def need(x,m):
 if not x:raise RuntimeError(m)
def pin(p):p=Path(p);return dict(path=str(p.resolve()),SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
def context():return getattr(_capture._context,'value',None)
def record(label,value,positions=None,**meta):
 c=context()
 if c is None or not hasattr(value,'device'):return
 p=c['positions'] if positions is None else positions;need(value.numel()%p.numel()==0,'actual tensor/position shape');_capture._capture(label,value.reshape(p.numel(),-1),p,registration_shape=list(value.shape),**meta)
def parameter(n,t):
 if not hasattr(t,'device'):return
 ident=(t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype));old=_params.setdefault(n,(t,ident));need(old[0] is t and old[1]==ident,'actual parameter changed '+n)
def selected_prefill(kv,indices,lengths,positions,meta):
 import torch
 need(kv.dtype==torch.bfloat16 and kv.ndim==3 and kv.shape[1:]==(1,512),'actual BF16 prefill workspace');need(isinstance(meta,tuple) and len(meta)==3 and all(type(x)is int for x in meta),'actual single-request chunk metadata')
 n=positions.numel();need(indices.shape[0]==n and indices.dtype==torch.int32 and 0<indices.shape[-1]<=640,'actual prefill sparse addresses');mask=positions.reshape(-1)==7045;row=mask.long().argmax().reshape(1);one=torch.where(mask.any(),positions.new_tensor(7045),positions.new_tensor(-1)).reshape(1);ids=indices.index_select(0,row).reshape(1,-1);k=ids.shape[1];size=ids.new_full((1,1),k) if lengths is None else lengths.index_select(0,row).reshape(1,1);valid=(torch.arange(k,device=kv.device).reshape(1,k)<size)&(ids>=0);inside=(ids>=0)&(ids<kv.shape[0]);safe=ids.clamp(0,kv.shape[0]-1).long().flatten();values=kv.index_select(0,safe).reshape(k,512);values=torch.where((valid&inside).flatten()[:,None],values,torch.zeros_like(values));offset,global_rows,swa_start=meta;kind=torch.where(ids<offset+global_rows,torch.zeros_like(ids),torch.ones_like(ids));canon=torch.where(kind==0,ids-offset,ids-offset-global_rows+swa_start);canon=torch.where(valid&inside,canon,torch.full_like(canon,-1));kind=torch.where(valid&inside,kind,torch.full_like(kind,-1))
 for name,t in [('prefill_indices',ids),('prefill_lengths',size),('prefill_valid',valid&inside),('prefill_mapping_valid',(~valid)|(inside&(ids>=offset))),('prefill_kind',kind),('prefill_canonical',canon)]:record(name,t,one)
 record('prefill_chunk_meta',ids.new_tensor(meta).reshape(1,3),one)
 for j,start in enumerate(range(0,k,128)):record('prefill_selected_KV_chunk'+str(j),values[start:start+128].reshape(1,-1),one)
def selected_decode(label,pool,slots,table,logical_entries,lengths=None):
 import torch
 c=context();n=c['positions'].numel();ids=slots.reshape(n,-1).long();valid=ids>=0
 if lengths is not None:valid=valid&(torch.arange(ids.shape[1],device=pool.device).reshape(1,-1)<lengths.reshape(n,1));record(label+'_lengths',lengths)
 inside=(ids>=0)&(ids<pool.shape[0]*pool.shape[1]);safe=ids.clamp(0,pool.shape[0]*pool.shape[1]-1);pages=safe//pool.shape[1];offset=safe%pool.shape[1];req=c['metadata'].req_id_per_token[:n].long().clamp_min(0);rows=table.index_select(0,req).long();match=(pages[:,:,None]==rows[:,None,:])&(rows[:,None,:]>=0);unique=match.sum(-1)==1;canon=match.long().argmax(-1)*logical_entries+offset;mapping=(~valid)|(inside&unique&(offset<logical_entries));raw=pool[pages,offset];raw=torch.where((valid&inside)[:,:,None],raw,torch.zeros_like(raw))
 for name,t in [('raw',raw),('physical_slots',ids),('inverse_coordinates',torch.where(valid&mapping,canon,torch.full_like(canon,-1))),('valid',valid),('mapping_valid',mapping),('request_block_table',rows)]:record(label+'_'+name,t)
 if label=='SWA':
  width=ids.shape[1];absolute=c['positions'].reshape(n,1)-width+1+torch.arange(width,device=pool.device).reshape(1,width);col=(absolute//logical_entries).clamp(0,rows.shape[1]-1);page=rows.gather(1,col.long());expected=page*pool.shape[1]+absolute%pool.shape[1];ok=(~valid)|((absolute>=0)&(expected==ids));record('SWA_window_mapping_valid',ok);record('SWA_canonical',torch.where(valid&ok,absolute,torch.full_like(absolute,-1)))
 else:record('GLOBAL_canonical',torch.where(valid&mapping,canon,torch.full_like(canon,-1)))
def block_wrapper(original,role):
 @functools.wraps(original)
 def call(block,*a,**kw):
  if block.layer_id!=24 or role!=_capture._config['role']:return original(block,*a,**kw)
  old=context()
  if old is not None and old.get('block') is block:return original(block,*a,**kw)
  import torch
  x=a[0]
  if role=='P':
   cp=getattr(block.attn,'_cp_ctx',None)
   if cp is None or cp.cp_size!=4 or not cp.kv_cache_sharded or len(cp.input_lengths_global_host)!=1:return original(block,*a,**kw)
   p=torch.where(cp.local_is_real,cp.global_positions,torch.full_like(cp.global_positions,-1));ids=a[1];metadata=None
  else:
   metadata=a[1];n=x.shape[0]*x.shape[1];p=metadata.position_ids[:n].reshape(-1);valid=metadata.valid_requests[:x.shape[0]].reshape(-1,1).expand(-1,x.shape[1]).reshape(-1);p=torch.where(valid,p,torch.full_like(p,-1));ids=a[2]
  need(x.numel()==p.numel()*20480,'actual role domain');attn=block.attn;_capture._context.value=dict(role=role,block=block,attn=attn,positions=p,metadata=metadata)
  try:
   record('block_entry4HC',x);record('actual_input_token',ids);topology=[attn.tp_size,attn.n_heads,attn.compress_ratio,attn.kv_source_layer_id,attn.index_source_layer_id,int(attn.is_kv_source),int(attn.is_index_source)];record('actual_topology',torch.tensor(topology,device=x.device,dtype=torch.int32).reshape(1,-1).expand(p.numel(),-1))
   for name in ['wq_a_wkv','wkv']:
    unit=getattr(attn,name,None)
    for field in ['weight','weight_scales']:parameter(name+'.'+field,getattr(unit,field,None))
   for name in ['kv_norm','q_norm']:parameter(name,getattr(attn,name,None))
   return original(block,*a,**kw)
  finally:_capture._context.value=old
 return call
def mega_wrapper(original,method):
 @functools.wraps(original)
 def call(mega,*a,**kw):
  c=context()
  if c is None or mega is not getattr(c['attn'],'_mega_attention',None):return original(mega,*a,**kw)
  for name in ['query_weight','query_scale','output_weight','output_scale','sink']:parameter('mega.'+name,getattr(mega,name))
  if method=='project_q':
   record('Qr_before_Q_projection',a[1]);quant=kw.get('qr_quantized',a[2] if len(a)>2 else None)
   if quant is not None:
    for i,t in enumerate(quant):record('Qr_actual_quant_'+str(i),t)
   c['QR_quantization_active']=True
   try:result=original(mega,*a,**kw)
   finally:c['QR_quantization_active']=False
   record('Q_after_projection',result);return result
  if method in ['prefill','decode']:
   import torch
   q=a[0];positions=a[3];need(q.shape[0]==c['positions'].numel() and positions.numel()==q.shape[0],'actual one natural query chunk');record('Q_before_core',q);record('core_positions',positions);record('sink',mega.sink.reshape(1,-1).expand(q.shape[0],-1));record('cos_sin_position_row',mega.cos_sin.index_select(0,positions.long().reshape(-1).clamp_min(0)));record('softmax_scale',torch.tensor(float(mega.softmax_scale),device=q.device,dtype=torch.float64).reshape(1,1).expand(q.shape[0],1));record('core_branch',torch.full_like(c['positions'],1 if method=='decode' else 2,dtype=torch.int8))
   if method=='prefill':selected_prefill(a[1],a[2],a[4] if len(a)>4 else kw.get('topk_length'),c['positions'],c['prefill_chunk_meta'])
   else:
    from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
    from rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils import require_pool_tokens_per_block
    attn=c['attn'];meta=c['metadata'];selected_decode('SWA',a[1],a[2],meta.pool_block_tables[SWA_KV],require_pool_tokens_per_block(attn._kv_cache,region=SWA_KV),kw.get('swa_lengths'));region=attn._global_region();need(kw.get('extra_pool') is not None and kw.get('extra_indices') is not None,'actual dual pool');selected_decode('GLOBAL',kw['extra_pool'],kw['extra_indices'],attn._block_tables_by_type[region],require_pool_tokens_per_block(attn._kv_cache,region=region)//attn.compress_ratio,kw.get('extra_lengths'))
   result=original(mega,*a,**kw);record('core_output',result[0]);record('core_scales',result[1]);return result
  result=original(mega,*a,**kw);record('output_projection5120',result);return result
 return call
def patch(name,m):
 need(pin(m.__file__)['SHA256']==_capture._config['hook_source_sha256'][name],'actual production source drift '+name);_capture._source_identity[name]=pin(m.__file__)
 if name==BLOCK:
  for method,role in [('forward','P'),('_forward_prefill_fast_fp8','P'),('forward_decode','D')]:setattr(m.Block,method,block_wrapper(getattr(m.Block,method),role))
 elif name==MEGA:
  for method in ['project_q','prefill','decode','project_output']:setattr(m.MegaAttention,method,mega_wrapper(getattr(m.MegaAttention,method),method))
 elif name==UTILS:
  original=m.V41MXFP8Linear._quantize_input
  @functools.wraps(original)
  def quant(unit,x):
   result=original(unit,x);c=context()
   if c is not None and unit is c['attn'].wq_b and c.get('QR_quantization_active'):
    for i,t in enumerate(result):record('Qr_actual_quant_'+str(i),t)
   return result
  m.V41MXFP8Linear._quantize_input=quant
 elif name==ATTN:
  for method,role in [('_prefill_produce','P'),('_forward_decode_body','D')]:
   def wrap(original,role):
    @functools.wraps(original)
    def body(attn,*a,**kw):
     c=context();active=c is not None and c['role']==role and attn is c['attn']
     if active:
      record('actual_attention_input5120',a[0]);q=kw.get('shared_input_quant',a[2] if len(a)>2 else None)
      if q is not None:
       for i,t in enumerate(q):record('actual_shared_input_quant_'+str(i),t)
     result=original(attn,*a,**kw)
     if active and role=='D':record('attention_output5120',result)
     return result
    return body
   setattr(m.AttentionV41FP8,method,wrap(getattr(m.AttentionV41FP8,method),role))
  original=m.AttentionV41FP8._prefill_sparse_attention
  @functools.wraps(original)
  def sparse(attn,qkv,common,*a,**kw):
   c=context();active=c is not None and c['role']=='P' and attn is c['attn']
   if active:
    need(common.cp_on and common.cp_ctx is attn._cp_ctx and qkv.qr.shape[0]==c['positions'].numel() and qkv.qr.shape[0]<=sys.modules[FRONT]._FLASH_MLA_SPARSE_Q_CHUNK,'actual single CP query chunk');c['prefill_chunk_meta']=attn._shared_attention.get('prefill_chunk_meta')
   result=original(attn,qkv,common,*a,**kw)
   if active:record('attention_output5120',result);c.pop('prefill_chunk_meta')
   return result
  m.AttentionV41FP8._prefill_sparse_attention=sparse
  original_ar=m.AttentionV41FP8._prefill_output_all_reduce
  @functools.wraps(original_ar)
  def ar(attn,out):
   c=context();active=c is not None and attn is c['attn']
   if active:record('pre_AR5120',out)
   result=original_ar(attn,out)
   if active:record('post_AR5120',out)
   return result
  m.AttentionV41FP8._prefill_output_all_reduce=ar
 elif name==QKV:
  original=m.decode_compute_qkv
  @functools.wraps(original)
  def qkv(attn,*a,**kw):
   result=original(attn,*a,**kw);c=context()
   if c is not None and attn is c['attn']:
    record('Qr_from_qkv',result.qr);record('new_KV_from_qkv',result.kv)
   return result
  m.decode_compute_qkv=qkv

def install(config):
 global _capture
 need(config['schema']=='history43-inner-graph-latches-v1' and config['layers']==[24] and config['positions']==[7045] and config['role'] in ['P','D'] and config['gpu_budget_bytes']==4194304 and config['artifact_kind']=='inner_python_overlay' and config['wheel_sha256'] is None,'explicit actual Q/KV/core profile');need(set(config['hook_source_sha256'])==set(TARGETS),'complete natural source pins');p=Path(__file__).with_name('history38_latches_v1.py');need(pin(p)['SHA256']==config['latches_SHA256'],'unchanged latch source');s=importlib.util.spec_from_file_location('_history43_latches',p);_capture=importlib.util.module_from_spec(s);s.loader.exec_module(_capture);_capture._config=config;save=_capture._save_json
 def with_parameters(p,v):
  if v.get('state')=='CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE':
   import torch
   out={};total=0
   for n,(t,ident) in _params.items():
    need((t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype))==ident,'actual parameter changed atEOF');raw=t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();total+=len(raw);need(total<=128*1024**2,'original parameter hash budget');out[n]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),shape=list(t.shape),dtype=str(t.dtype))
   v['actual_parameters_after_EOF']=out;v['all_registered_hit_counters']={k:h.cpu().tolist() for k,(_,h) in _capture._buffers.items()};v['input_RID_binding_scope']='sole actualnativeRID plus GPUtoken/positions; no GPU RID latch'
  save(p,v)
 _capture._save_json=with_parameters;need(not any(n in sys.modules for n in TARGETS),'no hot installation')
 class Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname not in TARGETS:return None
   s=importlib.machinery.PathFinder.find_spec(fullname,path);need(s is not None and s.loader is not None,'natural loader');original=s.loader.exec_module
   def execute(m):original(m);patch(fullname,m)
   s.loader.exec_module=execute;return s
 sys.meta_path.insert(0,Finder());return _capture
