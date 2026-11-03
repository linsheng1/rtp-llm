"""Extend the calibrated L24 observer with actual indexer scorer inputs/scores.

No math/result/selection replacement. Base44 latches and budgets stay unchanged.
The fixed7046-key prefix is the causal extent of the sole target7045 query.
"""
import functools,hashlib,importlib,importlib.util,sys
from pathlib import Path
BASE='history44_observer_v1.py'
P='rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer'
D='rtp_llm.models_py.modules.dsv41.fp8._v41_decode_indexer'
S='rtp_llm.models_py.modules.dsv41.fp8._indexer_score'
_base=None;_capture=None

def need(ok,why):
 if not ok:raise RuntimeError(why)
def context():return _base.context()
def record(*a,**kw):return _base.record(*a,**kw)
def active(role=None):
 c=context();return c if c is not None and c.get('indexer_active') and (role is None or c['role']==role) else None

def keys_record(raw,valid,p):
 import torch
 need(raw.shape==(7046,68) and valid.numel()==7046,'fixed actual indexer causal prefix')
 mask=p.reshape(-1)==7045;one=torch.where(mask.any(),torch.full((1,),7045,device=p.device,dtype=p.dtype),torch.full((1,),-1,device=p.device,dtype=p.dtype))
 record('indexer_key_valid',valid.reshape(1,-1),one)
 for j,start in enumerate(range(0,7046,2048)):record('indexer_key_raw_chunk'+str(j),raw[start:start+2048].reshape(1,-1),one)

def query_record(q,sf,w,p):
 record('indexer_q_payload',q.reshape(p.numel(),-1),p);record('indexer_q_packed_sf',sf.reshape(p.numel(),-1),p);record('indexer_scaled_head_weights',w.reshape(p.numel(),-1),p)

def prefill_score_wrapper(original):
 @functools.wraps(original)
 def call(q,sf,k,ks,w,visible,*a,**kw):
  c=active('P')
  if c is not None:
   import torch
   need(c['attn'].compress_ratio==1,'actual L24 ratio1 scorer');p=visible.reshape(-1)-1;need(q.shape[0]==p.numel() and k.shape[1]==64 and ks.numel()==k.shape[0],'actual P scorer geometry')
   query_record(q,sf,w,p);record('indexer_visible',visible,p);membership=(c['positions'].reshape(-1,1)==p.reshape(1,-1)).any(0);record('indexer_position_in_actual_block',membership,p)
   ids=torch.arange(7046,device=k.device);valid=ids<k.shape[0];safe=ids.clamp_max(k.shape[0]-1);raw=torch.cat([k.index_select(0,safe).contiguous().view(torch.uint8),ks.reshape(-1).index_select(0,safe).contiguous().view(torch.uint8).reshape(7046,4)],-1);keys_record(raw,valid,p)
  result=original(q,sf,k,ks,w,visible,*a,**kw)
  if c is not None:
   need(result.ndim==2 and result.shape[0]==p.numel(),'actual P score rows');part=result[:,:7046]
   if part.shape[1]<7046:part=torch.nn.functional.pad(part,(0,7046-part.shape[1]),value=-float('inf'))
   record('indexer_logits',part,p)
  return result
 return call

def paged_score_wrapper(original):
 @functools.wraps(original)
 def call(q,sf,pool,w,table,lengths,block_size,capacity,*a,**kw):
  c=active('D')
  if c is not None:
   import torch
   need(c['attn'].compress_ratio==1 and pool.shape[1]==block_size and pool.shape[2]==68,'actual L24 planar indexer pool')
   p=lengths.reshape(-1)-1;need(p.numel()==c['positions'].numel(),'actual D scorer/query domain');p=torch.where(c['positions']>=0,p,torch.full_like(p,-1));query_record(q,sf,w,p);record('indexer_visible',lengths.reshape(-1),p);record('indexer_position_in_actual_block',p==c['positions'],p)
   row=(p==7045).long().argmax().reshape(1);nq=lengths.shape[-1] if lengths.ndim>1 else 1;request=(row//nq).clamp_max(table.shape[0]-1);t=table.index_select(0,request).reshape(-1);ids=torch.arange(7046,device=pool.device);column=ids//block_size;page=t.index_select(0,column.clamp_max(t.numel()-1)).long();inside=(column<t.numel())&(page>0)&(page<pool.shape[0]);safe=page.clamp(0,pool.shape[0]-1);offset=ids%block_size;flat=pool.reshape(pool.shape[0],block_size*68);payload=flat[safe[:,None],offset[:,None]*64+torch.arange(64,device=pool.device).reshape(1,64)];scales=flat[safe[:,None],block_size*64+offset[:,None]*4+torch.arange(4,device=pool.device).reshape(1,4)];keys_record(torch.cat([payload,scales],-1),inside,p)
  return original(q,sf,pool,w,table,lengths,block_size,capacity,*a,**kw)
 return call

def decode_score_wrapper(original):
 @functools.wraps(original)
 def call(*a,**kw):
  c=active('D');result=original(*a,**kw)
  if c is not None:
   need(result is not None and result.ndim==2 and result.shape[0]==c['positions'].numel() and result.shape[1]>=7046,'actual D logical logits');record('indexer_logits',result[:,:7046],c['positions'])
  return result
 return call

def selection_wrapper(original,role):
 @functools.wraps(original)
 def call(attn,*a,**kw):
  c=context();on=c is not None and c['role']==role and c['attn'] is attn
  if not on:return original(attn,*a,**kw)
  cfg=getattr(attn,'v41_config',{});need([cfg.get(k) for k in ['candidate_source_layer_id','candidate_topk_blocks','candidate_block_size']]==[20,2048,8],'actual bounded dense indexer profile');need(attn._shared_attention.get('candidates') is None,'actual capture scope has no candidate mask');old=c.get('indexer_active',False);c['indexer_active']=True
  for name in ['index_wq']:
   unit=getattr(attn,name,None)
   for field in ['weight','weight_scales']:_base.parameter(name+'.'+field,getattr(unit,field,None))
  _base.parameter('index_weights',getattr(attn,'index_weights',None))
  try:
   result=original(attn,*a,**kw);record('indexer_selected_logical',result);return result
  finally:c['indexer_active']=old
 return call

def install(config):
 global _base,_capture,block_wrapper,mega_wrapper
 extension=config['indexer_extension'];need(extension['revision']==45 and extension['causal_prefix']==7046,'explicit indexer-only extension')
 here=Path(__file__).parent;p=here/BASE;need(hashlib.sha256(p.read_bytes()).hexdigest()==extension['base44_SHA256'],'calibrated base observer source')
 spec=importlib.util.spec_from_file_location('_base44_for_indexer45',p);_base=importlib.util.module_from_spec(spec);spec.loader.exec_module(_base);original_patch=_base.patch
 def patch(name,m):
  original_patch(name,m)
  if name==_base.ATTN:
   # Loaded before model construction; source pins checked before wrapping.
   modules={}
   for fullname in [P,D,S]:
    module=importlib.import_module(fullname);need(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==extension['source_sha256'][fullname],'actual indexer source drift '+fullname);modules[fullname]=module;_capture._source_identity[fullname]=_base.pin(module.__file__)
   modules[P].score_indexer_chunk=prefill_score_wrapper(modules[P].score_indexer_chunk)
   modules[D].score_decode_indexer=decode_score_wrapper(modules[D].score_decode_indexer)
   modules[S].fp8_fp4_paged_indexer_score=paged_score_wrapper(modules[S].fp8_fp4_paged_indexer_score)
   m.AttentionV41FP8._select_indices=selection_wrapper(m.AttentionV41FP8._select_indices,'P');m.AttentionV41FP8._select_indices_decode=selection_wrapper(m.AttentionV41FP8._select_indices_decode,'D')
  elif name==_base.UTILS:
   original=m.V41MXFP8Linear.forward
   @functools.wraps(original)
   def forward(unit,*a,**kw):
    result=original(unit,*a,**kw);c=active('P')
    if c is not None and unit is c['attn'].index_wq:record('indexer_raw_query_projection',result)
    return result
   m.V41MXFP8Linear.forward=forward
  elif name==_base.QKV:
   original=m.decode_compute_qkv
   @functools.wraps(original)
   def qkv(*a,**kw):
    result=original(*a,**kw);c=context()
    if c is not None:
     q=getattr(result,'index_q',None)
     if hasattr(q,'device'):record('indexer_raw_query_projection',q)
    return result
   m.decode_compute_qkv=qkv
 _base.patch=patch;_capture=_base.install(config);block_wrapper=_base.block_wrapper;mega_wrapper=_base.mega_wrapper
 return _capture
