"""Independent opt-in HC/FFN/Engram seam observer. No eager Torch/product import.
Frozen v2 owns RID admission, async completion, two-case dispatch and primary2/8.
This extension never computes HC values; only natural original tensors are cloned.
"""
import contextvars,functools,hashlib,importlib.util,inspect,json,os,sys,time
from pathlib import Path
BASE_SHA='d1f4bfb16039ffcc43905fe3ae8bfc0b10dbacebcd895f994200a07845c7fe70'
BASE_PATH=Path(__file__).with_name('p_only_state_observer_v2.py')
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA:raise RuntimeError('FROZEN_V2_CORE_SHA')
_spec=importlib.util.spec_from_file_location('_c3_hc_seam_private_v2_core',BASE_PATH)
base=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=base;_spec.loader.exec_module(base)
MODEL,ATTENTION,ASYNC,METADATA=base.MODEL,base.ATTENTION,base.ASYNC,base.METADATA
BLOCK='rtp_llm.models_py.modules.dsv41.block'
HC='rtp_llm.models_py.modules.dsv41.hc.delayed'
MOE='rtp_llm.models_py.modules.dsv41.moe.moe_layer'
PRENORM='rtp_llm.models_py.modules.dsv41.hc.v41_prenorm'
FUNCTIONAL='rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional'
HOOK_SOURCES=(BLOCK,HC,MOE,ATTENTION,PRENORM,FUNCTIONAL)
TARGETS=(MODEL,ATTENTION,ASYNC,METADATA,BLOCK,HC,MOE,PRENORM,FUNCTIONAL)
_hc_scope=contextvars.ContextVar('p_hc_seam_natural_pre_v4',default=None)
EXTENSION_SCHEMA='c3-p-hc-seam-extension-v4';EXTRA_LIMIT=4*1024**2
canonical,digest,read_arm=base.canonical,base.digest,base.read_arm
_scope=base._scope
REQUIRED=('L0.attnHC.residual4HC','L0.attnHC.prenorm_raw_mixes24','L0.attnHC.pre_mix4','L0.attnHC.post4','L0.attnHC.comb16','L0.attnHC.fn_FP32','L0.attnHC.scale_FP32','L0.attnHC.base_FP32','L0.projectedAR5120','L0.FFN.input_norm5120','L0.FFN.result5120','L0.ffnHC.pre_mix4','L0.ffnHC.post4','L0.ffnHC.comb16','L0.block.result4HC','L1.beforeEngram4HC','L1.afterEngram4HC','L1.HC.residual4HC','L1.HC.readout5120','L1.attn_input_norm5120')
STAGE_QUERY_LAYERS=(2,3,4,5,6,7)
STAGE_POINTS=('L3.stage.attn_input_norm5120', 'L4.stage.attn_input_norm5120', 'L5.stage.attn_input_norm5120', 'L6.stage.attn_input_norm5120', 'L7.stage.attn_input_norm5120', 'L2.stage.projectedAR5120', 'L3.stage.projectedAR5120', 'L4.stage.projectedAR5120', 'L5.stage.projectedAR5120', 'L6.stage.projectedAR5120', 'L7.stage.projectedAR5120')
REQUIRED=REQUIRED+STAGE_POINTS
def need(ok,message):
 if not ok:raise RuntimeError(message)
def source_pin(path):
 raw=path.read_bytes();return dict(path=str(path.resolve()),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def source_fields():return dict(extension_schema=EXTENSION_SCHEMA,extension_source=source_pin(Path(__file__)),base_source=source_pin(BASE_PATH),bootstrap_source=source_pin(Path(__file__).with_name('p_only_state_observer_bootstrap_v8.py')))
def extension_evidence(config,phase,fields=None):return base.evidence(config,phase,dict(source_fields(),**(fields or {})))
def validate(config):
 base.validate(config)
 need(config.get('additional_layers')==[0,1] and config.get('additional_query_positions')==[7168] and config.get('additional_budget_bytes')==EXTRA_LIMIT,'EXPLICIT_V4_LAYERS_QUERY_4MIB_REQUIRED')
 need(config.get('stage_query_layers')==list(STAGE_QUERY_LAYERS),'EXPLICIT_L2_TO_L8_GAP_ONLY_REQUIRED')
 need(set(config.get('hook_source_sha256',{}))==set(HOOK_SOURCES),'SIX_FROZEN_HOOK_SOURCE_PINS_REQUIRED')
 need(all(isinstance(x,str) and len(x)==64 and set(x)<=set('0123456789abcdef') for x in config['hook_source_sha256'].values()),'HOOK_SOURCE_FULL_SHA_REQUIRED')
 return config

def version_evidence(tensor):
 try:return dict(version=int(tensor._version),version_available=True)
 except RuntimeError as e:
  need('Inference tensors do not track version counter' in str(e),'UNEXPECTED_VERSION_ERROR')
  return dict(version=None,version_available=False,reason='inference_tensor_source_lifetime_proof_required')
def reference_identity(tensor):
 return dict(shape=list(tensor.shape),stride=list(tensor.stride()),storage_offset=int(tensor.storage_offset()),data_ptr=int(tensor.data_ptr()),dtype=str(tensor.dtype),device=str(tensor.device),**version_evidence(tensor))
def unchanged_reference(tensor,before):
 after=reference_identity(tensor)
 for key in ('shape','stride','storage_offset','data_ptr','dtype','device','version_available'):need(after[key]==before[key],'STASH_REFERENCE_CHANGED:'+key)
 if before['version_available']:need(after['version']==before['version'],'STASH_VERSION_CHANGED')
 return after

def geometry(shape,trailing,dtype,expected_dtype):
 need(len(shape)==len(trailing)+1 and tuple(shape[1:])==tuple(trailing) and dtype==expected_dtype,'ORIGINAL_TYPED_FULL_GEOMETRY_REQUIRED')
 return __import__('math').prod(trailing)
def reserve_additional(total,extra,retained,transient=0):
 need(extra+retained+transient<=EXTRA_LIMIT,'ADDITIONAL_4MIB_BUDGET')
 need(total+retained+transient<=base.GPU_LIMIT,'GPU_TENSOR_BUDGET')
 return extra+retained

def select_query_domain(domains):
 """CPU proof only; req_id_per_token is a batch-row key, never true RID."""
 need(len(domains)==4 and {d['cp_rank'] for d in domains}==set(range(4)),'FOUR_CP_DOMAINS_REQUIRED');owners=[]
 for d in domains:
  p,r,b=d['global_positions'],d['local_is_real'],d['req_id_per_token'];need(len(p)==len(r)==len(b),'CP_DOMAIN_ALIGNMENT')
  for i,(pos,real,row) in enumerate(zip(p,r,b)):
   if real and pos==7168:need(row==0,'QUERY_WRONG_BATCH_ROW');owners.append((d['cp_rank'],i))
 need(len(owners)==1,'ONE_ACTUAL_QUERY_OWNER_REQUIRED');return owners[0]

def budget_plan(local_rows=1794):
 """Actual worst-case retained/peak upper bound, 4HC and whole FP32 fn."""
 # one-row clone, one row index_select temporary, two-slot evidence container.
 widths=[20480,24,4,4,16,5120,5120,5120,4,4,16,20480,20480,20480,5120,5120,20480,20480]
 retained=1966080+12+96+3*64;peak=retained+4096
 for width in widths:
  elem=4 if width in (4,16,24) else 2;size=2*width*elem;transient=local_rows*32+width*elem+4096
  peak=max(peak,retained+size+64+transient);retained=reserve_additional(retained,retained,size+64,transient)
 # complete CP positions, real mask and batch-row carrier, one copy each.
 for elem in (8,1,4):
  size=2*local_rows*elem;peak=max(peak,retained+size+64+size+4096);retained=reserve_additional(retained,retained,size+64,size+4096)
 for _ in STAGE_POINTS:
  size=2*5120*2;transient=local_rows*32+5120*2+4096;peak=max(peak,retained+size+64+transient);retained=reserve_additional(retained,retained,size+64,transient)
 return dict(retained_bytes=retained,peak_bytes=peak,additional_limit=EXTRA_LIMIT,total_limit=base.GPU_LIMIT)

class Snapshots(base.Snapshots):
 def reserve(self,retained,transient=0):
  extra=reserve_additional(self.record.retained,self.record.extra_retained,retained,transient);self._reserve(retained,transient);self.record.extra_retained=extra
 def point(self,label,tensor,cp,trailing,dtype):
  t=self.torch;width=geometry(tuple(tensor.shape),trailing,str(tensor.dtype),dtype)
  need(tensor.device.type=='cuda' and not t.cuda.is_current_stream_capturing(),'CUDA_EAGER_ONLY')
  need(tensor.shape[0]==cp.global_positions.numel() and cp.local_is_real.shape==cp.global_positions.shape and cp.req_id_per_token.shape==cp.global_positions.shape,'ACTUAL_CP_QUERY_ROW_ALIGNMENT')
  size=2*width*tensor.element_size();self.reserve(size+64,cp.global_positions.numel()*32+width*tensor.element_size()+4096)
  indices,found=self._indices(cp.global_positions,cp.local_is_real);found[0]=False
  # Gather only one row; never clone full CP activation. Flatten the selected
  # row AFTER index_select so no implicit whole-input .reshape copy is possible.
  current=tensor.detach().index_select(0,indices[1:2]).reshape(1,width)
  value=tensor.new_zeros((2,width));value[1].copy_(current[0]);value.masked_fill_(~found[:,None],0)
  self.record.add(label,value,found,dict(dtype=str(tensor.dtype),original_shape=list(tensor.shape),original_trailing_shape=list(trailing),sample_shape=[2,width],additional_point=True,required_query_position=7168,full_width_preserved=True,finite_per_query=True,row_domain='actual_local_CP',diagnostic_non_target_slot_masked=True))
  self.record.additional_points.add(label)
 def parameter(self,label,tensor,shape):
  t=self.torch;need(tuple(tensor.shape)==tuple(shape) and tensor.dtype==t.float32 and tensor.device.type=='cuda' and tensor.is_contiguous() and not t.cuda.is_current_stream_capturing(),'ORIGINAL_HC_PARAMETER_FP32_GEOMETRY')
  size=tensor.numel()*tensor.element_size();self.reserve(size+64,4096);value=tensor.detach().view(1,-1).clone();mask=t.ones(1,device=tensor.device,dtype=t.bool)
  self.record.add(label,value,mask,dict(dtype=str(tensor.dtype),original_shape=list(shape),sample_shape=list(value.shape),additional_point=True,parameter_container=True,static_parameter=True,container_semantics='original_static_parameter_no_query_axis',finite_per_query=True,full_parameter_bytes_preserved=True));self.record.additional_points.add(label)
 def metadata(self,label,tensor):
  t=self.torch;need(tensor.ndim==1 and tensor.numel()<=1794 and tensor.device.type=='cuda','BOUNDED_LOCAL_CP_VECTOR')
  size=2*tensor.numel()*tensor.element_size();self.reserve(size+64,size+4096);value=tensor.detach().unsqueeze(0).expand(2,-1).clone();mask=t.ones(2,device=tensor.device,dtype=t.bool)
  self.record.add(label,value,mask,dict(dtype=str(tensor.dtype),original_shape=list(tensor.shape),sample_shape=list(value.shape),metadata_container=True,additional_point=True,finite_per_query=True))

 def shared_scale_point(self,label,tensor,cp,block_m):
  t=self.torch;need(tensor.ndim==2 and tensor.shape[1]==40 and tensor.dtype==t.int32 and tensor.device.type=='cuda' and type(block_m)is int and block_m>0,'ACTUAL_SHARED_SCALE_LAYOUT')
  width=40;self.reserve(2*width*4+64,8192)
  indices,found=self._indices(cp.global_positions,cp.local_is_real);found[0]=False
  within=indices%block_m;aligned=(block_m+127)//128*128
  mapped=indices//block_m*aligned+within//128*128+(within%32)*4+(within%128)//32
  current=tensor.detach().index_select(0,mapped[1:2]);value=tensor.new_zeros((2,width));value[1].copy_(current[0]);value.masked_fill_(~found[:,None],0)
  self.record.add(label,value,found,dict(dtype='torch.int32',original_shape=list(tensor.shape),original_stride=list(tensor.stride()),sample_shape=[2,width],additional_point=True,required_query_position=7168,full_width_preserved=True,finite_per_query=True,row_domain='actual_local_CP_query_mapped_shared_UTCCP',diagnostic_non_target_slot_masked=True,block_m=block_m,aligned_block_m=aligned,mapping_source='frozen stage_mega_moe_se_shared_l1_scales inverse-equivalent forward permutation'))
  self.record.additional_points.add(label)

SUPPLEMENTAL_REQUIRED={'L0.MoE.packed_x_uint8','L0.MoE.routed_sf_int32','L0.MoE.shared_sf_int32','L0.MoE.topk_idx','L0.MoE.topk_weights'}

class Record(base.Record):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.extra_retained=0;self.additional_points=set();self.blocks={};self.pending_hc=None;self.cp_metadata_captured=False;self.current_layer=None;self.cp_domains={};self.domains_validated=set()
 def completion_ready_to_record(self):
  if not super().completion_ready_to_record():return False
  if self.domains_validated!=({0,1}|set(STAGE_QUERY_LAYERS)):self.problem='NOT_CAPTURED_PRODUCER_DOMAIN_REVALIDATION';return False
  missing=(set(REQUIRED)|SUPPLEMENTAL_REQUIRED)-self.additional_points
  if missing:self.problem='NOT_CAPTURED_REQUIRED_HC_SEAM_POINTS:'+','.join(sorted(missing));return False
  if len([e for e in self.events if e.get('point')=='L0.MoE.actual_launch'])!=1:self.problem='NOT_CAPTURED_ONE_ACTUAL_MOE_LAUNCH';return False
  if self.pending_hc is not None:self.problem='NOT_CAPTURED_UNFLUSHED_EARLY_HC_REFS';return False
  return True
base.Record=Record

class Observer(base.Observer):
 def __init__(self,config,backend_factory=Snapshots,*a,**kw):validate(config);super().__init__(config,backend_factory,*a,**kw)
 def release_extension_refs(self,record):
  record.pending_hc=None;record.blocks.clear();record.cp_domains.clear()
  if record.problem:
   record.samples.clear();record.tokens.clear();record.previous_bindings.clear();record.state_stack.clear();record.launches.clear();record.snapshots_released=True
 def failure_evidence(self,record):
  try:super().failure_evidence(record)
  finally:self.release_extension_refs(record)
 def worker(self,record,*args,**kwargs):
  try:return super().worker(record,*args,**kwargs)
  finally:self.release_extension_refs(record)
 def cp(self,record,layer):
  block=record.blocks.get(layer);need(block is not None,'ACTUAL_BLOCK_BINDING_REQUIRED');cp=getattr(block.attn,'_cp_ctx',None);prefix=self.config['expected_prefix_tokens']
  need(cp is not None and cp.cp_size==4 and cp.kv_cache_sharded and tuple(cp.prefix_lengths_host)==(prefix,) and tuple(cp.input_lengths_global_host)==(7169-prefix,) and cp.global_positions.numel()<=1794,'ACTUAL_N7169_CP4_DOMAIN_REQUIRED')
  refs=(cp,cp.global_positions,cp.local_is_real,cp.req_id_per_token)
  if 0 in record.cp_domains:need(cp is record.cp_domains[0][0],'EARLY_LAYER_CP_OBJECT_CHANGED')
  old=record.cp_domains.setdefault(layer,refs)
  need(all(a is b for a,b in zip(refs,old)),'ACTUAL_CP_CARRIER_IDENTITY_CHANGED')
  return cp
 def copy_point(self,record,label,tensor,layer,trailing,dtype='torch.bfloat16'):
  cp=self.cp(record,layer);record.backend.point(label,tensor,cp,trailing,dtype)
 def hc_wrapper(self,name,original):
  @functools.wraps(original)
  def wrapped(unit,*args,**kwargs):
   record=_scope.get();layer=getattr(unit,'layer_id',-1)
   if record is None or record.problem or layer not in (0,1):return original(unit,*args,**kwargs)
   block=record.blocks.get(layer);kind='attn' if block is not None and unit is block.attn_hc else 'ffn' if block is not None and unit is block.ffn_hc else None
   if kind is None:record.guard(lambda:need(False,'HC_UNIT_NOT_BOUND_TO_ACTUAL_BLOCK'));return original(unit,*args,**kwargs)
   if name=='_pre_impl' and layer==0 and kind=='attn':
    x=args[0] if args else kwargs['x'];before=record.guard(lambda:reference_identity(x));context=dict(record=record,unit=unit,x=x,mixes=None,mixes_identity=None,route=None);token=_hc_scope.set(context)
    try:result=original(unit,*args,**kwargs)
    finally:_hc_scope.reset(token)
    def stash():
     need(before is not None and record.pending_hc is None,'DUPLICATE_L0_HC_STASH');post,comb=result[1:];pre=unit.pre_mix_out
     need(context['mixes'] is not None,'NATURAL_PRENORM_RAW_MIXES_REQUIRED')
     record.pending_hc=dict(raw_mixes=context['mixes'],raw_mixes_identity=context['mixes_identity'],raw_mixes_route=context['route'],unit=unit,residual=x,residual_identity=before,pre=pre,post=post,comb=comb,refs=[reference_identity(t) for t in (pre,post,comb)],stream=record.backend.stream())
    record.guard(stash);return result
   if name=='_pre_impl' and layer==1 and kind=='attn':record.guard(lambda:self.copy_point(record,'L1.HC.residual4HC',args[0] if args else kwargs['x'],1,(4,5120)))
   result=original(unit,*args,**kwargs)
   if name=='_pre_impl' and layer==0 and kind=='ffn':record.guard(lambda:self.ffn_coefficients(record,unit,result[1],result[2]))
   if name=='_pre_impl' and layer==1 and kind=='attn':record.guard(lambda:self.copy_point(record,'L1.HC.readout5120',result[0],1,(5120,)))
   if name=='_post_impl' and layer==0 and kind=='ffn':record.guard(lambda:self.copy_point(record,'L0.block.result4HC',result,0,(4,5120)))
   return result
  return wrapped
 def raw_mixes_wrapper(self,route,original):
  @functools.wraps(original)
  def wrapped(*args,**kwargs):
   context=_hc_scope.get();result=original(*args,**kwargs)
   if context is not None:
    record=context['record']
    def remember():
     if result is None:
      record.events.append(dict(point='L0_prenorm_natural_route',route=route,returned_None=True));return
     residual=args[0] if args else kwargs['residual'];fn=args[1] if len(args)>1 else kwargs.get('fn',kwargs.get('mhc_fn'))
     need(residual.data_ptr()==context['x'].data_ptr() and fn is context['unit'].fn,'PRENORM_NOT_ACTUAL_HC_RESIDUAL_AND_PARAMETER')
     need(context['mixes'] is None,'DUPLICATE_PRENORM_RAW_MIXES');context.update(mixes=result,mixes_identity=reference_identity(result),route=route);record.events.append(dict(point='L0_prenorm_natural_route',route=route,returned_None=False))
    record.guard(remember)
   return result
  return wrapped
 def ffn_coefficients(self,record,unit,post,comb):
  for label,value,shape in [('pre_mix4',unit.pre_mix_out,(4,)),('post4',post,(4,1)),('comb16',comb,(4,4))]:self.copy_point(record,'L0.ffnHC.'+label,value,0,shape,'torch.float32')
 def validate_producer_domain(self,record,attention,common):
  cp=self.cp(record,attention.layer_id);need(common.cp_ctx is cp,'PRODUCER_COMMON_CP_OBJECT_CHANGED');record.domains_validated.add(attention.layer_id)
  record.events.append(dict(point='actual_common_CP_domain_revalidated',layer=attention.layer_id,cp_rank=cp.cp_rank,cp_size=4,local_rows=cp.global_positions.numel(),prefixes_host=list(cp.prefix_lengths_host),lengths_host=list(cp.input_lengths_global_host),three_carrier_objects_identical=True,batch_row_is_not_RID=True))
 def flush_hc(self,record,attention,common):
  pending=record.pending_hc;need(pending is not None,'L0_NATURAL_HC_PRE_REQUIRED');cp=self.cp(record,0);need(attention is record.blocks[0].attn and common is not None,'ACTUAL_PRODUCER_COMMON_REQUIRED');need(record.backend.stream()==pending['stream'],'HC_STASH_STREAM_CHANGED')
  unchanged_reference(pending['residual'],pending['residual_identity'])
  for tensor,before in zip((pending['pre'],pending['post'],pending['comb']),pending['refs']):unchanged_reference(tensor,before)
  mixes=pending['raw_mixes'];unchanged_reference(mixes,pending['raw_mixes_identity']);need(tuple(mixes.shape)==(1,cp.global_positions.numel(),24) and mixes.is_contiguous(),'NATURAL_PRENORM_RAW_MIXES_GEOMETRY')
  record.backend.point('L0.attnHC.prenorm_raw_mixes24',mixes.view(mixes.shape[1],24),cp,(24,),'torch.float32')
  record.backend.point('L0.attnHC.residual4HC',pending['residual'],cp,(4,5120),'torch.bfloat16')
  for label,key,shape in [('pre_mix4','pre',(4,)),('post4','post',(4,1)),('comb16','comb',(4,4))]:record.backend.point('L0.attnHC.'+label,pending[key],cp,shape,'torch.float32')
  unit=pending['unit']
  for label,key,shape in [('fn_FP32','fn',(24,20480)),('scale_FP32','scale',(3,)),('base_FP32','base',(24,))]:record.backend.parameter('L0.attnHC.'+label,getattr(unit,key),shape)
  if not record.cp_metadata_captured:
   for f in ('global_positions','local_is_real','req_id_per_token'):record.backend.metadata('seam.cp.'+f,getattr(cp,f))
   record.cp_metadata_captured=True
  record.events.append(dict(point='L0_HC_STASH_FLUSHED_AT_ORIGINAL_PRODUCER_RETURN',cp_rank=cp.cp_rank,cp_size=4,chunk_length=cp.chunk_length,residual_reference_version=pending['residual_identity']['version'],version_available=pending['residual_identity']['version_available'],lifetime_proof='frozen_DelayedHC_pre_to_original_produce_return_before_HC_post',pre_norm_HC_readout_captured=False,prenorm_raw_route=pending['raw_mixes_route']));record.pending_hc=None
 def attention_wrapper(self,name,original):
  primary=super().attention_wrapper(name,original)
  @functools.wraps(original)
  def wrapped(attention,*args,**kwargs):
   record=_scope.get();layer=getattr(attention,'layer_id',-1)
   if record is not None and not record.problem and layer in STAGE_QUERY_LAYERS and name=='_prefill_produce':
    x=args[0] if args else kwargs['x'];before=reference_identity(x)
    result=primary(attention,*args,**kwargs)
    def capture_stage_input():
     cp=self.cp(record,layer);need(result[0].cp_ctx is cp,'STAGE_PRODUCER_ACTUAL_CP_OBJECT_REQUIRED');record.domains_validated.add(layer);record.events.append(dict(point='stage_query_CP_domain_revalidated',layer=layer,cp_rank=cp.cp_rank,cp_size=4,three_carrier_objects_identical=True,batch_row_is_not_RID=True))
     unchanged_reference(x,before)
     if layer>=3:self.copy_point(record,'L%s.stage.attn_input_norm5120'%layer,x,layer,(5120,))
    record.guard(capture_stage_input);return result
   if record is None or record.problem or layer not in (0,1):return primary(attention,*args,**kwargs)
   if name=='_prefill_produce':
    x=args[0] if args else kwargs['x']
    if layer==1:record.guard(lambda:self.copy_point(record,'L1.attn_input_norm5120',x,1,(5120,)))
    result=original(attention,*args,**kwargs)
    record.guard(lambda:self.validate_producer_domain(record,attention,result[0]))
    if layer==0:record.guard(lambda:self.flush_hc(record,attention,result[0]))
    return result
   return original(attention,*args,**kwargs)
  return wrapped
 def sparse_wrapper(self,original):
  @functools.wraps(original)
  def wrapped(attention,*args,**kwargs):
   result=original(attention,*args,**kwargs);record=_scope.get()
   if record is not None and not record.problem and attention.layer_id==0:record.guard(lambda:self.copy_point(record,'L0.projectedAR5120',result,0,(5120,)))
   if record is not None and not record.problem and attention.layer_id in STAGE_QUERY_LAYERS:record.guard(lambda:self.copy_point(record,'L%s.stage.projectedAR5120'%attention.layer_id,result,attention.layer_id,(5120,)))
   return result
  return wrapped
 def block_wrapper(self,name,original):
  @functools.wraps(original)
  def wrapped(block,*args,**kwargs):
   record=_scope.get();layer=getattr(block,'layer_id',-1)
   if record is not None and not record.problem and layer in STAGE_QUERY_LAYERS:record.blocks[layer]=block
   if record is None or record.problem or layer not in (0,1):return original(block,*args,**kwargs)
   record.blocks[layer]=block
   if name=='_inject_engram' and layer==1:record.guard(lambda:self.copy_point(record,'L1.beforeEngram4HC',args[0] if args else kwargs['hidden'],1,(4,5120)))
   result=original(block,*args,**kwargs)
   if name=='_inject_engram' and layer==1:record.guard(lambda:self.copy_point(record,'L1.afterEngram4HC',result,1,(4,5120)))
   if name=='_try_mega_mhc' and layer==0 and result is not None:
    # Observe natural fused seam outputs; never force/disable the fusion.
    record.guard(lambda:self.copy_point(record,'L0.mega.post4HC',result[0],0,(4,5120)))
    record.guard(lambda:self.ffn_coefficients(record,block.ffn_hc,result[2],result[3]))
   return result
  return wrapped
 def _moe_original_wrapper(self,original):
  @functools.wraps(original)
  def wrapped(moe,*args,**kwargs):
   record=_scope.get();block=record.blocks.get(0) if record else None;active=record is not None and not record.problem and block is not None and moe is block.ffn
   if active:record.guard(lambda:self.copy_point(record,'L0.FFN.input_norm5120',args[0] if args else kwargs['x'],0,(5120,)))
   result=original(moe,*args,**kwargs)
   if active:record.guard(lambda:self.copy_point(record,'L0.FFN.result5120',result,0,(5120,)))
   return result
  return wrapped
 def moe_wrapper(self,original):
  inherited=self._moe_original_wrapper(original)
  @functools.wraps(original)
  def wrapped(moe,*args,**kwargs):
   record=_scope.get();block=record.blocks.get(0) if record else None
   if record is None or record.problem or block is None or moe is not block.ffn:return inherited(moe,*args,**kwargs)
   strategy=moe._strategy;launch=getattr(strategy,'_launch',None)
   if launch is None:
    record.guard(lambda:need(False,'FFN_STRATEGY_HAS_NO_NATURAL_LAUNCH'));return inherited(moe,*args,**kwargs)
   need('_launch' not in strategy.__dict__,'FFN_LAUNCH_ALREADY_SHADOWED')
   def observe_launch(y,tokens,device,*a,**kw):
    def observe():
     buf=strategy._mega_buf
     need(tokens==self.cp(record,0).global_positions.numel(),'ACTUAL_MOE_CP_ROWS')
     record.events.append(dict(point='L0.MoE.actual_launch',strategy_class=type(strategy).__module__+'.'+type(strategy).__name__,local_tokens=tokens,shared_fp8_block_size=strategy.cfg.shared_fp8_block_size,gate_pack=moe._use_gate_pack(),gate_hash=bool(moe.gate.hash),topk=moe.gate.topk,gate_score_func=moe.gate.score_func,strategy_source_SHA256=source_pin(Path(inspect.getfile(type(strategy))))['SHA256'],gate_source_SHA256=source_pin(Path(inspect.getfile(type(moe.gate))))['SHA256'],gate_weight_SHA256=hashlib.sha256(moe.gate._weight_bf16().detach().cpu().view(record.backend.torch.uint8).numpy().tobytes()).hexdigest(),gate_bias_SHA256=None if moe.gate.bias is None else hashlib.sha256(moe.gate.bias.detach().cpu().view(record.backend.torch.uint8).numpy().tobytes()).hexdigest(),block_m=strategy._block_m(tokens),packed_x_dtype=str(buf.x.dtype),packed_x_shape=list(buf.x[:tokens].shape),x_sf_shape=list(buf.x_sf.shape),shared_sf_shape=list(buf.shared_l1_acts_sf.shape)))
     values=[('packed_x_uint8',buf.x[:tokens].view(record.backend.torch.uint8),(5120,),'torch.uint8'),('routed_sf_int32',buf.x_sf[:tokens],(40,),'torch.int32'),('topk_idx',buf.topk_idx[:tokens],(moe.gate.topk,),str(buf.topk_idx.dtype)),('topk_weights',buf.topk_weights[:tokens],(moe.gate.topk,),str(buf.topk_weights.dtype))]
     for label,value,shape,dtype in values:self.copy_point(record,'L0.MoE.'+label,value,0,shape,dtype)
     record.backend.shared_scale_point('L0.MoE.shared_sf_int32',buf.shared_l1_acts_sf,self.cp(record,0),strategy._block_m(tokens))
    record.guard(observe)
    return launch(y,tokens,device,*a,**kw)
   strategy._launch=observe_launch
   try:return inherited(moe,*args,**kwargs)
   finally:del strategy.__dict__['_launch']
  return wrapped
 def persist(self,record):
  # Same terminal/worker/rename ordering as frozen v2; only extra metadata and
  # finite serialization differ for the explicitly typed new samples.
  if not record.forward_returned or record.terminal_event is None or not record.terminal_event.query():raise RuntimeError('ACTUAL_TERMINAL_COMPLETION_REQUIRED')
  root=Path(self.config['output_dir']);partial=root/(self.config['nonce']+'-%d.partial'%os.getpid());final=root/(self.config['nonce']+'-%d'%os.getpid());partial.mkdir(parents=True,exist_ok=False);base.reserve_case_disk(self.config['batch_budget_dir']);total=0;samples=[];torch=record.backend.torch
  for index,(label,tensor,mask,metadata) in enumerate(record.samples):
   if time.monotonic()>=record.worker_deadline:raise RuntimeError('BOUNDED_DRAIN_DEADLINE')
   cpu=tensor.detach().contiguous().cpu();raw=cpu.view(torch.uint8).numpy().tobytes();total+=len(raw)
   if total+1024**2>base.CASE_DISK_LIMIT:raise RuntimeError('RANK_DISK_BUDGET')
   name='%04d.raw'%index;path=partial/name;path.write_bytes(raw);need(path.stat().st_size==len(raw) and hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest(),'SNAPSHOT_DISK_SIZE_OR_SHA')
   meta=dict(metadata);k_mask=meta.pop('valid_k_mask',None)
   if cpu.is_floating_point():
    finite=torch.isfinite(cpu)
    if meta.get('finite_per_query'):
     if k_mask is not None:
      valid=k_mask.detach().cpu();finite=finite.reshape(2,128,512)|~valid[:,:,None];meta['valid_k_mask']=valid.tolist()
     finite=finite.reshape(cpu.shape[0],-1).all(1).tolist()
    else:finite=finite.tolist()
   else:finite=None
   samples.append(dict(meta,label=label,file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),valid=mask.detach().cpu().tolist(),finite=finite))
  manifest=dict(profile='P_HC_SEAM_DIAGNOSTIC_V4_NOT_ACCEPTANCE',primary_compatibility_profile='P_STATE_DIAGNOSTIC_V2_NOT_ACCEPTANCE',request_id=record.arm['request_id'],nonce=self.config['nonce'],arm_sha256=record.arm_sha,input_ids_sha256=record.arm['input_ids_sha256'],independent_identity_ref_sha256=record.identity_ref['identity_ref_sha256'],model_identity_sha256_from_arm=record.arm['model_identity_sha256'],server_epoch_artifact_admission_verified_by_this_helper=False,server_single_rpc_isolation_verified_by_this_helper=False,source_process_provenance_verified=False,PID=os.getpid(),UID=os.getuid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],forward_stream=record.forward_stream,terminal_event_query_completed=True,terminal_covers_cpp_lm_head_or_rpc_completion=False,gpu_retained_tensor_bytes=record.retained,allocator_overhead_accounted=False,disk_raw_bytes=total,positions=[7167,7168],layers=[2,8],events=record.events,samples=samples,problem=record.problem,additional_layers=[0,1],stage_query_layers=list(STAGE_QUERY_LAYERS),additional_query_positions=[7168],additional_budget_bytes=EXTRA_LIMIT,additional_retained_bytes=record.extra_retained,additional_points=sorted(record.additional_points),mapping_status='CAPTURED_CP_QUERY_METADATA_PENDING_FOUR_RANK_CPU_JOIN',additional_schema=EXTENSION_SCHEMA,required_additional_points=list(REQUIRED),optional_Q_O_captured=False,**source_fields())
  encoded=canonical(manifest);need(len(encoded)<=1024**2,'METADATA_BUDGET');(partial/'manifest.json').write_bytes(encoded);need(not final.exists(),'OUTPUT_EXISTS');os.rename(partial,final);record.result=dict(path=str(final),manifest_sha256=hashlib.sha256(encoded).hexdigest())

class Dispatcher(base.Dispatcher):
 def __init__(self,config,observer_factory=Observer):
  need(config.get('schema')=='c3-p-state-observer-batch-v4','EXPLICIT_BATCH_V4_REQUIRED');compat=json.loads(canonical(config));compat['schema']='c3-p-state-observer-batch-v2';super().__init__(compat,observer_factory)
 def patch_module(self,name,module):
  if name in HOOK_SOURCES:
   actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
   for case in self.observers:need(actual==case.config['hook_source_sha256'][name],'HOOK_SOURCE_SHA:'+name)
  if name in (MODEL,ATTENTION,ASYNC,METADATA):super().patch_module(name,module)
  if name==BLOCK:targets=[(module.Block,m,'block') for m in ('forward','_forward_prefill_fast_fp8','prefill_fast_attn_pre','prefill_fast_attn_post','prefill_fast_ffn_pre','prefill_fast_ffn_post','_inject_engram','_try_mega_mhc')]
  elif name==HC:targets=[(module.DelayedHCUnit,m,'hc') for m in ('_pre_impl','_post_impl')]
  elif name==PRENORM:targets=[(module,'prenorm','raw')]
  elif name==FUNCTIONAL:targets=[(module,'mhc_pre_norm_fn','raw')]
  elif name==MOE:targets=[(module.MoE,'forward','moe')]
  elif name==ATTENTION:targets=[(module.AttentionV41FP8,'_prefill_sparse_attention','sparse')]
  else:targets=[]
  for owner,method,kind in targets:
   original=getattr(owner,method);need(not getattr(original,'_p_state_extension_v4',False),'EXTENSION_ALREADY_PATCHED');callbacks={id(c):(c.block_wrapper(method,original) if kind=='block' else c.hc_wrapper(method,original) if kind=='hc' else c.moe_wrapper(original) if kind=='moe' else c.raw_mixes_wrapper(name+'.'+method,original) if kind=='raw' else c.sparse_wrapper(original)) for c in self.observers}
   def route(*a,_original=original,_callbacks=callbacks,**kw):
    r=_scope.get();fn=_callbacks.get(id(r.owner)) if r else None;return (fn or _original)(*a,**kw)
   route=functools.wraps(original)(route);route._p_state_extension_v4=True;setattr(owner,method,route)
  for case in self.observers:extension_evidence(case.config,'extension-hooked.'+name,dict(module=name,module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),methods=[m for _,m,_ in targets]))
class Finder(base.Finder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname not in TARGETS:return None
  spec=base.importlib.machinery.PathFinder.find_spec(fullname,path)
  if spec is None or spec.loader is None or not hasattr(spec.loader,'exec_module'):raise ImportError('NATURAL_EXEC_MODULE_LOADER_REQUIRED')
  original,observer=spec.loader,self.observer
  class Loader:
   def create_module(self,spec):
    method=getattr(original,'create_module',None);return method(spec) if method else None
   def exec_module(self,module):original.exec_module(module);observer.patch_module(fullname,module)
  spec.loader=Loader();return spec
def install(config):
 need(not any(n in sys.modules for n in TARGETS),'BOOTSTRAP_TOO_LATE');observer=Dispatcher(config);sys.meta_path.insert(0,Finder(observer))
 for case in observer.observers:
  base.evidence(case.config,'lazy-finder-installed',dict(natural_targets=list(TARGETS),final_RID_required_at_startup=False,actual_model_epoch_required_at_startup=False))
  extension_evidence(case.config,'extension-lazy-finder-installed',dict(natural_targets=list(TARGETS),final_RID_required_at_startup=False,actual_model_epoch_required_at_startup=False))
 return observer

def verify_transition_case(entries):
 """Read-only device-CPU proof. Caller independently pins actual READY epochs.
 Decodes only original CP integer metadata, never emits tensor values or IDs.
 """
 import struct,math
 need(len(entries)==4,'NOT_CAPTURED_FOUR_MANIFESTS_REQUIRED');loaded=[]
 expected_sources=source_fields()
 def member(directory,sample):
  name=sample['file'];need(Path(name).name==name,'NOT_CAPTURED_RAW_PATH');p=directory/name
  need(p.is_file() and not p.is_symlink() and p.stat().st_uid==os.getuid() and p.stat().st_size==sample['bytes']<=EXTRA_LIMIT,'NOT_CAPTURED_RAW_MEMBER')
  raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==sample['sha256'],'NOT_CAPTURED_RAW_SHA');return raw
 def integers(directory,sample):
  types={'torch.int64':('q',8),'torch.int32':('i',4),'torch.bool':('?',1)};need(sample['dtype'] in types,'NOT_CAPTURED_ORIGINAL_CP_INTEGER_DTYPE');fmt,size=types[sample['dtype']];raw=member(directory,sample);n=sample['original_shape'][0]
  need(len(raw)==2*n*size,'NOT_CAPTURED_CP_METADATA_GEOMETRY');v=list(struct.unpack('<'+fmt*(2*n),raw));need(v[:n]==v[n:],'NOT_CAPTURED_CP_METADATA_CONTAINER');return v[:n]
 for entry in entries:
  path=Path(entry['path']);need(path.is_file() and not path.is_symlink() and path.stat().st_uid==os.getuid() and path.stat().st_size<=1024**2,'NOT_CAPTURED_MANIFEST_IDENTITY');raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==entry['sha256'],'NOT_CAPTURED_MANIFEST_SHA');m=json.loads(raw)
  need(m['profile']=='P_HC_SEAM_DIAGNOSTIC_V4_NOT_ACCEPTANCE' and m['primary_compatibility_profile']=='P_STATE_DIAGNOSTIC_V2_NOT_ACCEPTANCE' and m['additional_schema']==EXTENSION_SCHEMA,'NOT_CAPTURED_EXPLICIT_V4_SCHEMA')
  need(m['layers']==[2,8] and m['positions']==[7167,7168] and m['additional_layers']==[0,1] and m['stage_query_layers']==list(STAGE_QUERY_LAYERS) and m['additional_query_positions']==[7168] and m['additional_budget_bytes']==EXTRA_LIMIT and m['additional_retained_bytes']<=EXTRA_LIMIT and m['gpu_retained_tensor_bytes']<=base.GPU_LIMIT and m['terminal_event_query_completed'] is True and m['problem'] is None,'NOT_CAPTURED_MANIFEST_CONTRACT')
  for key in ('extension_source','base_source','bootstrap_source'):need(m[key]['SHA256']==expected_sources[key]['SHA256'],'NOT_CAPTURED_SOURCE_CHAIN:'+key)
  samples={s['label']:s for s in m['samples']};need(len(samples)==len(m['samples']) and set(REQUIRED)<=set(samples),'NOT_CAPTURED_REQUIRED_POINT_OR_DUPLICATE')
  ev=[e for e in m['events'] if e['point']=='actual_common_CP_domain_revalidated'];need({e['layer'] for e in ev}=={0,1} and len({e['cp_rank'] for e in ev})==1 and all(e['three_carrier_objects_identical'] is True and e['batch_row_is_not_RID'] is True for e in ev),'NOT_CAPTURED_PRODUCER_CP_REVALIDATION')
  stages=[e for e in m['events'] if e['point']=='stage_query_CP_domain_revalidated'];need({e['layer'] for e in stages}==set(STAGE_QUERY_LAYERS) and len(stages)==len(STAGE_QUERY_LAYERS) and all(e['cp_rank']==ev[0]['cp_rank'] and e['three_carrier_objects_identical'] is True and e['batch_row_is_not_RID'] is True for e in stages),'NOT_CAPTURED_ALL_SIX_ACTUAL_STAGE_CP_DOMAINS')
  domain=dict(cp_rank=ev[0]['cp_rank'],**{f:integers(path.parent,samples['seam.cp.'+f]) for f in ('global_positions','local_is_real','req_id_per_token')})
  loaded.append((m,path.parent,samples,domain))
 need(len({(r[0]['PID'],r[0]['UID'],str(r[0]['start_ticks'])) for r in loaded})==4,'NOT_CAPTURED_FOUR_DISTINCT_EPOCHS')
 for key in ('nonce','request_id','model_identity_sha256_from_arm','input_ids_sha256'):need(len({r[0][key] for r in loaded})==1,'NOT_CAPTURED_MIXED_CASE:'+key)
 owner_cp,owner_row=select_query_domain([r[3] for r in loaded]);owner=next(r for r in loaded if r[3]['cp_rank']==owner_cp);proof=[];parameters=[]
 shapes={'L0.attnHC.fn_FP32':(1,491520),'L0.attnHC.scale_FP32':(1,3),'L0.attnHC.base_FP32':(1,24)}
 for label in REQUIRED:
  if label in shapes:
   pins=[]
   for m,d,samples,domain in loaded:
    sm=samples[label];data=member(d,sm);shape=shapes[label];original_shape=(24,20480) if label.endswith('fn_FP32') else (3,) if label.endswith('scale_FP32') else (24,)
    need(sm['original_shape']==list(original_shape),'NOT_CAPTURED_STATIC_PARAMETER_ORIGINAL_SHAPE')
    need(sm['parameter_container'] is True and sm['static_parameter'] is True and sm['container_semantics']=='original_static_parameter_no_query_axis' and sm['dtype']=='torch.float32' and sm['sample_shape']==list(shape) and sm['valid']==[True] and sm['finite']==[True] and len(data)==shape[1]*4,'NOT_CAPTURED_FULL_FP32_PARAMETER')
    need(all(math.isfinite(value[0]) for value in struct.iter_unpack('<f',data)),'NOT_CAPTURED_STATIC_PARAMETER_RAW_NONFINITE')
    pins.append(dict(PID=m['PID'],path=str(d/sm['file']),bytes=len(data),SHA256=sm['sha256']))
   parameters.append(dict(point=label,shape=list(shapes[label]),original_parameter_shape=list(original_shape),original_dtype='torch.float32',raw_members=pins,byte_equal_across_ranks=len({x['SHA256'] for x in pins})==1));continue
  width=20480 if label.endswith('4HC') else 4 if label.endswith(('pre_mix4','post4')) else 16 if label.endswith('comb16') else 24 if label.endswith('prenorm_raw_mixes24') else 5120;dtype='torch.float32' if width in (4,16,24) else 'torch.bfloat16';size=4 if dtype=='torch.float32' else 2
  for candidate in loaded:
   m,d,samples,domain=candidate;sm=samples[label];data=member(d,sm);actual_owner=domain['cp_rank']==owner_cp
   need(sm['dtype']==dtype and sm['sample_shape']==[2,width] and sm['valid']==[False,actual_owner] and sm['required_query_position']==7168 and len(data)==2*width*size and sm['finite'][1] is True,'NOT_CAPTURED_TYPED_QUERY_OWNER_OR_WIDTH:'+label)
   if actual_owner:proof.append(dict(point=label,owner_PID=m['PID'],owner_cp_rank=owner_cp,source_local_row=owner_row,query_position=7168,width=width,original_dtype=dtype,raw_member=dict(path=str(d/sm['file']),bytes=len(data),SHA256=sm['sha256']),row_offset_bytes=width*size))
 return dict(schema='c3-p-hc-seam-four-rank-proof-v4',state='ACTUAL_HC_SEAM_QUERY_AND_PARAMETERS_CAPTURED',nonce=owner[0]['nonce'],request_id=owner[0]['request_id'],model_identity_sha256=owner[0]['model_identity_sha256_from_arm'],points=proof,parameters=parameters,query_position=7168,source_chain=expected_sources,coverage_complete=True,required_points=list(REQUIRED),coverage=dict(query_points=len(REQUIRED)-3,stage_query_layers=list(STAGE_QUERY_LAYERS),static_parameters=3,actual_query_owner_cp_rank=owner_cp,all_required_points_captured=True),allowed_claims=['original_dtype_query7168_boundary_observation_only','whole_static_parameter_bytes_preserved','independent_epoch_and_RID_admission_remain_callers_responsibility'],forbidden_claims=['operator_causality','all_tokens_or_all_KV_equal','CPU_replay_proves_GPU_numerics','formal_precision_acceptance'],formal_acceptance=False,operator_root_cause_proven=False)

def verify_supplemental_case(entries):
 primary=verify_transition_case(entries);owner=primary['points'][0]['owner_PID'];result=[];events=[]
 for entry in entries:
  p=Path(entry['path']);raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==entry['sha256'],'SUPPLEMENTAL_MANIFEST_SHA');m=json.loads(raw);seen=[]
  for label in sorted(SUPPLEMENTAL_REQUIRED):
   sample=next(x for x in m['samples'] if x['label']==label);f=p.parent/sample['file'];raw=f.read_bytes();need(len(raw)==sample['bytes']<=4*1024**2 and hashlib.sha256(raw).hexdigest()==sample['sha256'],'SUPPLEMENTAL_RAW_SHA');is_owner=m['PID']==owner
   width=5120 if label.endswith('packed_x_uint8') else 40 if '_sf_int32' in label else sample['sample_shape'][1];dtype='torch.uint8' if label.endswith('packed_x_uint8') else 'torch.int32' if '_sf_int32' in label else sample['dtype'];size={'torch.uint8':1,'torch.int32':4,'torch.int64':8,'torch.float32':4}[dtype]
   need(sample['dtype']==dtype and sample['sample_shape']==[2,width] and sample['valid']==[False,is_owner] and len(raw)==2*width*size,'SUPPLEMENTAL_TYPED_ACTUAL_CP_QUERY')
   if label.endswith('topk_weights'):need(sample['finite']==[True,True],'SUPPLEMENTAL_FINITE_ROUTE_WEIGHTS')
   if is_owner:result.append(dict(point=label,owner_PID=owner,query_position=7168,width=width,original_dtype=dtype,raw_member=dict(path=str(f),bytes=len(raw),SHA256=sample['sha256']),row_offset_bytes=width*size))
   seen.append(label)
  launch=[e for e in m['events'] if e.get('point')=='L0.MoE.actual_launch'];need(len(launch)==1 and len(seen)==5,'ONE_ACTUAL_MOE_LAUNCH_AND_ALL_POINTS_REQUIRED');events.append(dict(PID=m['PID'],launch=launch[0]))
 need(len(result)==5 and len(events)==4,'SUPPLEMENTAL_FOUR_RANK_COVERAGE');return dict(schema='c3-actual-MoE-query-input-proof-v1',state='FIVE_MOE_QUERY_INPUT_POINTS_AND_FOUR_RANK_LAUNCHES_VERIFIED',nonce=primary['nonce'],request_id=primary['request_id'],model_identity_sha256=primary['model_identity_sha256'],points=result,rank_events=events,model_requests=0,wholebatch_equal_claimed=False,operator_causality=False,formal_acceptance=False)
