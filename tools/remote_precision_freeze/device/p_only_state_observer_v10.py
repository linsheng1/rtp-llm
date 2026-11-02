"""L2 consumer inputs/core/projection add-on. Natural calls only; no eager Torch.
Parent8 keeps RID, stream, terminal, limits, raw persistence and all old points.
"""
import functools,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
PARENT=Path(__file__).with_name('p_only_state_observer_v8.py')
PARENT_SHA='9d986d662754a43681f2a7e27ce35fe9b043d8afecdfa7bc93daade266202341'
if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=PARENT_SHA:raise RuntimeError('IMMUTABLE_OBSERVER8_PARENT')
sp=importlib.util.spec_from_file_location('_c3_l2_parent8',PARENT);parent=importlib.util.module_from_spec(sp);sys.modules[sp.name]=parent;sp.loader.exec_module(parent)
base=parent.base;need=parent.need;canonical=parent.canonical;_scope=parent._scope
ATTN_SCHEMA='c3-l2-attention-readonly-v1'
CORE_POINTS=('L2.attn.Qr1280','L2.attn.Q_before_core','L2.attn.core_output','L2.attn.pre_AR5120','L2.attn.post_AR5120','L2.attn.sparse_indices')
EXTRA_CASE=dict(schema=ATTN_SCHEMA,layer=2,query_position=7168,max_selected_rows=640,max_heads=64,original_extra_limit=4*1024**2,natural_one_chunk_only=True)
def source_fields():
 return dict(extension_schema=parent.EXTENSION_SCHEMA,extension_source=parent.source_pin(Path(__file__)),base_source=parent.source_pin(parent.BASE_PATH),bootstrap_source=parent.source_pin(Path(__file__).with_name('p_only_state_observer_bootstrap_v10.py')),attention_parent_source=parent.source_pin(PARENT),attention_schema=ATTN_SCHEMA)
parent.source_fields=source_fields

def attention_budget():
 # Parent peak plus all new retained buffers, small index masks and bounded
 # point selection temporaries. KV index_select writes directly into snapshot.
 old=parent.budget_plan();added=640*512*2+640+2*(1280*2+64*512*2+32768+4096+640*4+8+2*5120*2)+8192
 peak=old['peak_bytes']+added+65536
 need(peak<=parent.EXTRA_LIMIT,'NEW_POINTS_EXCEED_ORIGINAL_EXTRA_4MIB')
 return dict(parent=old,added_retained_upper_bound=added,peak_upper_bound=peak,extra_limit=parent.EXTRA_LIMIT,total_limit=base.GPU_LIMIT)

def canonical_kv_indices(indices,length,offset,global_rows,swa_start):
 need(type(length)is int and 0<=length<=len(indices) and all(type(x)is int for x in [offset,global_rows,swa_start]),'ACTUAL_SINGLE_REQUEST_KV_METADATA')
 return [None if i>=length or x<0 else ('global',x-offset) if offset<=x<offset+global_rows else ('swa',x-offset-global_rows+swa_start) for i,x in enumerate(indices)]

class Record(parent.Record):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.attn_seen=set();self.attn_branch=None;self.attn_sources={};self.attn_parameters={};self.attn_sparse_count=0
 def completion_ready_to_record(self):
  if not super().completion_ready_to_record():return False
  required=set(CORE_POINTS)|{'L2.attn.selected_KV','L2.attn.selected_valid','L2.attn.sparse_exit'}
  if self.attn_branch=='mega':required.add('L2.attn.core_scales')
  if self.attn_branch not in ['mega','split'] or self.attn_sparse_count!=1 or not required<=self.attn_seen:self.problem='NOT_CAPTURED_L2_ACTUAL_INPUT_CORE_AR_CHAIN';return False
  return True
base.Record=Record

class Snapshots(parent.Snapshots):
 def selected_kv(self,kv,indices,lengths,cp):
  t=self.torch;need(kv.ndim==3 and tuple(kv.shape[1:])==(1,512) and kv.dtype==t.bfloat16 and kv.is_contiguous(),'ORIGINAL_BF16_KV_GEOMETRY')
  need(indices.shape[0]==cp.global_positions.numel() and indices.dtype==t.int32 and indices.ndim in (2,3),'ACTUAL_CP_SPARSE_INDEX_DOMAIN');k=indices.shape[-1];need(0<k<=640,'ACTUAL_SPARSE_WIDTH_UPPER_BOUND')
  if indices.ndim==3:need(indices.shape[1]==1,'SINGLE_SPARSE_KV_HEAD')
  ids,found=self._indices(cp.global_positions,cp.local_is_real);owner=found[1:2]
  self.reserve(k*512*2+k+128,65536+k*32)
  chosen=indices.detach().index_select(0,ids[1:2]).reshape(k)
  if lengths is None:valid=t.arange(k,device=kv.device)<k
  else:
   need(lengths.ndim==1 and lengths.shape[0]==indices.shape[0] and lengths.dtype in (t.int32,t.int64),'ACTUAL_LENGTH_DOMAIN')
   count=lengths.detach().index_select(0,ids[1:2]);valid=t.arange(k,device=kv.device)<count[0]
  valid=valid & (chosen>=0) & owner[0];in_bounds=(chosen>=0)&(chosen<kv.shape[0]);safe=chosen.clamp(0,kv.shape[0]-1).long()
  value=kv.new_zeros((1,k*512));t.index_select(kv.detach(),0,safe,out=value.view(k,1,512));value.view(k,512).masked_fill_(~(valid&in_bounds)[:,None],0)
  common=dict(metadata_container=True,attention_point=True,required_query_position=7168,finite_per_query=True,row_domain='actual_CP_query7168_original_sparse_addresses')
  self.record.add('L2.attn.selected_KV',value,owner,dict(common,dtype='torch.bfloat16',sample_shape=list(value.shape),original_shape=list(kv.shape),selected_rows=k,original_dtype='torch.bfloat16'))
  self.record.add('L2.attn.selected_valid',(valid&in_bounds).reshape(1,k),owner,dict(common,dtype='torch.bool',sample_shape=[1,k],positive_index_out_of_bounds_check_deferred_to_CPU=True))
  self.record.attn_seen.update(['L2.attn.selected_KV','L2.attn.selected_valid'])

class Observer(parent.Observer):
 def __init__(self,config,backend_factory=Snapshots,*a,**kw):
  need(config.get('l2_attention_diagnostic')==EXTRA_CASE and len(config.get('l2_attention_production_sha256',{}))==3,'EXPLICIT_L2_ATTENTION_CONSUMER_PROFILE');attention_budget();super().__init__(config,backend_factory,*a,**kw)
 def release_extension_refs(self,record):
  record.attn_parameters.clear();super().release_extension_refs(record)
 def persist(self,record):
  need(record.terminal_event is not None and record.terminal_event.query(),'ORIGINAL_TERMINAL_BEFORE_PARAMETER_FINGERPRINTS');t=record.backend.torch;parameters={};total=0
  for name,(tensor,before) in record.attn_parameters.items():
   parent.unchanged_reference(tensor,before);size=tensor.numel()*tensor.element_size();total+=size;need(size<=128*1024**2 and total<=512*1024**2,'BOUNDED_EXISTING_PARAMETER_CPU_HASH')
   cpu=tensor.detach().cpu().contiguous();raw=cpu.view(t.uint8).numpy().tobytes();parameters[name]=dict(original_dtype=str(tensor.dtype),shape=list(tensor.shape),stride=list(tensor.stride()),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());parent.unchanged_reference(tensor,before)
  record.events.append(dict(point='L2.attention.parameters',schema=ATTN_SCHEMA,branch=record.attn_branch,sources=record.attn_sources,parameters=parameters,parameter_hash_after_original_terminal=True,parameter_GPU_clones=0))
  return super().persist(record)
 def sparse_wrapper(self,original):
  inherited=super().sparse_wrapper(original)
  @functools.wraps(original)
  def wrapped(attention,*args,**kwargs):
   r=_scope.get()
   if r is None or r.problem or attention.layer_id!=2:return inherited(attention,*args,**kwargs)
   cp=self.cp(r,2);qkv=args[0] if args else kwargs['qkv'];common=args[1] if len(args)>1 else kwargs['common'];kv=kwargs['kv'];indices=kwargs['indices'];lengths=kwargs['topk_length'];shadow=[]
   def point(label,value,raw_bits=False):
    dtype=str(value.dtype)
    if raw_bits and value.element_size()==1 and value.is_floating_point():value=value.view(r.backend.torch.uint8)
    need(value.shape[0]==cp.global_positions.numel(),'ONE_NATURAL_CP_QUERY_CHUNK_REQUIRED');r.backend.point(label,value,cp,tuple(value.shape[1:]),str(value.dtype));r.attn_seen.add(label)
    r.events.append(dict(point='L2.attention.tensor_type',label=label,original_dtype=dtype,snapshot_dtype=str(value.dtype),original_shape=list(value.shape)))
   def source(name,fn):
    target=getattr(fn,'__func__',fn);p=Path(inspect.getfile(target));r.attn_sources[name]=parent.source_pin(p);pins=self.config['l2_attention_production_sha256'];matched=[key for key,wanted in pins.items() if str(p).endswith(key) and r.attn_sources[name]['SHA256']==wanted];need(len(matched)==1,'ACTUAL_ATTENTION_METHOD_SOURCE_DRIFT:'+name)
   def parameter(name,value):
    if hasattr(value,'device'):r.attn_parameters.setdefault(name,(value,parent.reference_identity(value)))
   def install(obj,name,callback):
    need(name not in obj.__dict__,'L2_METHOD_ALREADY_SHADOWED:'+name);fn=getattr(obj,name);source(name,fn);setattr(obj,name,callback(fn));shadow.append((obj,name))
   def setup():
    need(common.cp_on and common.cp_ctx is cp and qkv.qr.shape==(cp.global_positions.numel(),1280),'ACTUAL_L2_QR_CP_SHAPE')
    module=sys.modules[parent.ATTENTION];need(qkv.qr.shape[0]<=module._FLASH_MLA_SPARSE_Q_CHUNK,'ONE_ACTUAL_CHUNK_REQUIRED_NO_CHUNK_CHANGE');r.attn_sparse_count+=1
    meta=attention._shared_attention.get('prefill_chunk_meta');need(isinstance(meta,tuple) and len(meta)==3 and all(type(x)is int for x in meta),'ACTUAL_SINGLE_REQUEST_CHUNK_META_REQUIRED')
    r.events.append(dict(point='L2.attention.inputs',schema=ATTN_SCHEMA,physical_offset=meta[0],global_rows=meta[1],swa_start=meta[2],KV_shape=list(kv.shape),indices_shape=list(indices.shape),lengths_is_None=lengths is None,softmax_scale=float(attention.softmax_scale),n_heads=attention.n_heads,tp_size=attention.tp_size,head_dim=attention.head_dim,compress_ratio=attention.compress_ratio,query_rows=qkv.qr.shape[0],chunk_bound=module._FLASH_MLA_SPARSE_Q_CHUNK,actual_common_CP_identity=True))
    point('L2.attn.Qr1280',qkv.qr);point('L2.attn.sparse_indices',indices)
    if lengths is not None:point('L2.attn.topk_length',lengths.reshape(-1,1))
    r.backend.selected_kv(kv,indices,lengths,cp)
    for name in ['wq_a','wq_a_wkv','wq_b','wkv','wo_b']:
     module=getattr(attention,name,None)
     if module is not None:
      for field in ['weight','weight_scales']:parameter(name+'.'+field,getattr(module,field,None))
    for name in ['q_norm','kv_norm','attn_sink','_wo_a_stk_w','_wo_a_stk_s']:parameter(name,getattr(attention,name,None))
    def ar(fn):
     def call(out):
      r.guard(lambda:point('L2.attn.pre_AR5120',out));result=fn(out);r.guard(lambda:point('L2.attn.post_AR5120',out));return result
     return call
    install(attention,'_prefill_output_all_reduce',ar)
    mega=getattr(attention,'_mega_attention',None)
    if mega is not None:
     for name in ['query_weight','query_scale','output_weight','output_scale','cos_sin','sink']:parameter('mega.'+name,getattr(mega,name,None))
     def q(fn):
      def call(*a,**kw):
       result=fn(*a,**kw);r.guard(lambda:point('L2.attn.Q_before_core',result));return result
      return call
     def core(fn):
      def call(*a,**kw):
       r.attn_branch='mega';result=fn(*a,**kw);r.guard(lambda:point('L2.attn.core_output',result[0],True));r.guard(lambda:point('L2.attn.core_scales',result[1],True));return result
      return call
     install(mega,'project_q',q);install(mega,'prefill',core)
    def splitq(fn):
     def call(*a,**kw):
      result=fn(*a,**kw);r.guard(lambda:point('L2.attn.Q_before_core',result));return result
     return call
    def splitout(fn):
     def call(o,*a,**kw):
      r.attn_branch='split';r.guard(lambda:point('L2.attn.core_output',o,True));return fn(o,*a,**kw)
     return call
    install(attention,'_project_prefill_q',splitq);install(attention,'_prefill_output_proj_into',splitout)
   try:
    r.guard(setup);result=inherited(attention,*args,**kwargs);r.attn_seen.add('L2.attn.sparse_exit');return result
   finally:
    for obj,name in reversed(shadow):delattr(obj,name)
  return wrapped

class Dispatcher(parent.Dispatcher):
 def __init__(self,config):super().__init__(config,Observer)

def install(config):
 need(not any(n in sys.modules for n in parent.TARGETS),'BOOTSTRAP_TOO_LATE');observer=Dispatcher(config);sys.meta_path.insert(0,parent.Finder(observer))
 for case in observer.observers:
  base.evidence(case.config,'lazy-finder-installed',dict(natural_targets=list(parent.TARGETS),final_RID_required_at_startup=False,actual_model_epoch_required_at_startup=False))
  parent.extension_evidence(case.config,'extension-lazy-finder-installed',dict(natural_targets=list(parent.TARGETS),final_RID_required_at_startup=False,actual_model_epoch_required_at_startup=False))
  parent.extension_evidence(case.config,'attention-lazy-finder-installed',dict(profile=ATTN_SCHEMA,budget=attention_budget(),RID_required_at_startup=False))
 return observer

_parent_verify=parent.verify_transition_case
def verify_transition_case(entries):
 proof=_parent_verify(entries);proof['attention']=verify_attention_case(entries,proof);return proof
def verify_supplemental_case(entries):return parent.verify_supplemental_case(entries)

def verify_attention_case(entries,proof):
 """Owner/RID/domain validation plus original bytes; no token or tensor values."""
 import struct
 owner=proof['points'][0]['owner_PID'];ranks=[]
 def raw(directory,sample):
  p=directory/sample['file'];need(p.parent==directory and p.name==sample['file'] and not p.is_symlink() and p.stat().st_uid==os.getuid(),'ATTENTION_OWNED_RAW_MEMBER');data=p.read_bytes();need(len(data)==sample['bytes']<=parent.EXTRA_LIMIT and hashlib.sha256(data).hexdigest()==sample['sha256'],'ATTENTION_RAW_SHA_SIZE');return data,dict(path=str(p),bytes=len(data),SHA256=sample['sha256'])
 for entry in entries:
  p=Path(entry['path']);data=p.read_bytes();need(hashlib.sha256(data).hexdigest()==entry['sha256'],'ATTENTION_MANIFEST_SHA');m=json.loads(data);samples={x['label']:x for x in m['samples']};actual_owner=m['PID']==owner
  meta=[x for x in m['events'] if x.get('point')=='L2.attention.inputs'];params=[x for x in m['events'] if x.get('point')=='L2.attention.parameters'];need(len(meta)==len(params)==1,'ONE_ACTUAL_L2_INPUT_AND_PARAMETER_EVENT');meta,params=meta[0],params[0]
  need(meta['schema']==params['schema']==ATTN_SCHEMA and meta['actual_common_CP_identity'] and params['branch'] in ['mega','split'] and params['parameter_hash_after_original_terminal'] and params['parameter_GPU_clones']==0,'ACTUAL_ATTENTION_PROFILE_TERMINAL_CHAIN')
  need(meta['query_rows']<=meta['chunk_bound'] and meta['head_dim']==512 and meta['n_heads']<=64 and meta['compress_ratio']==2 and len(meta['KV_shape'])==3 and meta['KV_shape'][1:]==[1,512],'ACTUAL_BOUND_ATTENTION_SHAPES')
  needed=set(CORE_POINTS)|{'L2.attn.selected_KV','L2.attn.selected_valid'}
  if params['branch']=='mega':needed.add('L2.attn.core_scales')
  if not meta['lengths_is_None']:needed.add('L2.attn.topk_length')
  need(needed<=samples.keys(),'ALL_ACTUAL_ATTENTION_BOUNDARIES_REQUIRED');members={}
  for label in needed:
   sample=samples[label];value,pin=raw(p.parent,sample);selected=label in ['L2.attn.selected_KV','L2.attn.selected_valid'];need(sample['valid']==([actual_owner] if selected else [False,actual_owner]),'ACTUAL_ATTENTION_QUERY_OWNER_MASK')
   need(sample.get('required_query_position')==7168 and sample.get('finite') in [None,[True],[True,True]],'ATTENTION_QUERY_POSITION_OR_FINITE');members[label]=dict(raw=pin,dtype=sample['dtype'],sample_shape=sample['sample_shape'],original_shape=sample.get('original_shape'))
  index_sample=samples['L2.attn.sparse_indices'];idx,_=raw(p.parent,index_sample);need(index_sample['dtype']=='torch.int32' and index_sample['sample_shape'][0]==2 and index_sample['sample_shape'][1]<=640,'ACTUAL_INT32_SPARSE_QUERY');k=index_sample['sample_shape'][1];need(len(idx)==2*k*4,'TWO_INDEX_SLOTS');indices=list(struct.unpack('<'+'i'*k,idx[k*4:]));length=k
  if not meta['lengths_is_None']:
   sample=samples['L2.attn.topk_length'];value,_=raw(p.parent,sample);fmt={'torch.int32':'i','torch.int64':'q'}[sample['dtype']];size=struct.calcsize('<'+fmt);need(sample['sample_shape']==[2,1] and len(value)==2*size,'ACTUAL_LENGTH_SHAPE');length=struct.unpack('<'+fmt,value[size:])[0]
  need(0<=length<=k,'ACTUAL_LENGTH_BOUNDS')
  if actual_owner:need(all(x<meta['KV_shape'][0] for i,x in enumerate(indices) if i<length and x>=0),'ORIGINAL_POSITIVE_SPARSE_INDEX_OUT_OF_BOUNDS')
  kvs=samples['L2.attn.selected_KV'];valids=samples['L2.attn.selected_valid'];kvraw,_=raw(p.parent,kvs);validraw,_=raw(p.parent,valids);need(kvs['sample_shape']==[1,k*512] and kvs['dtype']=='torch.bfloat16' and len(kvraw)==k*512*2 and valids['sample_shape']==[1,k] and len(validraw)==k,'ORIGINAL_SELECTED_KV_WIDTH_AND_MASK')
  expected=bytes(int(actual_owner and i<length and 0<=x<meta['KV_shape'][0]) for i,x in enumerate(indices));need(validraw==expected,'EXACT_ACTUAL_READ_ROW_MASK')
  logical=canonical_kv_indices(indices,length,meta['physical_offset'],meta['global_rows'],meta['swa_start']) if actual_owner else None
  need(params['sources'] and params['parameters'] and all(len(x['SHA256'])==64 for x in params['parameters'].values()),'ACTUAL_STATIC_PARAMETER_FINGERPRINTS_REQUIRED')
  ranks.append(dict(PID=m['PID'],UID=m['UID'],start_ticks=m['start_ticks'],actual_query_owner=actual_owner,branch=params['branch'],input_metadata=meta,parameters=params['parameters'],sources=params['sources'],members=members,selected_width=k,actual_length=length if actual_owner else None,canonical_logical_indices_SHA256=hashlib.sha256(canonical(logical)).hexdigest() if actual_owner else None))
 need(len(ranks)==4 and sum(x['actual_query_owner'] for x in ranks)==1,'FOUR_RANK_SINGLE_ACTUAL_QUERY_OWNER')
 return dict(schema=ATTN_SCHEMA,state='ALL_ACTUAL_L2_ATTENTION_INPUT_CORE_AR_BOUNDARIES_CAPTURED',coverage_complete=True,request_id=proof['request_id'],nonce=proof['nonce'],model_identity_sha256=proof['model_identity_sha256'],query_position=7168,owner_PID=owner,ranks=ranks,formal_acceptance=False,operator_causality_proven=False)
