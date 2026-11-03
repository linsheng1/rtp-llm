"""Synthetic actual indexer wrapper/record/export contracts; CPU or real graph."""
import argparse,hashlib,importlib.util,json,os,socket,sys
from pathlib import Path
from types import SimpleNamespace as NS

def need(x,m):
 if not x:raise AssertionError(m)
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run(mode,config,observer,root):
 import torch
 need(not root.exists(),'new contract output');root.mkdir(mode=0o700);old_full=torch.full;old_arange=torch.arange;old_sync=torch.cuda.synchronize;old_capture=torch.cuda.is_current_stream_capturing
 if mode=='cpu':
  need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU hidden')
  class Boundary(torch.Tensor):
   @property
   def is_cuda(self):return True
  mark=lambda t:t.as_subclass(Boundary);torch.full=lambda *a,**kw:mark(old_full(*a,**kw));torch.arange=lambda *a,**kw:mark(old_arange(*a,**kw));torch.cuda.synchronize=lambda device=None:need(device is None or device.type=='cpu','CPU sync only');torch.cuda.is_current_stream_capturing=lambda:False;device='cpu'
 else:mark=lambda t:t;device='cuda'
 try:
  reports=[]
  for role in ['P','D']:
   m=module(observer,'exact_h45_'+role);cfg=dict(config['capture'],role=role,output=str(root/role));cap=m.install(cfg);sys.meta_path.pop(0)
   def make(shape,dtype,value=0):return mark(torch.full(shape,value,dtype=dtype,device=device))
   n=257 if role=='P' else 1
   positions=mark(torch.arange(n,device=device,dtype=torch.int32)+6945) if role=='P' else make((1,),torch.int32)
   # Warm with a no-hit position, then mutate the very same storage for replay.
   if role=='P':positions.sub_(10000)
   cp=NS(cp_size=4,kv_cache_sharded=True,input_lengths_global_host=(n,),global_positions=positions,local_is_real=make((n,),torch.bool,True))
   tiny=make((2,),torch.float32,1);unit=NS(weight=tiny,weight_scales=tiny)
   attn=NS(_cp_ctx=cp,tp_size=1,n_heads=64,compress_ratio=1,kv_source_layer_id=20,index_source_layer_id=24,is_kv_source=False,is_index_source=True,index_wq=unit,index_weights=tiny,v41_config=dict(candidate_source_layer_id=20,candidate_topk_blocks=2048,candidate_block_size=8),_shared_attention={})
   block=NS(layer_id=24,attn=attn);x=make((n,4,5120) if role=='P' else (1,n,4,5120),torch.bfloat16,3);tokens=make((n,) if role=='P' else (1,n),torch.int32,123);metadata=NS(position_ids=positions,valid_requests=make((1,),torch.bool,True));q=make((n,32,64),torch.int8,17);sf=make((n,32),torch.int32,0x01010101);weights=make((n,32),torch.bfloat16,2)
   if role=='P':
    k=make((7169,64),torch.int8,17);ks=make((7169,),torch.int32,0x01010101)
    score=m.prefill_score_wrapper(lambda *a,**kw:make((n,7169),torch.bfloat16,1))
    def calculate(attn,*a):score(q,sf,k,ks,weights,positions+1,return_bf16=True);return make((n,512),torch.int32,7)
   else:
    pool=make((32,512,68),torch.uint8,17);flat=pool.reshape(32,-1);flat[:,512*64:]=1;table=mark(torch.arange(1,33,device=device,dtype=torch.int32).reshape(1,32));paged=m.paged_score_wrapper(lambda *a,**kw:make((n,16384),torch.bfloat16,1))
    def underlying(*a,**kw):return paged(q.reshape(1,n,32,64),sf.reshape(1,n,32),pool,weights,table,(positions+1).reshape(1,n),512,16384,return_bf16=True)
    score=m.decode_score_wrapper(underlying)
    def calculate(attn,*a):score();return make((n,512),torch.int32,7)
   select=m.selection_wrapper(calculate,role)
   wrapped=m.block_wrapper(lambda block,*a:select(attn),role)
   def call():return wrapped(block,x,tokens) if role=='P' else wrapped(block,x,metadata,tokens)
   def control(op,label=''):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
     client.settimeout(20);client.connect(str(root/role/(str(os.getpid())+'.sock')));client.sendall((json.dumps(dict(operation=op,label=label,request_drained=True))+'\n').encode());raw=b''
     while not raw.endswith(b'\n'):
      part=client.recv(65536);need(bool(part),'control EOF');raw+=part
    return json.loads(raw)
   with torch.inference_mode():
    call();need(m.context() is None,'actual block context restored')
    if mode=='gpu' and role=='D':
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):answer=call()
     execute=graph.replay
    else:execute=call
    need(control('reset')['state']=='CAPTURE_LATCHES_RESET','inference reset');execute();empty=control('snapshot','empty');need(empty['coverage_empty'],'no-hit export')
    need(control('reset')['state']=='CAPTURE_LATCHES_RESET','selected reset')
    if role=='P':positions.add_(10000)
    else:positions.fill_(7045)
    execute();out=control('snapshot','selected');need(out['state']=='CAPTURE_EXPORTED_NOT_NUMERICAL_ACCEPTANCE',str(out));path=Path(out['file']);need(hashlib.sha256(path.read_bytes()).hexdigest()==out['SHA256'],'real export SHA');data=torch.load(path,map_location='cpu',weights_only=True);prefix=role+'-L24-T7045-';required=['block_entry4HC','actual_input_token','actual_topology','indexer_q_payload','indexer_q_packed_sf','indexer_scaled_head_weights','indexer_visible','indexer_position_in_actual_block','indexer_key_valid','indexer_logits','indexer_selected_logical']+['indexer_key_raw_chunk'+str(i) for i in range(4)];need(set(prefix+k for k in required)<=set(data),'complete actual indexer point chain');need(all(v==[1,7045] for v in out['hits'].values()),'one query hit');get=lambda k:data[prefix+k]
    need(get('indexer_position_in_actual_block').all() and get('indexer_key_valid').all() and get('indexer_visible').item()==7046,'positions/visible/physical table mapping');keys=torch.cat([get('indexer_key_raw_chunk'+str(i)).reshape(-1,68) for i in range(4)]);need(keys.shape==(7046,68) and (keys[:,:64]==17).all() and (keys[:,64:]==1).all(),'all7046 actual planar/gathered keys exact');need(get('indexer_q_payload').shape==(1,2048) and (get('indexer_q_payload')==17).all(),'actual query packing');need(get('indexer_q_packed_sf').shape==(1,32) and (get('indexer_scaled_head_weights')==2).all(),'actual query scales/weights');need(get('indexer_logits').shape==(1,7046) and (get('indexer_logits')==1).all(),'exact logical score prefix');need(get('indexer_selected_logical').shape==(1,512) and (get('indexer_selected_logical')==7).all(),'selection returned unchanged');need(out['actual_parameters_after_EOF'],'original parameter export hook');used=sum(t.numel()*t.element_size()+8 for t,h in cap._buffers.values());need(used<=4194304 and len(cap._buffers)<=192,'original latch budget');reports.append(dict(role=role,points=len(data),latch_bytes=used,source_SHA256=out['SHA256'],query_rows=n,GPU_graph=mode=='gpu' and role=='D',keys=7046))
  need(mode!='cpu' or not torch.cuda.is_initialized(),'CPU must not initialize CUDA');result=dict(state='ACTUAL_INDEXER_CONTROL_CHAIN_PASS',mode=mode,roles=reports,synthetic_only=True,model_requests=0,whole_model_fixed=False);p=root/'result-safe.partial';p.write_text(json.dumps(result,indent=2));p.replace(root/'result-safe.json');return result
 finally:torch.full=old_full;torch.arange=old_arange;torch.cuda.synchronize=old_sync;torch.cuda.is_current_stream_capturing=old_capture

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--mode',choices=['cpu','gpu'],required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--observer',type=Path,required=True);p.add_argument('--observer-sha256',required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();need(hashlib.sha256(a.config.read_bytes()).hexdigest()==a.config_sha256 and hashlib.sha256(a.observer.read_bytes()).hexdigest()==a.observer_sha256,'actual CPU input identity');print(json.dumps(run(a.mode,json.loads(a.config.read_text()),a.observer,a.output_root)))
if __name__=='__main__':main()
