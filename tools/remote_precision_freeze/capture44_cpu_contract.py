"""CPU-only gather/canonical metadata contracts; synthetic data is not model evidence."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--observer',type=Path,required=True);p.add_argument('--observer-sha256',required=True);p.add_argument('--self-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==a.self_sha256 and not a.output.exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and hashlib.sha256(a.observer.read_bytes()).hexdigest()==a.observer_sha256
import torch
assert not torch.cuda.is_initialized();s=importlib.util.spec_from_file_location('exact_new_gather',a.observer);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);seen={};m._capture=SimpleNamespace(_context=SimpleNamespace(value=dict(positions=torch.tensor([7000,7045]))))
def capture(label,t,positions=None,**meta):
 if positions is None:positions=m._capture._context.value['positions']
 assert positions.shape==(1,) and t.shape[0]==1 and t.numel()*t.element_size()<=256*1024
 seen[label]=(t.clone(),positions.clone())
real_record=m.record;m.record=capture
kv=(torch.arange(647*512,dtype=torch.float32).remainder(127).reshape(647,1,512)).bfloat16();ids=torch.cat([torch.arange(7,519),torch.arange(519,647)]).int().reshape(1,640).expand(2,640).contiguous();m.selected_prefill(kv,ids,None,torch.tensor([7000,7045]),(7,512,6918));assert seen['prefill_mapping_valid'][0].all() and seen['prefill_valid'][0].all();assert torch.equal(seen['prefill_canonical'][0][0,:512],torch.arange(512)) and torch.equal(seen['prefill_canonical'][0][0,512:],torch.arange(6918,7046));values=torch.cat([seen['prefill_selected_KV_chunk'+str(j)][0].reshape(128,512) for j in range(5)]);assert torch.equal(values,kv[7:].reshape(640,512));assert sum(t.numel()*t.element_size()+8 for t,_ in seen.values())<4194304
seen.clear();m.selected_prefill(kv,ids,torch.tensor([640,0],dtype=torch.int32),torch.tensor([7000,7045]),(7,512,6918));assert not seen['prefill_valid'][0].any() and all(not seen['prefill_selected_KV_chunk'+str(j)][0].any() for j in range(5))
seen.clear();m.selected_prefill(kv,ids,None,torch.tensor([7000,7001]),(7,512,6918));assert all(int(pos[0])==-1 for _,pos in seen.values())
seen.clear();m._capture._context.value=dict(positions=torch.tensor([7045]),metadata=SimpleNamespace(req_id_per_token=torch.tensor([0])));absolute=torch.arange(6918,7046).reshape(1,128);table=torch.arange(1,21).reshape(1,20);slots=table[:,absolute.flatten()//512]*128+absolute%128;pool=torch.zeros((32,128,528),dtype=torch.uint8);m.selected_decode('SWA',pool,slots,table,512);assert seen['SWA_window_mapping_valid'][0].all() and seen['SWA_mapping_valid'][0].all() and torch.equal(seen['SWA_canonical'][0],absolute)
m.record=real_record
for role in ['P','D']:
 touched=[]
 def actual_latch(label,value,positions,**metadata):
  c=m.context();assert c['layer']==24 and c['role']==role and c['positions'].numel()==positions.numel() and value.shape[0]==positions.numel();touched.append(label)
 m._capture=SimpleNamespace(_context=SimpleNamespace(value=None),_config=dict(role=role),_capture=actual_latch)
 cp=SimpleNamespace(cp_size=4,kv_cache_sharded=True,input_lengths_global_host=(2,),global_positions=torch.tensor([7044,7045]),local_is_real=torch.tensor([True,True]))
 attn=SimpleNamespace(_cp_ctx=cp,tp_size=1,n_heads=64,compress_ratio=1,kv_source_layer_id=20,index_source_layer_id=24,is_kv_source=False,is_index_source=True)
 block=SimpleNamespace(layer_id=24,attn=attn);fn=m.block_wrapper(lambda block,*a,**kw:a[0],role)
 if role=='P':fn(block,torch.zeros((2,4,5120),dtype=torch.bfloat16),torch.tensor([1,2],dtype=torch.int32))
 else:fn(block,torch.zeros((1,2,4,5120),dtype=torch.bfloat16),SimpleNamespace(position_ids=cp.global_positions,valid_requests=torch.tensor([True])),torch.tensor([[1,2]],dtype=torch.int32))
 assert set(touched)=={'block_entry4HC','actual_input_token','actual_topology'} and m.context() is None
assert not torch.cuda.is_initialized();result=dict(state='CPU_GATHER_CANONICAL_BOUNDS_AND_NO_HIT_PASS',contracts=6,model_requests=0,CUDA_initialized=False,synthetic_only=True);assert a.output.parent==Path('/home/admin/linshuo.zjl/publication-precision-20261101/control');pending=a.output.with_suffix('.partial');pending.write_text(json.dumps(result));pending.replace(a.output);print(json.dumps(result))
