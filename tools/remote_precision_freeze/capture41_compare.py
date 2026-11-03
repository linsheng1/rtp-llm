"""Join historical7045 source D and normal P samples; no RPC/GPU/model change."""
import argparse,hashlib,json,os,re,signal
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(x,m):
 if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();raw=a.plan.read_bytes();need(sha(raw)==a.sha256 and os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','actual CPU plan');plan=json.loads(raw);need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'actual comparator');root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new result root');root.mkdir(mode=0o700)
 def save(n,x):
  p=root/(n+'.partial');p.write_text(json.dumps(x,indent=2,sort_keys=True));p.replace(root/n)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite historical CPU join')));signal.alarm(120)
 try:
  opener=build_opener(ProxyHandler({}))
  def fetch(pin,name):
   p=Path(pin['path']);need(C in p.parents,'owned data path')
   if p.exists():need(not p.is_symlink() and p.stat().st_uid==500,'owned regular data');raw=p.read_bytes()
   else:
    with opener.open('http://10.44.61.73:19091/publication-precision-20261101/control/'+str(p.relative_to(C)),timeout=20) as r:raw=r.read(8*1024**2+1)
   need(len(raw)<=8*1024**2 and sha(raw)==pin['SHA256'],'exact bounded input bytes');out=root/name
   with out.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
   return out
  import torch
  torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'CPU only');native={};samples={};coverage={};params={}
  for case,nout in [('source',295),('normal',176)]:
   n=json.loads(fetch(plan[case+'_native'],case+'-native-private.json').read_text());native[case]=n;need(n['response_eof_complete'] and len(n['output_ids'])==nout,'natural calibrated native EOF');samples[case]={};coverage[case]={};params[case]={}
   for role in ['P','D']:
    ap=plan[case+'_ack_'+role];ack=json.loads(fetch(ap,case+'-'+role+'-ack.json').read_text());epoch=n['identity']['roles'][role];ranks={r['PID']:r for r in epoch['GPU_ranks']};need(len(ranks)==4 and {r['PID'] for r in ack['rank_controls']}==set(ranks),'all4rank export');need(all(str(ack['model_root'][k])==str(epoch['model_root'][k]) for k in ['UID','PID','start_ticks']),'same native/export model epoch');points={};coverage[case][role]=[];params[case][role]={}
    for rec in ack['rank_controls']:
     cfg=rec['identity'];pid=rec['PID'];need(cfg['schema']=='history41-inner-graph-latches-v1' and cfg['source_commit']==plan['candidate_commit'] and cfg['role']==role and cfg['positions']==[7045] and cfg['layers']==list(range(26)) and cfg['gpu_budget_bytes']==4194304,'actual profile');need(all(v[0] in [0,1] for v in rec['all_registered_hit_counters'].values()) and len(rec['all_registered_hit_counters'])<=192,'actual0/1 counters')
     f=fetch(dict(path=str(Path(ap['path']).parent/'capture-bytes'/role/(str(pid)+'.pt')),SHA256=rec['SHA256']),case+'-'+role+'-'+str(pid)+'.pt');t=torch.load(f,map_location='cpu',weights_only=True);need(set(t)==set(rec['hits']) and bool(t)==(not rec['coverage_empty']),'actual typed hit coverage');need(sum(v.numel()*v.element_size()+8 for v in t.values())<=4194304,'original buffer bound');coverage[case][role].append(dict(PID=pid,world_rank=ranks[pid]['world_rank'],empty=not bool(t),points=len(t),SHA256=rec['SHA256']));params[case][role][pid]=rec['actual_parameters_after_EOF']
     for k,v in t.items():
      m=re.fullmatch(role+r'-L(\d+)-T7045-(.+)',k);need(m is not None and rec['hits'][k]==[1,7045] and v.device.type=='cpu' and v.shape[0]==1,'GPU position and unique sample');key=(int(m[1]),m[2]);need(key not in points,'one rank owner');points[key]=v
    samples[case][role]=points
  s,n=native['source'],native['normal'];sequence=s['input_ids']+s['output_ids'][:-1];need(len(s['input_ids'])==7041 and len(n['input_ids'])==7169 and n['input_ids']==sequence[:7169],'actual G+Y forcedR join');need(s['actual_native_request_profile']==n['actual_native_request_profile'],'actual generation profile')
  for role in ['P','D']:
   x,y=s['identity']['roles'][role],n['identity']['roles'][role];epoch=lambda r:tuple(str(r[k]) for k in ['UID','PID','start_ticks']);need(epoch(x['model_root'])==epoch(y['model_root']) and sorted(epoch(r) for r in x['GPU_ranks'])==sorted(epoch(r) for r in y['GPU_ranks']) and all(x[k]==y[k] for k in ['source_commit','env_SHA256','manifest_SHA256','plan_SHA256','namespace','model_service_helper_SHA256']),'same retained role epochs/runtime')
  need(not samples['source']['P'] and not samples['normal']['D'],'sourceG excludes7045 and normalD startsafter7168; otherroles must be empty');left,right=samples['source']['D'],samples['normal']['P'];need(left[0,'actual_input_token'].item()==right[0,'actual_input_token'].item()==n['input_ids'][7045]==s['output_ids'][4],'actual GPU token7045 joined to nativeIDs')
  # Actual same-layer small parameter fingerprints, independently materialized on the two role devices.
  pa={};pb={}
  for group,out in [(params['source']['D'],pa),(params['normal']['P'],pb)]:
   for perrank in group.values():
    for k,v in perrank.items():
     if k in out:need(out[k]==v,'same-role parameter replicas differ')
     out[k]=v
  shared=set(pa)&set(pb);need(shared and all(pa[k]==pb[k] for k in shared),'actual common producer parameter bytes differ');checks=[]
  for layer in range(26):
   for name in ['block_entry4HC','actual_attention_input5120']:
    key=layer,name;need(key in left and key in right,'required actual historical point missing '+str(key));x,y=left[key],right[key];need(x.dtype==y.dtype and x.shape==y.shape,'same point geometry');mask=x!=y;indices=mask.flatten().nonzero().flatten();checks.append(dict(layer=layer,point=name,byte_equal=torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)),elements=x.numel(),unequal_elements=int(mask.sum()),first_unequal_element=int(indices[0]) if indices.numel() else None,max_abs_difference=float((x.float()-y.float()).abs().max())))
  first=next((r for r in checks if not r['byte_equal']),None);result=dict(state='COMPARED_ACTUAL_HISTORICAL7045_PRODUCERS',model_requests=0,primitive_calls=0,candidate_commit=plan['candidate_commit'],GPU_tokens_joined=True,same_P_D_epochs=True,first_observed_difference=first,checks=checks,coverage=coverage,common_actual_parameter_names=sorted(shared),common_actual_parameters_byte_equal=True,whole_model_fixed=False,formal_acceptance=False,source_remote_destination_byte_equality_proven=False)
  save('result-safe.json',result);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha((root/'result-safe.json').read_bytes()))));print(json.dumps(dict(state='SUCCEEDED',first_observed_difference=first,model_requests=0)))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),model_requests=0));raise
if __name__=='__main__':main()
