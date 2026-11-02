"""Prepared finite device peer: readonly phase probes, metadata bridge and actual barrier.
No operation without --execute. Seed coordinator alone releases max2 client calls.
"""
import argparse,hashlib,json,os,subprocess,time,signal,socket,importlib.util
from pathlib import Path
from urllib.request import Request,urlopen
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control');H=C/'inner-replay-tools-v2';N=None;SEED='http://10.44.61.73:19108';PY='/opt/conda310/bin/python3.10'
PPLAN=None;DPLAN='__EXPLICIT_D_BATCH_PLAN_REQUIRED__';DROOT=None;HELPER='717bcfa095ddeabc01d9c0cac9ff73b8f663a79eecd90b873bcc1306faac99f5'
TRANSITION_QUERY_POINTS={'L0.attnHC.prenorm_raw_mixes24': (24, 'torch.float32'), 'L0.attnHC.residual4HC': (20480, 'torch.bfloat16'), 'L0.attnHC.pre_mix4': (4, 'torch.float32'), 'L0.attnHC.post4': (4, 'torch.float32'), 'L0.attnHC.comb16': (16, 'torch.float32'), 'L0.projectedAR5120': (5120, 'torch.bfloat16'), 'L0.FFN.input_norm5120': (5120, 'torch.bfloat16'), 'L0.FFN.result5120': (5120, 'torch.bfloat16'), 'L0.ffnHC.pre_mix4': (4, 'torch.float32'), 'L0.ffnHC.post4': (4, 'torch.float32'), 'L0.ffnHC.comb16': (16, 'torch.float32'), 'L0.block.result4HC': (20480, 'torch.bfloat16'), 'L1.beforeEngram4HC': (20480, 'torch.bfloat16'), 'L1.afterEngram4HC': (20480, 'torch.bfloat16'), 'L1.HC.residual4HC': (20480, 'torch.bfloat16'), 'L1.HC.readout5120': (5120, 'torch.bfloat16'), 'L1.attn_input_norm5120': (5120, 'torch.bfloat16')}
TRANSITION_QUERY_POINTS.update({'L3.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L4.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L5.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L6.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L7.stage.attn_input_norm5120': (5120, 'torch.bfloat16'), 'L2.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L3.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L4.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L5.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L6.stage.projectedAR5120': (5120, 'torch.bfloat16'), 'L7.stage.projectedAR5120': (5120, 'torch.bfloat16')})
TRANSITION_PARAMETERS={'L0.attnHC.fn_FP32': ([1, 491520], [24, 20480]), 'L0.attnHC.scale_FP32': ([1, 3], [3]), 'L0.attnHC.base_FP32': ([1, 24], [24])}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p))
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def exchange(path,value=None):
 req=Request(SEED+path,data=None if value is None else canonical(value),headers={'Content-Type':'application/json'})
 with urlopen(req,timeout=25) as r:return json.loads(r.read(2*1024**2))
def finite_nonfinite(mask,finite):
 if type(mask) is bool:return sum_false(finite) if mask else 0
 assert isinstance(mask,list) and isinstance(finite,list) and len(mask)==len(finite),'actual finite/mask row alignment unavailable'
 return sum(finite_nonfinite(m,f) for m,f in zip(mask,finite))
def sum_false(x):
 if type(x) is bool:return int(not x)
 assert isinstance(x,list);return sum(sum_false(v) for v in x)

def verify_transition_points(case,ranks):
 path=C/'p_only_state_observer_v10.py';assert sha(path)=='1a7d13ddad273ff3814f8ee667a1acb0d4861cfeb9d339daa0dcdb7112b8ec51','frozen HC seam verifier source differs'
 spec=importlib.util.spec_from_file_location('c3_actual_transition_cpu_verifier',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 entries=[dict(path=r['manifest']['path'],sha256=r['manifest']['SHA256']) for r in ranks];proof=module.verify_transition_case(entries)
 assert proof['schema']=='c3-p-hc-seam-four-rank-proof-v4' and proof['state']=='ACTUAL_HC_SEAM_QUERY_AND_PARAMETERS_CAPTURED' and proof['nonce']==case['nonce'] and proof['query_position']==7168 and proof['coverage_complete'] is True
 expected={r['PID'] for r in ranks};assert len(proof['points'])==len(TRANSITION_QUERY_POINTS) and {r['point'] for r in proof['points']}==set(TRANSITION_QUERY_POINTS)
 for row in proof['points']:
  width,dtype=TRANSITION_QUERY_POINTS[row['point']];assert row['width']==width and row['original_dtype']==dtype and row['owner_PID'] in expected and row['query_position']==7168 and 0<=row['owner_cp_rank']<4 and row['source_local_row']>=0
  member=row['raw_member'];assert Path(member['path']).is_absolute() and member['bytes']==2*width*(4 if dtype=='torch.float32' else 2) and len(member['SHA256'])==64 and row['row_offset_bytes']==member['bytes']//2
 assert len(proof['parameters'])==3 and {r['point'] for r in proof['parameters']}==set(TRANSITION_PARAMETERS)
 for row in proof['parameters']:
  shape,original=TRANSITION_PARAMETERS[row['point']];assert row['shape']==shape and row['original_parameter_shape']==original and row['original_dtype']=='torch.float32' and row['byte_equal_across_ranks'] is True and len(row['raw_members'])==4 and {r['PID'] for r in row['raw_members']}==expected
  assert all(Path(v['path']).is_absolute() and len(v['SHA256'])==64 and v['bytes']==shape[1]*4 for v in row['raw_members'])
 supplement=module.verify_supplemental_case(entries)
 output=Path(case['evidence_dir'])/'four-rank-transition-capture-proof.json'
 with output.open('xb') as f:f.write(canonical(proof));f.flush();os.fsync(f.fileno())
 return dict(schema=proof['schema'],state=proof['state'],nonce=proof['nonce'],request_id=proof['request_id'],model_identity_sha256=proof['model_identity_sha256'],query_position=7168,coverage_complete=True,report=pin(output),verifier_source=pin(path),points=proof['points'],parameters=proof['parameters'],MoE_supplemental=supplement,attention=proof['attention'],raw_tensors_exported=False,static_parameter_semantics='original_static_parameter_no_query_axis',formal_acceptance=False)

def barrier(case,expected_rid):
 ref=json.loads(Path(case['identity_ref']).read_text());ranks=[];coverage={}
 for i,row in enumerate(ref['rank_epochs']):
  prefix='%s-%s-%s-'%(case['nonce'],row['PID'],row['start_ticks']);folder=Path(case['evidence_dir'])
  live=folder/(prefix+'live-rid-bound.json');drain=folder/(prefix+'drain-finished.json')
  l=json.loads(live.read_text());d=json.loads(drain.read_text());assert all(str(l[k])==str(row[k]) and str(d[k])==str(row[k]) for k in ['PID','UID','start_ticks']) and l['nonce']==d['nonce']==case['nonce'] and l['request_id']==expected_rid and d['snapshot_tensor_references_released'] is True
  manifest=Path(d['result']['path'])/'manifest.json';assert Path(case['output_dir']) in manifest.parents and manifest.stat().st_uid==500 and not manifest.is_symlink() and manifest.stat().st_size<=1024**2;assert sha(manifest)==d['result']['manifest_sha256'];m=json.loads(manifest.read_text());assert all(str(m[k])==str(row[k]) for k in ['PID','UID','start_ticks']) and m['nonce']==case['nonce'] and m['request_id']==expected_rid and m['model_identity_sha256_from_arm']==ref['model_identity_sha256'] and m['positions']==[7167,7168] and m['terminal_event_query_completed'] is True and not m['problem']
  assert m['extension_schema']=='c3-p-hc-seam-extension-v4' and m['extension_source']['SHA256']=='1a7d13ddad273ff3814f8ee667a1acb0d4861cfeb9d339daa0dcdb7112b8ec51' and m['base_source']['SHA256']=='d1f4bfb16039ffcc43905fe3ae8bfc0b10dbacebcd895f994200a07845c7fe70' and m['bootstrap_source']['SHA256']=='8546d5e1ed19bf1193b6803625843108c5199b8736e6a648837454a4a3b20def' and m['profile']=='P_HC_SEAM_DIAGNOSTIC_V4_NOT_ACCEPTANCE' and m['primary_compatibility_profile']=='P_STATE_DIAGNOSTIC_V2_NOT_ACCEPTANCE' and m['additional_schema']=='c3-p-hc-seam-extension-v4' and set(m['required_additional_points'])==set(TRANSITION_QUERY_POINTS)|set(TRANSITION_PARAMETERS),'actual transition manifest source/profile/required points differ'
  summaries=[]
  for sample in m['samples']:
   if sample.get('metadata_container') is True or sample.get('parameter_container') is True:continue
   valid=sample['valid'];assert isinstance(valid,list) and len(valid)==2 and all(type(v)is bool for v in valid),'actual 2-position mask required'
   nonfinite=finite_nonfinite(valid,sample['finite']) if sample['finite'] is not None else 0;assert nonfinite==0
   counts={str(p):int(v) for p,v in zip(m['positions'],valid)};label=sample['label'];summaries.append(dict(label=label,valid_count=sum(counts.values()),nonfinite_valid_count=nonfinite,per_position_valid_counts=counts))
   total=coverage.setdefault(label,{'7167':0,'7168':0})
   for pos,count in counts.items():total[pos]+=count
  # worldrank is independently joined by actual current READY epoch, not list order.
  current=json.loads((N/'model-service-status.json').read_text());rank=next(v for v in current['GPU_ranks'] if str(v['PID'])==str(row['PID']) and str(v['start_ticks'])==str(row['start_ticks']))
  ranks.append(dict(**row,world_rank=rank['world_rank'],live_rid_marker=pin(live),drain_marker=pin(drain),manifest=pin(manifest),snapshot_tensor_references_released=True,terminal_event_query_completed=True,sample_labels=sorted({x['label'] for x in summaries}),valid_count=sum(x['valid_count'] for x in summaries),nonfinite_valid_count=sum(x['nonfinite_valid_count'] for x in summaries),sample_summary=summaries))
 assert len(ranks)==4 and {r['world_rank'] for r in ranks}=={0,1,2,3}
 for layer in [2,8]:
  assert coverage.get(str(layer)+'._prefill_produce.full_hidden5120',{}).get('7168',0)>=1,'missing actual hidden7168 coverage'
  target='._read_state.original_result1024_post_gather' if case['expected_prefix_tokens']==7168 else '._write_states.raw_after'
  assert coverage.get(str(layer)+target,{}).get('7167',0)>=1,'missing actual STATE7167 coverage'
 return dict(schema='c3-p-state-four-rank-'+('short' if case['expected_prefix_tokens']==7168 else 'full')+'-drain-barrier-v2',state='FOUR_ACTUAL_RANK_'+('SHORT' if case['expected_prefix_tokens']==7168 else 'FULL')+'_CAPTURES_DRAINED',nonce=case['nonce'],request_id=expected_rid,model_identity_sha256=ref['model_identity_sha256'],observed_at=time.time(),ranks=ranks,coverage=coverage,transition_capture=verify_transition_points(case,ranks))

def process(pid):
 try:
  path=Path('/proc')/str(pid);f=(path/'stat').read_text().rsplit(')',1)[1].split()
  return dict(PID=pid,UID=path.stat().st_uid,start_ticks=f[19],state=f[0],PGID=int(f[2]),SID=int(f[3]),argv=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v])
 except (OSError,ValueError,IndexError):return None

def group_members(expected):
 rows=[]
 if expected is None:return rows
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  row=process(int(path.name))
  if row and row['PGID']==expected['PGID'] and row['SID']==expected['SID'] and row['state'] not in ['Z','X','x'] and int(row['start_ticks'])>=int(expected['start_ticks']):rows.append(row)
 return rows

class FailureLatch:
 def __init__(self):
  import threading
  self.lock=threading.RLock();self.values=[]
 def fail(self,value):
  with self.lock:self.values.append(value)
 def gate(self):
  with self.lock:
   if self.values:raise RuntimeError('upstream failure: '+str(self.values[-1]))

class OwnedRunner:
 """Finite actual own-group cleanup; RPC-inflight failure remains UNKNOWN."""
 def __init__(self,root,save,gate=lambda:None):self.root=root;self.save=save;self.gate=gate
 def run(self,argv,label,env,limit,request=False):
  self.gate();child=None;expected=None;interrupted=None;timed_out=False;remaining=[];cleanup_errors=[]
  def members():return group_members(expected)
  def signal_own(sig):
   rows=members()
   if not rows:return
   assert expected['PGID']==expected['SID']==child.pid and all(r['UID'] in [500,expected['UID']] for r in rows),'own process-group identity changed; refuse signal'
   try:os.killpg(child.pid,sig)
   except ProcessLookupError:pass
   except PermissionError as e:cleanup_errors.append(str(e))
  try:
   with (self.root/(label+'.stdout')).open('xb') as out,(self.root/(label+'.stderr')).open('xb') as err:child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True)
   expected=process(child.pid);assert expected is not None and expected['PGID']==expected['SID']==child.pid,'actual isolated own child group required';self.save(stage=label,child=expected);end=time.monotonic()+limit;next_heartbeat=0
   while child.poll() is None:
    try:self.gate()
    except BaseException as e:
     first_failure=interrupted is None;interrupted=e
     if not request:raise
     if first_failure:self.save(state='UPSTREAM_FAILED_REQUEST_INFLIGHT_UNKNOWN',model_requests='UNKNOWN_AFTER_DISPATCH',upstream_failure=str(e))
    if time.monotonic()>=end:timed_out=True;raise TimeoutError(label+' finite original child limit')
    try:child.wait(timeout=min(.25,max(.01,end-time.monotonic())))
    except subprocess.TimeoutExpired:pass
    if time.monotonic()>=next_heartbeat:self.save();next_heartbeat=time.monotonic()+20
   if interrupted is not None:raise interrupted
   self.gate();assert child.returncode==0,label+' actual child exit '+str(child.returncode);return child.returncode
  finally:
   if child is not None:
    if child.poll() is None or members():
     try:signal_own(signal.SIGTERM)
     except BaseException as e:cleanup_errors.append(type(e).__name__+': '+str(e))
     until=time.monotonic()+3
     while time.monotonic()<until and (child.poll() is None or members()):time.sleep(.05)
     if child.poll() is None or members():
      try:signal_own(signal.SIGKILL)
      except BaseException as e:cleanup_errors.append(type(e).__name__+': '+str(e))
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:cleanup_errors.append('actual own child not reaped within finite cleanup')
    remaining=members();value=dict(actual_child_returncode=child.returncode,timed_out=timed_out,child=expected,remaining_owned=remaining,cleanup_errors=cleanup_errors)
    (self.root/(label+'-exit.json')).write_bytes(canonical(value));self.save(**value)
    assert not remaining and child.returncode is not None,'own child/group remains after bounded cleanup'

def load_batch(path,expected):
 assert sha(path)==expected and not path.is_symlink() and path.stat().st_uid==500
 v=json.loads(path.read_text());assert v['schema']=='c3-p-state-pilot-batch-v3' and v['p_helper_sha256']==HELPER and len(v['d_plan_sha256'])==64 and all(c in '0123456789abcdef' for c in v['d_plan_sha256']) and Path(v['d_model_root']).parent==C and Path(v['d_model_root']).name=='model91b-D-reader-v6' and '..' not in Path(v['d_model_root']).parts
 assert v['p_probe_sha256']=='36860e75faebb673d89a0c6a3a1c0378ff46278504360b48f74907007e4bdb22' and v['p_empty_sha256']=='93b89290727331d3b2b067c23a5a4d9d349a33c19903d6929a10f8a7f5687e51' and v['d_helper_sha256']=='03d77ab5fa9e03e1f7af47ff0b049a25eb5b7df12fd6b8c8f41819d4370b7d2d' and v['d_probe_sha256']=='37e17a974492dcd98dadf0ac3c487765b8ae24e622594753d42573b2ed18b5e1','explicit batch per-role source pins differ'
 assert v['observer_extension_sha256']=='1a7d13ddad273ff3814f8ee667a1acb0d4861cfeb9d339daa0dcdb7112b8ec51' and v['observer_base_sha256']=='d1f4bfb16039ffcc43905fe3ae8bfc0b10dbacebcd895f994200a07845c7fe70' and v['observer_bootstrap_sha256']=='8546d5e1ed19bf1193b6803625843108c5199b8736e6a648837454a4a3b20def','explicit batch observer/base/bootstrap differs'
 assert v['short_nonce']!=v['full_nonce'] and all(32<=len(v[k])<=64 and all(c in '0123456789abcdef' for c in v[k]) for k in ['short_nonce','full_nonce']),'explicit distinct case nonces required'
 root=Path(v['p_model_root']);assert root.parent==C and root.name.startswith('airline4-P-') and '..' not in root.parts and len(v['p_plan_sha256'])==64 and all(c in '0123456789abcdef' for c in v['p_plan_sha256'])
 assert v['job_prefix'].startswith('airline4-P-state-pilot-') and all(c.isalnum() or c in '_-' for c in v['job_prefix'])
 assert len(v['observer_config_sha256'])==64 and all(c in '0123456789abcdef' for c in v['observer_config_sha256']),'explicit final observer config full SHA required'
 return v

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--role',choices=['P','D'],required=True);p.add_argument('--batch-json',type=Path,required=True);p.add_argument('--batch-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print(json.dumps(dict(state='PREPARED_NO_DEVICE_PHASE_ACTION',model_requests=0,role=a.role)));return 0
 global N,PPLAN,DROOT,DPLAN
 assert os.getuid()==500;batch=load_batch(a.batch_json,a.batch_sha256);N=Path(batch['p_model_root']);PPLAN=batch['p_plan_sha256'];DROOT=Path(batch['d_model_root']);DPLAN=batch['d_plan_sha256'];R=C/(batch['job_prefix']+'-peer-'+a.role);R.mkdir(mode=0o700,exist_ok=False);state=dict(state='WAITING_SEED_PHASE',model_requests=0,role=a.role,PID=os.getpid(),started_at=time.time(),interval_seconds=20);bridge=None;seen=set();until=time.monotonic()+900
 def save(**kw):
  state.update(kw,heartbeat_at=time.time());q=R/'status.json.partial';q.write_text(json.dumps(state,indent=2));q.replace(R/'status.json')
 runner=OwnedRunner(R,save)
 def run(argv,label,limit):
  runner.run(argv,label,dict(os.environ,CUDA_VISIBLE_DEVICES=''),limit)
  return R/(label+'.stdout')

 save()
 try:
  while time.monotonic()<until:
   command=exchange('/peer/'+a.role);action=command['action'];token=command['token']
   if action=='WAIT':
    dep=N if a.role=='P' else DROOT;current=json.loads((dep/'model-service-status.json').read_text());assert current['state'] not in ['FAILED','EXITED','CANCELLED','TIMED_OUT','DRAINING_OWN_MODEL'] and -2<=time.time()-current['heartbeat_at']<=45,'upstream model failed/stale while waiting'
    if bridge is not None and bridge.poll() is not None:assert bridge.returncode==0,'own ARM receiver failed while waiting'
    save();continue
   if action in ['STOP','BLOCKED']:save(state='EXITED' if action=='STOP' else 'BLOCKED',reason=command.get('reason'));break
   assert token not in seen,'repeated phase token';seen.add(token)
   root=N if a.role=='P' else DROOT;s=json.loads((root/'model-service-status.json').read_text());assert s['state']=='MODEL_READY_KV_EMPTY_PENDING' and s['model_READY'] and time.time()-s['heartbeat_at']<=45 and s['deadline_epoch']-420>time.time()+(0 if action in ['SHORT_BARRIER','FULL_BARRIER'] else 195),'actual READY/active lease cannot admit phase'
   if bridge is not None and bridge.poll() is not None:assert bridge.returncode==0,'own ARM receiver failed'
   answer=dict(role=a.role,token=token,phase=command.get('phase'),action=action)
   if action=='PROBE':
    tool=C/'c3_P_state_runtime_probe_v13.py' if a.role=='P' else C/'c3_inner_precision_replay_v6.py';expected='36860e75faebb673d89a0c6a3a1c0378ff46278504360b48f74907007e4bdb22' if a.role=='P' else '37e17a974492dcd98dadf0ac3c487765b8ae24e622594753d42573b2ed18b5e1';assert sha(tool)==expected
    argv=['sudo','-n','/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONPATH='+str(H)+':'+str(C),PY,'-B','-S',str(tool),'probe','--model-root',str(root),'--plan-sha256',PPLAN if a.role=='P' else DPLAN]
    if a.role=='P':argv+=['--helper-sha256',HELPER]
    out=run(argv,'probe-'+token,125);answer['source']=json.loads(out.read_text());answer['source_raw']=out.read_text();answer['source_SHA256']=sha(out)
    if a.role=='P' and command.get('include_startup'):
     cfg=N/'observer-runtime-config.json';answer['observer_config_raw']=cfg.read_text();answer['observer_config_SHA256']=sha(cfg);assert answer['observer_config_SHA256']==batch['observer_config_sha256'],'actual config differs from batch';answer['refs']=[dict(raw=Path(c['identity_ref']).read_text(),**pin(Path(c['identity_ref']))) for c in json.loads(cfg.read_text())['cases']]
   elif action=='EMPTY_SHORT':
    assert a.role=='P';path=C/(batch['job_prefix']+'-short-empty');tool=C/'c3_P_state_empty_probe_v13.py';assert sha(tool)=='93b89290727331d3b2b067c23a5a4d9d349a33c19903d6929a10f8a7f5687e51'
    run([PY,'-B','-S',str(tool),'--model-root',str(N),'--plan-sha256',PPLAN,'--helper-sha256',HELPER,'--namespace',json.loads((N/'plan.json').read_text())['namespace'],'--output-dir',str(path),'--seconds','120'],'empty-short',125);proof=path/'empty-proof.json';answer['fresh_proof_raw']=proof.read_text();answer['fresh_proof_SHA256']=sha(proof)
   elif action=='BRIDGE_CONFIG':
    assert a.role=='P' and bridge is None;raw=command['config_raw'];assert hashlib.sha256(raw.encode()).hexdigest()==command['config_SHA256'];cfg=json.loads(raw);assert cfg['root']==str(N/'arm-receiver') and cfg['bind_host']=='10.44.12.175' and cfg['port']==19109 and 0<cfg['seconds']<=1200
    path=R/'receiver-config.json';path.write_text(raw);tool=C/'c3_arm_http_transport_v1.py';assert sha(tool)=='05ab9d4f61aaf6636e1b418e20eff2b40ca4b4f8b316f56555203d5aaea33914'
    with (R/'receiver.stdout').open('xb') as out,(R/'receiver.stderr').open('xb') as err:bridge=subprocess.Popen([PY,'-B','-S',str(tool),'--config',str(path),'--config-sha256',sha(path)],stdout=out,stderr=err,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=False)
    end=time.monotonic()+20
    while True:
     assert bridge.poll() is None,'actual receiver startup failed';sp=N/'arm-receiver/status.json'
     if sp.exists():
      b=json.loads(sp.read_text())
      if b['state'] in ['LISTENING_METADATA_ONLY_READY_REF_MAY_BE_PENDING','WAITING_FOR_EXPLICIT_PREDECLARED_ARM']:answer['receiver_status']=b;break
     assert time.monotonic()<end,'receiver READY finite deadline';time.sleep(.2)
   elif action in ['SHORT_BARRIER','FULL_BARRIER']:
    assert a.role=='P';cfg=json.loads((N/'observer-runtime-config.json').read_text());case=cfg['cases'][0 if action=='SHORT_BARRIER' else 1];end=time.monotonic()+125
    while True:
     failures=list(Path(case['evidence_dir']).glob('*-drain-failed.json'))+list(Path(case['evidence_dir']).glob('*-capture-not-captured.json'))+list(Path(case['evidence_dir']).glob('*-identity-not-captured.json'));assert not failures,'actual observer failure marker'
     try:answer['barrier']=barrier(case,command['request_id']);break
     except FileNotFoundError:
      assert time.monotonic()<end,'four-rank short drain not complete';save();time.sleep(2)
   else:raise ValueError('unknown explicit phase action')
   exchange('/metadata/'+a.role,answer);save(state='PHASE_COMPLETED',last_completed=token)
  else:raise TimeoutError('finite peer controller lease exhausted')
 except BaseException as e:
  save(state='BLOCKED',error_kind=type(e).__name__,error=str(e));
  try:exchange('/blocked/'+a.role,dict(role=a.role,error_kind=type(e).__name__,error=str(e),token=state.get('stage')))
  except Exception:pass
  raise
 finally:
  if bridge is not None and bridge.poll() is None:
   bridge.terminate()
   try:bridge.wait(timeout=5)
   except subprocess.TimeoutExpired:state['own_receiver_remaining_PID']=bridge.pid
  save(finished_at=time.time())
 return 0
if __name__=='__main__':raise SystemExit(main())
