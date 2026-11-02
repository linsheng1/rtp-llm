"""Read-only CPU adapter: verified frozen capture -> existing raw comparators.
No model/RPC/ARM/cache actions. Raw tensor bytes never leave the P device.
"""
import argparse,hashlib,importlib.util,json,os,signal,time
from pathlib import Path
from urllib.request import urlopen,ProxyHandler,build_opener
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(ok,why):
    if not ok:raise ValueError(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(pin):
    p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and p.stat().st_uid==500 and p.stat().st_size<=4*1024**2,'owned bounded control evidence required')
    before=p.stat();raw=p.read_bytes();after=p.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns) and sha(raw)==pin['SHA256'],'immutable evidence SHA differs: '+str(p));return raw

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
    plan=json.loads(read(dict(path=str(a.plan),SHA256=a.sha256)))
    need(os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','P UID500 CPU-only required')
    need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'analysis source differs')
    root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new analysis output root required');root.mkdir(mode=0o700)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite90s readonly analysis')));signal.alarm(90)
    def save(value):
        path=root/'status.json';t=root/'status.partial';t.write_text(json.dumps(value,sort_keys=True,indent=2));t.replace(path)
    save(dict(state='VERIFYING_FROZEN_CAPTURE',PID=os.getpid(),started_at=time.time(),model_requests=0))
    try:
        modules={}
        for name,pin in plan['helpers'].items():
            read(pin);spec=importlib.util.spec_from_file_location('capture_analysis_'+name,pin['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);modules[name]=m
        comp,observer,primary=[modules[k] for k in ['comparator','observer','primary']]
        opener=build_opener(ProxyHandler({}))
        def fetch(pin,name):
            path=Path(pin['path']);need(C in path.parents,'metadata outside task control root')
            url='http://10.44.61.73:19091/publication-precision-20261101/control/'+str(path.relative_to(C))
            with opener.open(url,timeout=12) as response:raw=response.read(4*1024**2+1)
            need(len(raw)<=4*1024**2 and sha(raw)==pin['SHA256'],'device-direct metadata SHA differs')
            local=root/name
            with local.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            return json.loads(raw)
        report=fetch(plan['report'],'actual-request-report.json')
        need(report['plan_SHA256']==plan['capture_plan_SHA256'] and report['candidate_commit']==plan['candidate_commit'] and report['model_requests_completed']==2 and report['all_input_ids_identical'] and report['all_generation_profiles_identical'],'capture/native calibration differs')
        need(report['full_ids']['new_normal_equals_reference'] and report['full_ids']['new_Dremote_equals_reference'],'same-candidate full-ID behavior not reproduced')
        batch=json.loads(read(plan['batch']));need(batch['p_plan_sha256']==plan['p_plan_SHA256'],'actual batch model plan differs')
        config_path=Path(batch['p_model_root'])/'observer-runtime-config.json';config=json.loads(read(dict(path=str(config_path),SHA256=batch['observer_config_sha256'])))
        proofs={};rows={};spec={};barriers={}
        for i,label in enumerate(['short','full']):
            case=config['cases'][i];need(case['nonce']==batch[label+'_nonce'] and case['input_ids_sha256']==plan['input_SHA256'] and case['plan_sha256']==batch['p_plan_sha256'],'actual observer case/input differs')
            barrier=fetch(report['barriers'][label]['proof'],label+'-barrier.json');barriers[label]=barrier
            ref=json.loads(Path(case['identity_ref']).read_text());model=ref['model']
            need(model['epoch']==dict(host='10.44.12.175',role='P',**report['epochs']['P']) and model['plan_sha256']==batch['p_plan_sha256'] and barrier['model_identity_sha256']==ref['model_identity_sha256']==comp.digest(model),'actual request/source model identity differs')
            need(barrier['nonce']==case['nonce'] and barrier['request_id']==report['results'][label]['request_id'] and len(barrier['ranks'])==4 and {r['world_rank'] for r in barrier['ranks']}=={0,1,2,3},'four-rank native RID binding differs')
            entries=[];loaded=[]
            for rank in barrier['ranks']:
                need(rank['snapshot_tensor_references_released'] and rank['terminal_event_query_completed'],'original capture not drained')
                raw=read(rank['manifest']);m=json.loads(raw);path=Path(rank['manifest']['path'])
                need(m['nonce']==case['nonce'] and m['request_id']==barrier['request_id'] and m['input_ids_sha256']==plan['input_SHA256'] and m['model_identity_sha256_from_arm']==ref['model_identity_sha256'] and all(str(m[k])==str(rank[k]) for k in ['PID','UID','start_ticks']),'actual manifest epoch/RID differs')
                for key in ['live_rid_marker','drain_marker']:read(rank[key])
                entries.append(dict(path=str(path),sha256=rank['manifest']['SHA256']));loaded.append((m,path,rank['manifest']))
            spec[label]=entries;rows[label]=loaded
            proof=observer.verify_transition_case(entries)
            old=json.loads(read(barrier['transition_capture']['report']))
            need(proof==old and proof['coverage_complete'] and proof['request_id']==barrier['request_id'],'original four-rank verifier changed or coverage differs')
            proofs[label]=proof
        need(proofs['short']['model_identity_sha256']==proofs['full']['model_identity_sha256'],'paired captures use different model')
        checks=[]
        for suffix,width in [('original_values512',512),('original_scores512',512),('raw_after',1024)]:
            label='2._write_states.'+suffix;members={}
            for case in ['short','full']:
                selected=[]
                for manifest,path,mp in rows[case]:
                    for sample in manifest['samples']:
                        if sample['label']!=label:continue
                        need(sample['sample_shape']==[2,width] and len(sample['valid'])==2 and all(type(x)is bool for x in sample['valid']),'actual original projection query shape/mask required')
                        if not sample['valid'][1]:continue
                        need(sample['dtype']=='torch.float32','original FP32 projection/state required');member=path.parent/sample['file'];need(member.parent==path.parent and member.name==sample['file'],'raw member basename required')
                        raw,pin=comp.read(str(member),sample['sha256'],1024**2,json_value=False);need(len(raw)==sample['bytes']==2*width*4,'full two-position FP32 member size');selected.append((raw[width*4:],pin,manifest['PID']))
                need(bool(selected),'actual query7168 projection coverage absent');members[case]=selected
            pairs=[]
            for a,ap,arank in members['short']:
                for b,bp,brank in members['full']:
                    item=comp.numeric_report(a,b,'torch.float32',label);item.update(raw_members=[ap,bp],left_rank_PID=arank,right_rank_PID=brank);pairs.append(item)
            checks.append(dict(point=label,query_position=7168,left_valid_rows=len(members['short']),right_valid_rows=len(members['full']),all_valid_owner_pairs_byte_equal=all(x['byte_equal'] for x in pairs),pairs=pairs))
        branches={case:[event for manifest,_,_ in rows[case] for event in manifest['events'] if event.get('point')=='2._produce_global'] for case in ['short','full']}
        result=dict(state='COMPARED_EXISTING_FROZEN_CAPTURE',at=time.time(),analysis_plan_SHA256=a.sha256,capture_plan_SHA256=plan['capture_plan_SHA256'],candidate_commit=plan['candidate_commit'],input_SHA256=plan['input_SHA256'],actual_request_ids={k:v['request_id'] for k,v in report['results'].items()},projection_checks=checks,actual_producer_branches=branches,model_requests=0,ARM_creations=0,operator_root_cause_proven=False,wholebatch_equal_claimed=False,formal_acceptance=False,limit='Only already-captured L2 query7168 projection values/scores and STATE store; does not compare Q, selected KV history, indices, attention sink or MLA/output-projection internals.')
        with (root/'comparison-safe.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
        save(dict(state=result['state'],finished_at=time.time(),model_requests=0,comparison=dict(path=str(root/'comparison-safe.json'),SHA256=sha((root/'comparison-safe.json').read_bytes()))))
        print(json.dumps(dict(state=result['state'],model_requests=0)))
    except BaseException as e:save(dict(state='BLOCKED',error_kind=type(e).__name__,error=str(e),finished_at=time.time(),model_requests=0));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
