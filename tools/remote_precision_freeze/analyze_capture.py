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
        for point in proofs['short']['points']:
            label=point['point'];pair=[];pins=[]
            for case in ['short','full']:
                row=next(x for x in proofs[case]['points'] if x['point']==label);raw,pin=comp.read(row['raw_member']['path'],row['raw_member']['SHA256'],4*1024**2,json_value=False)
                need(row['width']==point['width'] and row['original_dtype']==point['original_dtype'] and len(raw)==2*row['row_offset_bytes'],'paired typed query geometry differs')
                pair.append(raw[row['row_offset_bytes']:]);pins.append(pin)
            item=comp.numeric_report(pair[0],pair[1],point['original_dtype'],label);item['raw_members']=pins;checks.append(item)
        statics=[]
        for row in proofs['short']['parameters']:
            label=row['point'];members=[]
            for case in ['short','full']:
                r=next(x for x in proofs[case]['parameters'] if x['point']==label)
                need(r['byte_equal_across_ranks'],'same-case static parameter bytes differ')
                members.extend(read(pin) for pin in r['raw_members'])
            statics.append(dict(point=label,all_eight_original_members_byte_equal=all(raw==members[0] for raw in members),bytes=len(members[0])))
        need(all(x['all_eight_original_members_byte_equal'] for x in statics),'static parameter differs')
        ps,pf=comp.primary_rows(rows['short'],7168),comp.primary_rows(rows['full'],0);primary_checks=[]
        for layer in [2,8]:
            for pos,width in [(7168,5120),(7167,1024)]:primary_checks.append(primary.compare_rows(ps.get((layer,pos),[]),pf.get((layer,pos),[]),f'layer{layer}.width{width}.position{pos}'))
        supplement=comp.moe_supplemental(spec,observer)
        result=dict(state='COMPARED_EXISTING_FROZEN_CAPTURE',at=time.time(),analysis_plan_SHA256=a.sha256,capture_plan_SHA256=plan['capture_plan_SHA256'],candidate_commit=plan['candidate_commit'],input_SHA256=plan['input_SHA256'],actual_request_ids={k:v['request_id'] for k,v in report['results'].items()},transition_checks=checks,static_parameters=statics,primary_checks=primary_checks,MoE_supplemental=supplement,model_requests=0,ARM_creations=0,operator_root_cause_proven=False,wholebatch_equal_claimed=False,formal_acceptance=False,limit='Observed query7168 boundaries and selected STATE7167 only; equality is not full-batch/expert-row equality. Earliest observed difference alone does not prove causality.')
        with (root/'comparison-safe.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
        save(dict(state=result['state'],finished_at=time.time(),model_requests=0,comparison=dict(path=str(root/'comparison-safe.json'),SHA256=sha((root/'comparison-safe.json').read_bytes()))))
        print(json.dumps(dict(state=result['state'],model_requests=0)))
    except BaseException as e:save(dict(state='BLOCKED',error_kind=type(e).__name__,error=str(e),finished_at=time.time(),model_requests=0));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
