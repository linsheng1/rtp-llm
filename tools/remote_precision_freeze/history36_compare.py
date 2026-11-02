"""CPU-only join of completed historical producer samples. Zero model calls.

Raw IDs and tensors are fetched directly from the seed onto this P device.
Reports expose counts, hashes and coordinates only. A matching sampled input
does not establish whole-batch, routed-expert, scale or parameter equivalence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
from urllib.request import build_opener, ProxyHandler

C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    a=p.parse_args()
    raw=a.plan.read_bytes()
    need(sha(raw)==a.sha256 and os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only frozen plan required')
    plan=json.loads(raw)
    need(sha(Path(__file__).read_bytes())==plan['script']['SHA256'],'analysis source drift')
    root=Path(plan['output_root'])
    need(root.parent==C and not root.exists(),'new owned analysis root required')
    root.mkdir(mode=0o700)
    def save(name,value):
        t=root/(name+'.partial')
        t.write_text(json.dumps(value,sort_keys=True,indent=2))
        t.replace(root/name)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('finite120s CPU join')))
    signal.alarm(120)
    save('status.json',dict(state='VERIFYING_EXISTING_EXPORTS',model_requests=0,at=time.time()))
    try:
        opener=build_opener(ProxyHandler({}))
        def fetch(pin,name,maximum=8*1024**2):
            path=Path(pin['path'])
            need(C in path.parents,'source outside task control directory')
            url='http://10.44.61.73:19091/publication-precision-20261101/control/'+str(path.relative_to(C))
            with opener.open(url,timeout=15) as response:
                data=response.read(maximum+1)
            need(len(data)<=maximum and sha(data)==pin['SHA256'],'immutable device-direct evidence drift')
            dest=root/name
            with dest.open('xb') as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            return dest
        import torch
        torch.set_num_threads(1)
        need(not torch.cuda.is_initialized(),'CPU join initialized CUDA')
        source=json.loads(fetch(plan['source_native'],'source-native-private.json').read_text())
        normal=json.loads(fetch(plan['normal_native'],'normal-native-private.json').read_text())
        need(source['response_eof_complete'] and normal['response_eof_complete'] and
             len(source['input_ids'])==7041 and len(source['output_ids'])==241 and
             len(normal['input_ids'])==7169 and len(normal['output_ids'])==143 and
             normal['input_ids_sha256']==plan['R_SHA256'],'actual native calibration geometry differs')
        sequence=source['input_ids']+source['output_ids'][:-1]
        need(normal['input_ids']==sequence[:7169],'normal R is not the observed source history')
        points,params,coverage={}, {}, {}
        for case,role,native in [('source','D',source),('normal','P',normal)]:
            pin=plan[case+'_ack']
            ack=json.loads(fetch(pin,case+'-ack.json').read_text())
            epoch=native['identity']['roles'][role]
            expected={r['PID']:r for r in epoch['GPU_ranks']}
            need(len(expected)==4 and {r['world_rank'] for r in expected.values()}=={0,1,2,3}
                 and len(ack['rank_controls'])==4 and
                 {r['PID'] for r in ack['rank_controls']}==set(expected),'actual four-rank export/epoch join differs')
            need(all(str(ack['model_root'][k])==str(epoch['model_root'][k]) for k in ['PID','UID','start_ticks']),
                 'native/export server root epoch differs')
            points[case]={}
            params[case]={}
            coverage[case]=[]
            for record in ack['rank_controls']:
                pid=record['PID']
                need(record['identity']['source_commit']==plan['candidate_commit'] and
                     record['identity']['positions']==[7042,7043] and record['identity']['layers']==[0,1,2],
                     'actual sample selector/candidate differs')
                path=Path(pin['path']).parent/'capture-bytes'/role/(str(pid)+'.pt')
                local=fetch(dict(path=str(path),SHA256=record['SHA256']),case+'-'+str(pid)+'.pt')
                data=torch.load(local,map_location='cpu',weights_only=True)
                need(bool(data)==(not record['coverage_empty']) and set(data)==set(record['hits']),
                     'typed export/hit coverage differs')
                for key,value in record['all_registered_hit_counters'].items():
                    need(value[0] in [0,1],'nonisolated repeated historical position')
                params[case][pid]=record['actual_parameters_after_EOF']
                coverage[case].append(dict(PID=pid,world_rank=expected[pid]['world_rank'],empty=record['coverage_empty'],points=len(data)))
                for key,tensor in data.items():
                    m=re.fullmatch(role+r'-L([012])-T(704[23])-(.+)',key)
                    need(m is not None and record['hits'][key]==[1,int(m[2])],'actual late GPU position/hit identity differs')
                    coordinate=(int(m[1]),int(m[2]),m[3])
                    meta=record['metadata'][key]
                    need(tensor.device.type=='cpu' and tensor.shape[0]==1 and
                         tensor.numel()==meta['shape_per_query'][0],'actual typed row geometry differs')
                    points[case].setdefault(coordinate,[]).append((tensor,pid,meta))
        def bits(tensor):
            return tensor.contiguous().view(torch.uint8).numpy().tobytes()
        def value(case,key):
            rows=points[case].get(key,[])
            if not rows:
                return None
            need(all(bits(t)==bits(rows[0][0]) and t.dtype==rows[0][0].dtype for t,_,_ in rows),
                 'same-position replicas differ: '+str(key))
            return rows[0][0]
        def compare(left,right,label):
            need(left.shape==right.shape and left.dtype==right.dtype,'paired original logical geometry/type differs: '+label)
            out=dict(point=label,elements=left.numel(),dtype=str(left.dtype),byte_equal=bits(left)==bits(right))
            if left.is_floating_point():
                l,r=left.double().reshape(-1),right.double().reshape(-1)
                mask=l!=r
                finite=torch.isfinite(l)&torch.isfinite(r)
                out.update(unequal_elements=int(mask.sum()),first_difference_column=int(torch.nonzero(mask)[0,0]) if mask.any() else None,
                           max_abs_error=float((l[finite]-r[finite]).abs().max()) if finite.any() else None,
                           nonfinite_left=int((~torch.isfinite(l)).sum()),nonfinite_right=int((~torch.isfinite(r)).sum()))
            else:
                out['unequal_elements']=int((left!=right).sum())
            return out
        token_checks={}
        for case in ['source','normal']:
            for pos in [7042,7043]:
                sampled=[value(case,(layer,pos,'actual_input_token')) for layer in [0,1,2]]
                sampled=[x for x in sampled if x is not None]
                need(sampled and all(x.numel()==1 and int(x.reshape(-1)[0])==sequence[pos] for x in sampled),
                     'sampled GPU input token does not join actual native history')
                token_checks[case+':'+str(pos)]=dict(actual_GPU_token_matches_native_history=True,observed_layers=len(sampled))
        order=[(0,'attn_input5120'),(0,'attn_output5120'),(0,'FFN_input5120'),(0,'FFN_output5120')]
        order += [(1,label) for label in ['block_entry4HC','after_Engram4HC',
            'natural_attnHC_residual4HC','natural_attnHC_pre4','natural_attnHC_post4','natural_attnHC_comb16',
            'attn_input5120','attn_output5120','seam_actual_rows','seam_residual4HC','seam_attn_out5120',
            'attnHC_pre4','attnHC_post4','attnHC_comb16','seam_fused_route','seam_post4HC',
            'FFNHC_pre4','FFNHC_post4','FFNHC_comb16','seam_FFN_norm5120',
            'natural_FFNHC_residual4HC','natural_FFNHC_pre4','natural_FFNHC_post4','natural_FFNHC_comb16',
            'natural_FFNHC_output5120','natural_FFN_norm_input5120','natural_FFN_norm_output5120',
            'FFN_input5120','FFN_output5120']]
        order += [(2,'block_entry4HC'),(2,'attn_input5120'),(2,'compressor_values512'),(2,'compressor_scores512')]
        checks=[]
        for layer,label in order:
            for pos in [7042,7043]:
                key=(layer,pos,label)
                l,r=value('source',key),value('normal',key)
                if l is None or r is None:
                    checks.append(dict(point=label,layer=layer,position=pos,state='NOT_CAPTURED_BOTH_PATHS'))
                else:
                    item=dict(compare(l,r,label),layer=layer,position=pos,state='COMPARED')
                    if label in ['seam_actual_rows','seam_fused_route']:
                        item.update(source_value=int(l.reshape(-1)[0]),normal_value=int(r.reshape(-1)[0]),control_variable=True)
                    if label.startswith('FFNHC_') or label in ['seam_FFN_norm5120','seam_post4HC']:
                        item['ordering_scope']='same MegaMHC call outputs; internal order/causality not established'
                    checks.append(item)
        for pos in [7042,7043]:
            raw=value('source',(2,pos,'SWA_raw528'))
            normal_swa=value('normal',(2,pos,'fresh_SWA_decoded512'))
            if raw is not None and normal_swa is not None:
                need(raw.dtype==torch.uint8 and raw.numel()==528,'original SWA528 format required')
                flat=raw.reshape(-1)
                payload=flat[:512].contiguous().view(torch.float8_e4m3fn).float().reshape(16,32)
                exponent=flat[512:].to(torch.int32)-127
                decoded=torch.ldexp(payload,exponent[:,None]).to(torch.bfloat16).reshape(1,512)
                checks.append(dict(compare(decoded,normal_swa,'SWA_decoded512'),layer=2,position=pos,state='COMPARED',
                                   original_D_format='512E4M3+16UE8M0 group32, decoded on CPU'))
        key=(2,7043,'GLOBAL_raw288')
        l,r=value('source',key),value('normal',key)
        if l is not None and r is not None:
            checks.append(dict(compare(l,r,'GLOBAL_raw288'),layer=2,position=7043,state='COMPARED'))
        parameter_checks=[]
        for key in sorted(set.intersection(*(set(v) for group in params.values() for v in group.values()))):
            members=[v[key] for group in params.values() for v in group.values()]
            parameter_checks.append(dict(parameter=key,all_eight_logical_parameter_fingerprints_equal=all(v==members[0] for v in members)))
        residual_checks=[]
        for layer in [0,1,2]:
            for pos in [7042,7043]:
                left,right=value('source',(layer,pos,'block_entry4HC')),value('normal',(layer,pos,'block_entry4HC'))
                if left is not None and right is not None:
                    residual_checks.append(dict(compare(left,right,'block_entry4HC'),layer=layer,position=pos,
                        ordering_scope='UNORDERED: cross-layer/Engram prepared branch may place entry before or after fusion'))
        kernel_policies={case:document['identity']['precision_config'] for case,document in [('source',source),('normal',normal)]}
        actual_hc_parameters=[]
        owner_rows={case:points[case].get((1,7043,'seam_residual4HC'),[]) for case in ['source','normal']}
        need(all(owner_rows.values()),'both actual L1 seam owners required')
        for name in ['L1.attnHC.fn','L1.attnHC.base','L1.attnHC.scale','L1.FFNHC.fn','L1.FFNHC.base','L1.FFNHC.scale','L1.FFN_norm.weight']:
            values=[params[case][pid][name] for case in ['source','normal'] for _,pid,_ in owner_rows[case]]
            actual_hc_parameters.append(dict(parameter=name,all_actual_sample_owners_equal=all(v==values[0] for v in values),fingerprints=values))
        first=next((v for v in checks if v.get('state')=='COMPARED' and v['point'] not in ['seam_actual_rows','seam_fused_route'] and not v['byte_equal']),None)
        epochs={case:{role:{k:document['identity']['roles'][role]['model_root'][k] for k in ['PID','UID','start_ticks']}
                      for role in ['P','D']} for case,document in [('source',source),('normal',normal)]}
        need(not torch.cuda.is_initialized(),'CPU join initialized CUDA')
        result=dict(state='HISTORY36_EXISTING_PRODUCER_CAPTURE_COMPARED',at=time.time(),model_requests=0,
                    candidate_commit=plan['candidate_commit'],input_R_SHA256=plan['R_SHA256'],token_checks=token_checks,
                    actual_request_ids=dict(source=source['request_id'],normal=normal['request_id']),epochs=epochs,
                    coverage=coverage,checks=checks,residual_checks=residual_checks,kernel_policies=kernel_policies,first_sampled_difference=first,selected_parameter_checks=parameter_checks,actual_L1_HC_parameter_checks=actual_hc_parameters,
                    source_full_IDs_calibrated_241=True,normal_full_IDs_calibrated_143=True,
                    operator_root_cause_proven=False,whole_model_fixed=False,formal_acceptance=False,
                    limit='Natural L1 Engram/HC/RMSNorm seams added. Actual rows sampled as a GPU selected value; registration_shape metadata is not actual runtime M. MegaMHC coefficient/normalized outputs do not imply mutual causality. Per-token layer-order sampled boundaries only. Source decode and normal prefill execute tokens in different orders; block-entry Engram fusion boundaries are intentionally not ordered against each other. Equal sampled FFN input does not prove routing/expert/scales/full-batch equality. D-source and P-normal process epochs differ and are explicitly retained. No whole-source/remote/destination byte provenance claim.')
        save('comparison-safe.json',result)
        save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'comparison-safe.json'),SHA256=sha((root/'comparison-safe.json').read_bytes()))))
        print(json.dumps(dict(state=result['state'],first_sampled_difference=first,model_requests=0)))
    except BaseException as error:
        save('status.json',dict(state='FAILED',error_kind=type(error).__name__,error=str(error),model_requests=0))
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    main()
