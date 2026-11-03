"""One existing native source request plus finite existing-latch barriers.

Only device-to-device capture bytes use this task's existing metadata server.
No model start, RPC retry, new scheduler, or control-host tensor transfer.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

C = Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
HOSTS = dict(P='10.44.12.175', D='10.44.61.71')
CORE_SHA = 'a425ba18e22e5fe639a36237cc943cda94c2dbe57a822ca04bc9a68a1432a528'
p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--action', choices=['d-source'], required=True)
p.add_argument('--tag', required=True)
p.add_argument('--port', type=int, default=19107)
p.add_argument('--seconds', type=int, default=900)
p.add_argument('--request-timeout-seconds', type=float, default=180)
a = p.parse_args()
assert os.getuid() == 500 and a.root.parent == C and a.port == 19107
assert 0 < a.seconds <= 900 and re.fullmatch('[A-Za-z0-9_-]{1,40}', a.tag)
a.root.mkdir(mode=0o700)
deadline = time.time()+a.seconds
got, ready, failures = {}, set(), {}
acks = {'RESET': {}, 'EXPORT': {}}
phase, rid = 'WAIT', 0
actual_requests = 0


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = a.root/name
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2))
    temporary.replace(path)


def status(state, **fields):
    save('status.json', dict(state=state, at=time.time(), model_requests=actual_requests,
                             capture_phase=phase, **fields))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def response(self, code, value):
        self.send_response(code)
        self.end_headers()
        self.wfile.write(json.dumps(value).encode())

    def role(self, role):
        assert role in HOSTS and self.client_address[0] == HOSTS[role]
        return role

    def body(self, maximum):
        size = int(self.headers['Content-Length'])
        assert 0 < size <= maximum
        raw = self.rfile.read(size)
        assert len(raw) == size
        return raw

    def do_GET(self):
        try:
            if self.path.startswith('/capture/'):
                self.role(self.path.rsplit('/',1)[1])
                self.response(200,dict(action=phase,request_id=rid))
            else:
                admitted = len(got)==2 if self.path=='/all-probes' else ready=={'P','D'}
                self.response(200 if admitted else 409,{})
        except Exception as error:
            self.response(400,dict(error=str(error)))

    def do_POST(self):
        try:
            if self.path.startswith('/ready/'):
                role=self.role(self.path.rsplit('/',1)[1])
                assert json.loads(self.body(1000))['role']==role
                ready.add(role)
                self.response(200,{})
                return
            if self.path.startswith('/failed/'):
                role=self.role(self.path.rsplit('/',1)[1])
                failures[role]=json.loads(self.body(8000))
                self.response(200,{})
                return
            if self.path.startswith('/snapshot-bytes/'):
                _,_,role,pid=self.path.split('/')
                self.role(role)
                assert phase=='EXPORT' and pid.isdecimal()
                source=json.loads(Path(got[role]['path']).read_text())
                assert int(pid) in [r['PID'] for r in source['GPU_ranks']]
                expected=self.headers['X-Content-SHA256']
                assert re.fullmatch('[0-9a-f]{64}',expected)
                raw=self.body(8*1024**2)
                assert digest(raw)==expected
                root=a.root/'capture-bytes'/role
                root.mkdir(mode=0o700,parents=True,exist_ok=True)
                destination=root/(pid+'.pt')
                assert not destination.exists()
                temporary=destination.with_suffix('.partial')
                with temporary.open('xb') as out:
                    out.write(raw)
                    out.flush()
                    os.fsync(out.fileno())
                assert temporary.stat().st_size==len(raw) and digest(temporary.read_bytes())==expected
                os.link(temporary,destination)
                temporary.unlink()
                self.response(200,dict(path=str(destination),SHA256=expected,bytes=len(raw)))
                return
            if self.path.startswith('/capture-ack/'):
                role=self.role(self.path.rsplit('/',1)[1])
                raw=self.body(2*1024**2)
                value=json.loads(raw)
                action=value['action']
                assert action==phase and role==value['role'] and role not in acks[action]
                if action=='EXPORT':
                    for record in value['rank_controls']:
                        path=a.root/'capture-bytes'/role/(str(record['PID'])+'.pt')
                        assert path.stat().st_size==record['bytes'] and digest(path.read_bytes())==record['SHA256']
                path=a.root/(action+'-'+role+'-ack.json')
                with path.open('xb') as out:
                    out.write(raw)
                    out.flush()
                    os.fsync(out.fileno())
                acks[action][role]=dict(path=str(path),SHA256=digest(raw))
                self.response(200,{})
                return
            role=self.role(self.path.strip('/'))
            assert role not in got
            raw=self.body(200000)
            value=json.loads(raw)
            assert value['role']==role and value['host']==HOSTS[role]
            assert value['state']=='CURRENT_V7_MODEL_NATIVE_ARTIFACTS_AND_PROCESS_OBSERVED'
            assert -2 <= time.time()-value['observed_at'] <= 45
            path=a.root/(role+'.json')
            path.write_bytes(raw)
            got[role]=dict(path=str(path),bytes=len(raw),SHA256=digest(raw))
            self.response(200,got[role])
        except Exception as error:
            self.response(400,dict(error=str(error)))


server=HTTPServer(('0.0.0.0',a.port),Handler)
server.timeout=2
status('RECEIVING_CONTROL_METADATA')
try:
    while len(got)<2:
        assert not failures and time.time()<deadline, 'actual source probe dependency failed'
        server.handle_request()
    threading.Thread(target=server.serve_forever,daemon=True).start()
    tool=C/'c3_hc031_capture41_core_v1.py'
    assert digest(tool.read_bytes())==CORE_SHA
    case=C/'airline4-native-case-v1/case-private.json'
    case_sha='0e2f29a837287d6201c27e005c4f2f70ce40d41885235b3ce8075738a6c13753'
    assert digest(case.read_bytes())==case_sha
    ep=C/'airline4-cpu-driver-v1/environment.json'
    assert digest(ep.read_bytes())=='d5075b5c7ef68c5d1c1a4960f970c677a64d64227360c30505883abe2a39a2cc'
    edoc=json.loads(ep.read_text())
    env=dict(edoc['environment'])
    assert env['CUDA_VISIBLE_DEVICES']==''
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(C/'inner-replay-tools-v2')+':'+str(C)+':'+env['PYTHONPATH'])
    prefix=[edoc['interpreter'],'-B','-S',str(tool)]
    def call(name, argv, timeout):
        with (a.root/(name+'.stdout')).open('xb') as out,(a.root/(name+'.stderr')).open('xb') as err:
            result=subprocess.run(argv,env=env,stdout=out,stderr=err,timeout=timeout)
        save(name+'-exit.json',dict(actual_child_returncode=result.returncode))
        assert result.returncode==0, name+' actual CLI failed; inspect native request count before any retry'
    status('IDENTITY_BUILDING_ZERO_RPC')
    ident=a.root/'identity'
    call('identity',prefix+['identity','--action','d-source','--p-source',got['P']['path'],'--p-source-sha256',got['P']['SHA256'],
         '--d-source',got['D']['path'],'--d-source-sha256',got['D']['SHA256'],'--request-timeout-seconds',str(a.request_timeout_seconds),'--output-dir',str(ident)],60)
    ip=ident/'identity.json'
    identity_sha=digest(ip.read_bytes())
    common=['run','--action','d-source','--case-private',str(case),'--case-sha256',case_sha,
            '--identity-json',str(ip),'--identity-sha256',identity_sha,'--timeout-seconds',str(a.request_timeout_seconds)]
    prepare=a.root/'prepare'
    call('prepare',prefix+common+['--output-dir',str(prepare)],30)
    prep=json.loads((prepare/'summary.json').read_text())
    assert prep['state']=='INNER_REPLAY_NATIVE_PROTOBUF_CHECK_NO_REQUEST' and prep['model_requests']==0
    assert prep['input_tokens']==7041 and prep['input_ids_sha256']=='22510597b00881511fec56f072a5ef0ed3593b8a32b069e4189f0d75f258e49c'
    phase='RESET'
    status('CAPTURE41_RESET_AND_CURRENT_EMPTY_PROOF')
    phase_deadline=min(deadline,time.time()+150)
    while set(acks['RESET'])!={'P','D'}:
        assert not failures and time.time()<phase_deadline, 'finite reset/empty proof dependency failed'
        time.sleep(.2)
    proof=json.loads(Path(acks['RESET']['P']['path']).read_text())['fresh_empty_proof']
    import sys
    sys.path[:0]=[str(C),str(C/'inner-replay-tools-v2')]
    spec=importlib.util.spec_from_file_location('capture41_actual_core',tool)
    core=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.fresh_check(proof,core.read(ip,identity_sha),time.time())
    save('fresh-empty-proof.json',proof)
    with (a.root/'request-once.json').open('x') as marker:
        json.dump(dict(identity_SHA256=identity_sha,case_SHA256=case_sha),marker)
        marker.flush()
        os.fsync(marker.fileno())
    result=a.root/'request'
    actual_requests='SEE_DEVICE_NATIVE_STATUS'
    status('ACTUAL_D_SOURCE_REQUEST_EXECUTING',actual_count_source='device native/status.json')
    call('request',prefix+common+['--execute','--output-dir',str(result)],210)
    actual_requests=1
    summary=json.loads((result/'summary.json').read_text())
    assert summary['state']=='COMPLETE_INNER_NATIVE_REQUEST_EVIDENCE_PENDING' and summary['model_requests']==1
    native=json.loads((result/'native/result.json').read_text())
    rid=native['request_id']
    reference=C/'hc031-source-source1/request/native/result.json'
    assert digest(reference.read_bytes())=='5ce06d9e050da0c6581ea3dfa0315fa6d38d98791dd4b41ba5cabe73ee2eb0aa'
    old=json.loads(reference.read_text())
    equal=native['output_ids']==old['output_ids']
    first=next((i for i,(x,y) in enumerate(zip(native['output_ids'],old['output_ids'])) if x!=y),
               min(len(native['output_ids']),len(old['output_ids']))) if not equal else None
    calibration=dict(complete_EOF=native['response_eof_complete'],input_equal=native['input_ids']==old['input_ids'],
       full_output_IDs_equal=equal,output_tokens=len(native['output_ids']),reference_output_tokens=len(old['output_ids']),
       first_difference_zero_based=first)
    save('source-calibration-safe.json',calibration)
    phase='EXPORT'
    status('CAPTURE41_EXPORT_AFTER_NATIVE_EOF',request_id=rid)
    phase_deadline=min(deadline,time.time()+150)
    while set(acks['EXPORT'])!={'P','D'}:
        assert not failures and time.time()<phase_deadline, 'finite EOF export dependency failed'
        time.sleep(.2)
    assert calibration['complete_EOF'] and calibration['input_equal'] and equal and calibration['output_tokens']==295, \
           'source IDs not exactly calibrated; dependent normal capture blocked'
    phase='DONE'
    status('D_SOURCE_COMPLETE',summary=summary,probes=got,source_calibration=calibration,capture_barriers=acks,
           CPU_prepare_returncode=0,finished_at=time.time())
    print(json.dumps(dict(state='D_SOURCE_COMPLETE',model_requests=1,output_tokens=295)))
except BaseException as error:
    phase='BLOCKED'
    status('FAILED_DEPENDENTS_BLOCKED',error_kind=type(error).__name__,error=str(error),
           actual_count_source='inspect request/native/status.json if request-once exists',finished_at=time.time())
    raise
finally:
    # serve_forever is only started after the two probes arrive.
    if len(got)==2:
        server.shutdown()
    server.server_close()
