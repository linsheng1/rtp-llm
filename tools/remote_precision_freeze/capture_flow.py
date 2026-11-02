"""Bind the existing two-request four-rank capture protocol; no execution."""
import ast
import re
from pathlib import Path, PurePosixPath
from freeze_stage import CONTROL, HOSTS, need, read, doc, literals, option, remote_path

PYTHON = '/opt/conda310/bin/python3.10'

def build_capture(recipe):
    bench = Path(recipe['bench'])
    tag = recipe['tag']
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,40}', tag), 'invalid new batch tag')
    pins = {}
    source, pins['source_receipt'] = doc(recipe['source_receipt'])
    batch, pins['batch'] = doc(recipe['batch'])
    prepare, pins['prepare'] = doc(recipe['prepare'])
    for name, helper in recipe['helpers'].items():
        need(PurePosixPath(helper).name == helper, 'helper must be a basename')
        raw, pins[name] = read(bench / helper)
        ast.parse(raw)
    for name in ['remote_control', 'cluster23_dispatch']:
        pins[name] = read(bench / (name + '.py'))[1]
    core = literals(pins['core']['path'])
    diag = literals(pins['diagnostic']['path'])
    need(recipe['candidate_commit'] == core['B1'], 'candidate differs from existing core')
    need(source['source_SHA256'] == diag['SOURCE_SHA'] and source['EOF_complete'] and
         source['R_tokens'] == 7169 and re.fullmatch('[0-9a-f]{64}', source['R_input_ids_sha256']),
         'actual source receipt/input differs')
    for helper, expected in diag['PINS'].items():
        _, pin = read(bench / helper)
        need(pin['SHA256'] == expected, 'diagnostic dependency SHA differs: ' + helper)
        pins['dependency:' + helper] = pin
    need(batch['schema'] == 'c3-p-state-pilot-batch-v3' and
         batch['job_prefix'] == 'airline4-P-state-pilot-' + tag, 'new batch identity differs')
    for key in ['short_nonce', 'full_nonce', 'observer_config_sha256']:
        need(re.fullmatch('[0-9a-f]{32,64}', batch[key]), 'invalid frozen ' + key)
    need(batch['short_nonce'] != batch['full_nonce'], 'distinct one-use cases required')
    for field, key in [('p_helper_sha256','model_P'), ('d_helper_sha256','model_D'),
                       ('p_probe_sha256','probe_P'), ('d_probe_sha256','core'),
                       ('p_empty_sha256','fresh_proof'), ('observer_extension_sha256','observer'),
                       ('observer_bootstrap_sha256','bootstrap'), ('observer_base_sha256','observer_base')]:
        need(batch[field] == pins[key]['SHA256'], 'batch script differs: ' + field)
    need(pins['model_D']['SHA256'] == core['MODEL_HELPER'], 'D model helper differs from core')
    for outer, inner in [('receiver','sender'),('readiness','sender'),('client','core'),
                         ('client','diagnostic'),('client','probe_P'),('probe_P','core')]:
        text = Path(pins[outer]['path']).read_text()
        need(Path(pins[inner]['path']).stem in text, 'embedded import differs: ' + outer + '/' + inner)
    client_text = Path(pins['client']['path']).read_text()
    for key in ['core','diagnostic','probe_P']:
        need(pins[key]['SHA256'] in client_text, 'client embedded SHA differs: '+key)
    need(pins['core']['SHA256'] in Path(pins['probe_P']['path']).read_text(), 'probe embedded core SHA differs')
    client_literals = literals(pins['client']['path'])
    need(client_literals['PHELPER'] == pins['model_P']['SHA256'], 'client P helper differs')
    roles = {}
    for role in ['P','D']:
        row, pins['dispatch:' + role] = doc(recipe['roles'][role]['dispatch'])
        argv = row['argv']; lower = role.lower()
        need(option(argv,'--role') == role and option(argv,'--host') == HOSTS[role] and
             option(argv,'--prefill-host') == HOSTS['P'], 'role/host differs')
        root = remote_path(option(argv,'--root'))
        need(root == batch[lower+'_model_root'] and row['plan_SHA256'] == batch[lower+'_plan_sha256'],
             'actual role model plan differs')
        need(option(argv,'--env-sha256') == core['ENV'] and
             option(argv,'--manifest-sha256') == core['MANIFESTS'][HOSTS[role]], 'role env/manifest differs')
        need(option(argv,'--namespace') == source['namespace'], 'D-origin namespace differs')
        need(CONTROL+'/'+recipe['helpers']['model_'+role] in argv, 'model helper argv differs')
        # Parent receipt is audit identity only; live state/PID/lease is checked on device.
        _, pins['launch:'+role] = doc(recipe['roles'][role]['launch_receipt'])
        roles[role] = dict(host=HOSTS[role], root=root, plan_SHA256=row['plan_SHA256'],
                           namespace=source['namespace'],
                           parent_controller_status=remote_path(recipe['roles'][role]['parent_controller_status']))
    manifest, pins['observer_manifest'] = doc(recipe['observer_manifest'])
    actual = manifest['observer-config-manifest.json']
    need(actual['plan_SHA256'] == roles['P']['plan_SHA256'] and
         actual['derived_config']['SHA256'] == batch['observer_config_sha256'], 'actual observer config differs')
    argv = prepare['argv']
    need(argv[:5] == [PYTHON,'-B','-S',CONTROL+'/'+recipe['helpers']['client'],'prepare'], 'prepare client differs')
    need(option(argv,'--source') == source['source_path'] and
         option(argv,'--source-sha256') == source['source_SHA256'], 'prepare actual input differs')
    need(option(argv,'--normal-result-sha256') == recipe['normal_reference_sha256'], 'normal reference differs')
    for flag in ['--d-native-evidence','--d-manager-report','--normal-result']:
        remote_path(option(argv,flag))
        need(re.fullmatch('[0-9a-f]{64}',option(argv,flag+'-sha256')), 'full evidence SHA required')
    # Client's actual parser/prepare (including placeholders resolved from live identity)
    # still runs inside the original coordinator before any request-once claim.
    batch_remote = remote_path(CONTROL+'/'+Path(recipe['batch']).name)
    prepare_remote = remote_path(CONTROL+'/'+Path(recipe['prepare']).name)
    common = ['--batch-json',batch_remote,'--batch-sha256',pins['batch']['SHA256']]
    recv = [PYTHON,'-B','-S',CONTROL+'/'+recipe['helpers']['receiver'],
            '--client',CONTROL+'/'+recipe['helpers']['client'],'--client-sha256',pins['client']['SHA256'],
            '--prepare-argv-json',prepare_remote,'--prepare-argv-sha256',pins['prepare']['SHA256']] + common + ['--execute']
    prefix = batch['job_prefix']
    jobs = {'receiver':dict(host=HOSTS['receiver'], argv=recv, name='root-'+tag+'-coordinator',
             timeout=1140, verify=[dict(path=prefix+'-coordinator/status.json',
             json_fields=dict(state='BOTH_ACTUAL_CLIENT_PHASES_COMPLETE_CAPTURE_REVIEW_PENDING',model_requests=2))])}
    for role in ['P','D']:
        jobs[role] = dict(host=HOSTS[role], argv=[PYTHON,'-B','-S',CONTROL+'/'+recipe['helpers']['sender'],
                          '--role',role]+common+['--execute'],name='root-'+tag+'-peer-'+role,
                          timeout=1200,verify=[dict(path=prefix+'-peer-'+role+'/status.json',json_fields=dict(state='EXITED'))])
    return dict(schema='existing-c3-stage-freeze-v1', flow='capture', scope='OFFLINE_CONTROL_BINDING_ONLY',
                cluster='cluster3',candidate_commit=core['B1'],tag=tag,roles=roles,jobs=jobs,
                input=dict(tokens=source['R_tokens'],SHA256=source['R_input_ids_sha256'],
                           source_path=source['source_path'],source_SHA256=source['source_SHA256']),
                device_files={batch_remote:pins['batch'],prepare_remote:pins['prepare']},
                fresh_proof_max_age_seconds=75, max_model_requests=2,local_pins=pins,
                runtime_gates='unchanged original controller: live PID/start/READY/lease, probes, actual prepare, empty75, once, EOF, fourrank')
