"""CPU-only MXFP8 field comparison of already completed SWA samples."""
import argparse,hashlib,json,os
from pathlib import Path
C=Path('/home/admin/linshuo.zjl/publication-precision-20261101/control')
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(pin):
 p=Path(pin['path']);need(p.is_file() and not p.is_symlink() and sha(p)==pin['SHA256'],'immutable input drift');return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--plan',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();plan=load(dict(path=str(a.plan),SHA256=a.sha256));need(os.getuid()==500 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sha(__file__)==plan['script']['SHA256'],'CPU actual source');base=load(plan['replay_plan']);proof=load(plan['replay_result']);need(proof['both_actual_outputs_and_scales_reproduced'] and proof['primitive_calls']==2,'actual causal calibration required');env=load(base['environment']);need(sha(Path(env['site'])/'rtp_llm/models_py/modules/dsv41/fp8/_v41_swa_triton.py')==plan['codec_SHA256'],'actual codec source drift');root=Path(plan['output_root']);need(root.parent==C and not root.exists(),'new result root');root.mkdir(mode=0o700)
 def save(n,v):
  p=root/(n+'.partial');p.write_text(json.dumps(v,indent=2,sort_keys=True));p.replace(root/n)
 try:
  import torch
  torch.set_num_threads(1);need(not torch.cuda.is_initialized(),'no CUDA');data={}
  for case,pin in base['cases'].items():
   need(sha(pin['path'])==pin['SHA256'],'actual tensor drift');data[case]=torch.load(pin['path'],map_location='cpu',weights_only=True)
  get=lambda case,n:data[case]['D-L25-T7169-'+n]
  raw={c:get(c,'SWA_raw').reshape(128,528) for c in data};decoded={c:t[:,:512].contiguous().view(torch.float8_e4m3fn).float().reshape(128,16,32)*torch.exp2(t[:,512:].float()-127).unsqueeze(-1) for c,t in raw.items()};x,y=raw['source'],raw['normal'];a,b=decoded['source'],decoded['normal'];need(torch.isfinite(a).all() and torch.isfinite(b).all(),'finite decoded rows')
  group_bytes=(x[:,:512]!=y[:,:512]).reshape(128,16,32).sum(-1);scale_diff=x[:,512:]!=y[:,512:];value_diff=(a!=b).sum(-1);group_diff=(group_bytes>0)|scale_diff;delta=x[:,512:].int()-y[:,512:].int();hist=lambda t:{str(int(k)):int((t==k).sum()) for k in torch.unique(t)}
  numeric_rows=(a!=b).any(-1).any(-1).nonzero().flatten();need(numeric_rows.numel()==1,'one numeric row');row=int(numeric_rows[0]);coords={};geometry={}
  for case in ['source','normal']:
   positions=torch.arange(7169-128+1,7169+1,dtype=torch.int64);table=get(case,'SWA_request_block_table').flatten().long();slots=get(case,'SWA_physical_slots').flatten().long();inverse=get(case,'SWA_logical_indices').flatten().long();need(positions.numel()==slots.numel()==128 and get(case,'SWA_valid').all(),'actual128 fullwindow');page_end=int(table[7169//512]);need(page_end>0 and (int(slots[-1])-1)%page_end==0,'terminal slot geometry');entries=(int(slots[-1])-1)//page_end;need(entries==128,'actual physical SWA entries');expected=table[positions//512]*entries+positions%entries;expected_inverse=positions//512*512+positions%entries;need(torch.equal(expected,slots) and torch.equal(expected_inverse,inverse),'all actual slots must match production512span/128ring formula');coords[case]=int(positions[row]);geometry[case]=dict(table_row_token_span=512,pool_entries_per_page=entries,window_size=128,all128_actual_slots_and_inverse_coordinates_match=True)
  need(coords['source']==coords['normal']==7045 and int(get('source','SWA_logical_indices').flatten()[row])==6661,'joined numeric row')
  report=dict(state='JOINED_CAPTURED_SWA_TO_WINDOW_POSITIONS',model_requests=0,primitive_calls=0,numeric_row_ordinal=row,inverse_table_coordinate=6661,window_formula_absolute_position=7045,geometry=geometry,mapping_scope='all128 captured slots and inverse coordinates match production fixed-window translation; historical producer value identity and transport still unproven',source_remote_destination_byte_equality_proven=False,whole_model_fixed=False)
  save('result-safe.json',report);save('status.json',dict(state='SUCCEEDED',model_requests=0,result=dict(path=str(root/'result-safe.json'),SHA256=sha(root/'result-safe.json'))));print(json.dumps(report))
 except BaseException as e:save('status.json',dict(state='FAILED',error=str(e),model_requests=0));raise
if __name__=='__main__':main()
