"""Next evidence-only profile: real prepared inputs before assigning a Block root.

P: query7168/all40 entries. D:7169/entries+prepared+attention inputs0..26,
with L25 attention/HC/FFN seams. Same runtime, no old operator replay.
"""
from pathlib import Path
import hashlib,re
HERE=Path(__file__).resolve().parent
B=HERE.parents[3]/'benchmarks/publication_precision_20261101'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def read(n):return (HERE/'device'/n).read_text()
def put(n,s):
 for p in [HERE/'device'/n,B/n]:
  if p.exists():assert p.read_text()==s,'immutable source '+str(p)
  else:p.write_text(s)
 return sha(s)
def main():
 old=read('history36_observer_v2.py');prefix=old[old.index('import functools'):old.index('def block_wrapper')]
 prefix=prefix.replace('TARGETS = (BLOCK, ATTN, FRONT, MOE, COMP, GLOBAL, HC, NORM)','TARGETS = (BLOCK, ATTN, MOE, HC, NORM)')
 block='''def block_wrapper(original,mode):
    @functools.wraps(original)
    def call(block,*args,**kwargs):
        if block.layer_id not in _capture._config['layers'] or mode!=_capture._config['role']:
            return original(block,*args,**kwargs)
        previous=getattr(_capture._context,'value',None)
        if previous is not None and previous.get('block') is block:
            return original(block,*args,**kwargs)
        if mode=='P':
            cp=getattr(block.attn,'_cp_ctx',None)
            if cp is None or cp.cp_size!=4 or not cp.kv_cache_sharded:return original(block,*args,**kwargs)
        hidden,ids,positions=domain(block,args,kwargs,mode)
        _capture._context.value=dict(role=mode,layer=block.layer_id,attn=block.attn,block=block,positions=positions)
        try:
            record('block_entry4HC',hidden,positions)
            if block.layer_id==0:
                need(ids is not None and ids.numel()==positions.numel(),'real token geometry required')
                record('actual_input_token',ids,positions)
            if mode=='D':
                import torch
                prepared=kwargs.get('_prepared_attn')
                record('prepared_attn_present',torch.full((positions.numel(),1),int(prepared is not None),device=positions.device,dtype=torch.int8),positions)
                if prepared is not None:
                    residual,x_pre,post,comb=prepared[:4]
                    # forward_decode_layer passes prepared[0] as x. Capture the
                    # other natural values too; residual equality is insufficient.
                    record('prepared_normalized_input5120',x_pre,positions)
                    coeff=torch.cat([rows(block.attn_hc.pre_mix_out,positions),rows(post,positions),rows(comb,positions)],dim=1)
                    need(coeff.shape[1]==24 and coeff.dtype==torch.float32,'actual HC pre/post/comb packing differs')
                    record('prepared_HC_coeff24',coeff,positions)
                if block.layer_id==25:bind_parameters(block.attn)
            return original(block,*args,**kwargs)
        finally:_capture._context.value=previous
    return call


def attention_wrapper(original,method):
    @functools.wraps(original)
    def call(attn,*args,**kwargs):
        context=getattr(_capture._context,'value',None)
        if context is None or context['role']!='D' or attn is not context['attn']:
            return original(attn,*args,**kwargs)
        positions=context['positions']
        if method=='_forward_decode_body':
            record('actual_attention_input5120',args[0] if args else kwargs['x'],positions)
            if attn.layer_id==25:
                import torch
                quant=kwargs.get('shared_input_quant')
                if quant is None and len(args)>2:quant=args[2]
                record('shared_quant_present',torch.full((positions.numel(),1),int(quant is not None),device=positions.device,dtype=torch.int8),positions)
                if quant is not None:
                    for i,value in enumerate(quant):record('actual_shared_input_quant_%d'%i,value,positions)
        result=original(attn,*args,**kwargs)
        if attn.layer_id==25 and method=='_forward_decode_body':record('attn_output5120',result,positions)
        return result
    return call


'''
 seam=old[old.index('def seam_context'):old.index('\ndef patch(name, module):')]
 seam=seam.replace("context['layer'] == 1","context['role']=='D' and context['layer']==25").replace("'L1.'","'L25.'").replace("'L1.FFN_norm.weight'","'L25.FFN_norm.weight'")
 patch='''def patch(name,module):
    need(pin(module.__file__)['SHA256']==_capture._config['hook_source_sha256'][name],'actual hook source drift '+name)
    _capture._source_identity[name]=pin(module.__file__)
    if name==BLOCK:
        patch_seam_block(module)
        for method,mode in [('forward','P'),('_forward_prefill_fast_fp8','P'),('forward_decode','D')]:
            setattr(module.Block,method,block_wrapper(getattr(module.Block,method),mode))
    elif name==HC:patch_seam_hc(module)
    elif name==NORM:patch_seam_norm(module)
    elif name==ATTN:
        module.AttentionV41FP8._forward_decode_body=attention_wrapper(module.AttentionV41FP8._forward_decode_body,'_forward_decode_body')
    elif name==MOE:
        original=module.MoE.forward
        @functools.wraps(original)
        def moe(unit,*args,**kwargs):
            context=seam_context();active=context is not None and unit is context['block'].ffn
            if active:record('FFN_input5120',args[0] if args else kwargs['x'],context['positions'])
            result=original(unit,*args,**kwargs)
            if active:record('FFN_output5120',result,context['positions'])
            return result
        module.MoE.forward=moe

'''
 install=old[old.index('def install(config):'):]
 start=install.index('    need(config');end=install.index("    need(set(config",start)
 install=install[:start]+'''    need(config['schema']=='history39-inner-graph-latches-v1' and config['role'] in ['P','D']
         and config['layers']==list(range(40 if config['role']=='P' else 27))
         and config['positions']==([7168] if config['role']=='P' else [7169])
         and config['artifact_kind']=='inner_python_overlay' and config['wheel_sha256'] is None
         and config['gpu_budget_bytes']==4194304,'explicit prepared-input trajectory profile required')
'''+install[end:]
 install=install.replace('history35_latches_v1.py','history38_latches_v1.py').replace('_history35_existing_graph_latches','_history39_existing_graph_latches')
 put('history39_observer_v1.py','"""Evidence-only prepared-input and L25 seam capture; unchanged model/graph math."""\n'+prefix+block+seam+patch+install)
 boot=read('history38_bootstrap_v1.py').replace('history38_bootstrap_v1','history39_bootstrap_v1').replace('history38_observer_v1','history39_observer_v1')
 put('history39_bootstrap_v1.py',boot)
 helper=read('c3_hc031_capture38_model_v2.py').replace('history38','history39').replace('history39_latches_v1','history38_latches_v1')
 helper=helper.replace("cfg['positions']==[7169,7200] and cfg['layers']==list(range(40))","cfg['positions']==([7168] if role=='P' else [7169]) and cfg['layers']==list(range(40 if role=='P' else 27))").replace("('h38-'+role)","('h39-'+role)")
 mh=put('c3_hc031_capture39_model_v1.py',helper)
 core=read('c3_hc031_capture38_core_v2.py');oldmh=re.search(r"^MODEL_HELPER='([^']+)'",core,re.M)[1]
 core=core.replace(oldmh,mh).replace('history38-inner-graph-latches-v1','history39-inner-graph-latches-v1').replace("cfg['positions']==[7169,7200]","cfg['positions']==([7168] if role=='P' else [7169])").replace("cfg['layers']==list(range(40))","cfg['layers']==list(range(40 if role=='P' else 27))")
 ch=put('c3_hc031_capture39_core_v1.py',core)
 diag=read('c3_hc031_capture38_client_v2.py').replace('c3_hc031_capture38_core_v2','c3_hc031_capture39_core_v1').replace(sha(read('c3_hc031_capture38_core_v2.py')),ch)
 dh=put('c3_hc031_capture39_client_v1.py',diag)
 for kind in ['receiver','sender']:
  s=read('c3_hc031_capture38_'+kind+'_v2.py').replace(sha(read('c3_hc031_capture38_core_v2.py')),ch).replace('c3_hc031_capture38_core_v2','c3_hc031_capture39_core_v1').replace('c3_hc031_capture38_client_v2','c3_hc031_capture39_client_v1').replace(sha(read('c3_hc031_capture38_client_v2.py')),dh).replace('CAPTURE38','CAPTURE39')
  if kind=='sender':
   start=s.index("   if a.role=='P':\n    assert all(item['record']");end=s.index('   exports=[]',start);fresh=s[start:end]
   # P has real eager latches after this first CP request; existing lazy handling below applies.
   freshproof=fresh[fresh.index("    if action=='RESET' and a.fresh_proof:"):fresh.index('    with urlopen',fresh.index("ack['fresh_empty_proof']"))]
   s=s[:start]+s[end:]
   marker='   data=json.dumps(ack,sort_keys=True).encode();'
   s=s.replace(marker,''.join(line[1:]+'\n' for line in freshproof.splitlines())+marker)
  put('c3_hc031_capture39_'+kind+'_v1.py',s)
 print('current39 controls bound')
if __name__=='__main__':main()
