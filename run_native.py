"""Standalone native-text inference for the published experimental reference017.

This is a portable extraction, not a new GPU-qualified runtime. See README.md.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_files(folder):
    manifest=json.loads((folder/'file_manifest.json').read_text())
    for name,expected in manifest.items():
        p=folder/name
        if digest(p)!=expected:raise ValueError('Artifact checksum mismatch: '+name)
    return manifest


def load_native(folder, base=None, device='cuda', bridge_path=None):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from go_suture import MokaGraft,SpatialRoundTrip
    verify_files(folder)
    cfg=json.loads((folder/'suture_config.json').read_text())
    # The research native path used this reference implementation, not the
    # separately optimized packed-decision kernels. Fail rather than swap it.
    for name in ('torch_recurrent_gated_delta_rule','torch_chunk_gated_delta_rule'):
        closure=inspect.getclosurevars(getattr(native,name)).nonlocals
        impl=closure.get('implementation')
        if closure.get('is_new_implementation') is not False or impl is None or impl.__module__!=native.__name__ or impl.__name__!=name:
            raise RuntimeError('Unqualified native recurrent kernel dispatch: '+name)
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2026110509)
    source=base or cfg['base_model']
    revision=cfg['base_revision'] if base is None else None
    tokenizer=AutoTokenizer.from_pretrained(source,revision=revision)
    owner=Qwen3_5ForConditionalGeneration.from_pretrained(source,revision=revision,dtype=torch.bfloat16,device_map=device,attn_implementation='sdpa').eval().requires_grad_(False)
    layers=owner.model.language_model.layers
    patch=load_file(str(folder/'native_base_patch.safetensors'),device='cpu')
    weight=layers[cfg['base_patch_layer']].mlp.down_proj.weight
    ix=patch['columns'].to(weight.device)
    if not torch.equal(weight[:,ix].detach().cpu(),patch['original']):
        raise ValueError('Base model columns differ from the recorded source')
    with torch.no_grad():weight[:,ix]=patch['edited'].to(weight)
    actual=hashlib.sha256(weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if actual!=cfg['patched_native_matrix_sha256']:raise ValueError('Patched native matrix mismatch')
    bridge=SpatialRoundTrip(MokaGraft(folder/'moka'),cfg['hidden_size'],cfg['width']).to(device).float().eval().requires_grad_(False)
    bridge.load_state_dict(load_file(str(bridge_path or folder/'bridge017.safetensors'),device=device),strict=True)
    handles=bridge.install(layers,cfg['reader_layer'],cfg['writer_layer'])
    return owner,tokenizer,bridge,handles


def generate(owner,tokenizer,bridge,prompts,max_new_tokens=24):
    import torch
    if not 1<=len(prompts)<=4:raise ValueError('Use one to four prompts; original cached generation used batches of four')
    tokens=[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for p in prompts]
    if any(len(p)>1024 for p in tokens):raise ValueError('Prompt exceeds the study token bound')
    width=max(map(len,tokens));device=next(owner.parameters()).device
    ids=torch.tensor([[tokenizer.pad_token_id]*(width-len(p))+p for p in tokens],device=device)
    mask=ids.ne(tokenizer.pad_token_id).long()
    bridge.begin(mask.bool(),'normal')
    try:
        with torch.no_grad():
            result=owner.generate(input_ids=ids,attention_mask=mask,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(row,skip_special_tokens=True).strip() for row in result[:,width:]]
    finally:bridge.clear()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--prompt-file',type=Path,action='append',help='One UTF-8 ordinary board-text prompt per file; repeat up to four times')
    p.add_argument('--base',help='Optional local copy of the pinned Qwen model')
    p.add_argument('--device',default='cuda')
    p.add_argument('--bridge',type=Path,help='Optional experimental variant from reproduce_edits.py; reference017 is default')
    p.add_argument('--verify-only',action='store_true',help='CPU-only package SHA-256 verification; does not load Qwen')
    args=p.parse_args()
    if args.verify_only:
        print(json.dumps({'verified_files':len(verify_files(args.model_dir)),'gpu_used':False}));return
    owner,tok,bridge,handles=load_native(args.model_dir,args.base,args.device,args.bridge)
    try:
        prompts=[f.read_text().strip() for f in (args.prompt_file or [args.model_dir/'example.txt'])]
        print(json.dumps({'outputs':generate(owner,tok,bridge,prompts)},indent=2))
    finally:
        for h in handles:h.remove()

if __name__=='__main__':main()
