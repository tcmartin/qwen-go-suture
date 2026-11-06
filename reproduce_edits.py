"""Reproduce the two retained receiver edits on CPU; never edits reference017.

These are experimental tradeoff variants. Neither is a promoted replacement.
"""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file


def edited(source,variant):
    if variant not in ('position_erase022','remove_mean023'):raise ValueError(variant)
    out={k:v.detach().cpu().clone() for k,v in source.items()}
    w=out['writer.value.weight']
    if w.shape!=(64,66) or w.dtype!=torch.float32:raise ValueError('Expected original FP32 reference017 receiver')
    p=w[:,48:]
    if variant=='position_erase022':p.mul_(0.)
    else:
        mean=torch.cat([p[:,:9].mean(1,keepdim=True).expand(-1,9),p[:,9:].mean(1,keepdim=True).expand(-1,9)],1)
        p.copy_(p-mean)
    return out


def check(folder):
    source=load_file(str(folder/'bridge017.safetensors'))
    results={}
    for name in ('position_erase022','remove_mean023'):
        result=edited(source,name)
        expected=load_file(str(folder/(name+'-patch.safetensors')))
        for key,value in expected.items():
            if not torch.equal(result[key].contiguous().view(torch.uint8),value.contiguous().view(torch.uint8)):
                raise AssertionError('Bitwise saved-patch mismatch: '+name+'/'+key)
        changed=sum(int((result[k]!=source[k]).sum()) for k in result)
        assert changed==1152
        assert all(torch.equal(result[k],source[k]) for k in result if k!='writer.value.weight')
        results[name]={'bitwise_saved_patch_match':True,'changed_values':changed}
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--check',action='store_true',help='Compare deterministic edits with exported original patch tensors on CPU')
    p.add_argument('--variant',choices=['position_erase022','remove_mean023'])
    p.add_argument('--output',type=Path,help='New full bridge safetensors path; refuses an existing file')
    args=p.parse_args()
    if args.check:print(json.dumps(check(args.model_dir),indent=2));return
    if not args.variant or args.output is None:p.error('--variant and --output required unless --check')
    if args.output.exists():raise FileExistsError(args.output)
    source=load_file(str(args.model_dir/'bridge017.safetensors'))
    save_file(edited(source,args.variant),str(args.output),metadata={'parent':'bridge017','variant':args.variant,'experimental_tradeoff':'true'})
    print(str(args.output))
if __name__=='__main__':main()
