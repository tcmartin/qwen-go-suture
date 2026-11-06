"""Small synchronous CPU-only reproducibility checks. No base download or GPU."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from go_suture import MokaGraft,SpatialRoundTrip
from run_native import verify_files
from reproduce_edits import check


def main():
    folder=Path(__file__).resolve().parent;verify_files(folder);torch.set_num_threads(2)
    state=load_file(str(folder/'bridge017.safetensors'))
    bridge=SpatialRoundTrip(MokaGraft(folder/'moka')).eval().requires_grad_(False)
    bridge.load_state_dict(state,strict=True)
    torch.manual_seed(78);x=torch.randn(2,6,2560);y=torch.randn_like(x)
    mask=torch.tensor([[0,1,1,1,1,1],[1,1,1,1,1,1]],dtype=torch.bool)
    checks=[]
    with torch.no_grad():
        for condition in ('normal','shuffled','zero'):
            results=[]
            for _ in range(2):
                bridge.begin(mask,condition);bridge.read_hook(None,(),x)
                out=bridge.write_hook(None,(y,),x)
                assert out.shape==x.shape and bool(torch.isfinite(out).all())
                memory=bridge.memory.clone();bridge.read_hook(None,(),x[:,:1])
                cached=bridge.write_hook(None,(y[:,:1],),x[:,:1])
                assert torch.equal(memory,bridge.memory)
                results.append((out.clone(),cached.clone()));bridge.clear()
            assert all(torch.equal(a,b) for a,b in zip(*results))
            checks.append(condition+' exact repeated prefill/cached-token replay')
        assert bridge.read_hook(None,(),x) is x and bridge.write_hook(None,(y,),x) is x
    print(json.dumps({'gpu_used':False,'full_model_loaded':False,'state_tensors':len(state),'checks':checks,'edit_reproduction':check(folder)},indent=2))
if __name__=='__main__':main()
