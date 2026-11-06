"""Portable native-only Go suture, extracted from the evaluated research implementation.
Modified for public packaging: local imports and operational/training code removed.
No parser, labels, query coordinates or donor policy logits enter inference.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
DONOR = Path(__file__).resolve().parent / 'moka'


class MokaGraft(nn.Module):
    def __init__(self, folder=DONOR):
        super().__init__()
        m = json.loads((folder/'go-model.json').read_text())
        raw = (folder/'go-model.bin').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == m['sha256']
        assert len(raw) == m['weightsBytes'] and m['format']=='million-go-int8'
        self.architecture=m['architecture'];self.keys={}
        for i,(name,t) in enumerate(m['tensors'].items()):
            shape=t['shape']; n=int(np.prod(shape))
            if t['dtype']=='float32':
                x=np.frombuffer(raw,dtype='<f4',count=n,offset=t['dataOffset']).copy().reshape(shape)
            else:
                assert t['dtype']=='int8' and 'quantizationGroupSize' not in t
                q=np.frombuffer(raw,dtype=np.int8,count=n,offset=t['dataOffset']).reshape(shape)
                scale=np.frombuffer(raw,dtype='<f4',count=shape[0],offset=t['scaleOffset'])
                x=q.astype(np.float32)*scale.reshape([-1]+[1]*(len(shape)-1))
            key=f'tensor_{i}';self.keys[name]=key
            self.register_buffer(key,torch.from_numpy(x.copy()))
        self.requires_grad_(False)

    def tensor(self,name):return getattr(self,self.keys[name])

    def conv(self,x,name):
        weight=self.tensor(name+'.weight').permute(0,3,1,2).contiguous()
        return F.conv2d(x,weight,self.tensor(name+'.bias'),padding=weight.shape[-1]//2)

    def linear(self,x,name):return F.linear(x,self.tensor(name+'.weight'),self.tensor(name+'.bias'))

    def forward(self,features):
        assert features.ndim==4 and tuple(features.shape[1:])==(9,9,12)
        x=F.relu(self.conv(features.permute(0,3,1,2).contiguous(),'stem'))
        for i in range(self.architecture['residualBlockCount']):
            p=f'residual.{i}'
            h=F.relu(self.conv(x,p+'.reduce'));h=F.relu(self.conv(h,p+'.first'))
            if (i+1)%self.architecture['globalResidualBlockInterval']==0:
                g=torch.cat([h.mean((2,3)),h.amax((2,3))],dim=1)
                g=self.linear(F.relu(self.linear(g,p+'.global.hidden')),p+'.global.output')
                h=h+g[:,:,None,None]
            h=F.relu(self.conv(h,p+'.second'));x=F.relu(x+self.conv(h,p+'.expand'))
        def flatten(z):return z.permute(0,2,3,1).contiguous().flatten(1)
        p=self.linear(flatten(F.relu(self.conv(x,'policy.convolution'))),'policy.linear')
        v=self.linear(F.relu(self.linear(flatten(F.relu(self.conv(x,'value.convolution'))),'value.hidden')),'value.output').tanh()
        return p,v

def donor_hidden(donor, features):
    x=F.relu(donor.conv(features.permute(0,3,1,2).contiguous(),'stem'))
    for i in range(donor.architecture['residualBlockCount']):
        p=f'residual.{i}';h=F.relu(donor.conv(x,p+'.reduce'));h=F.relu(donor.conv(h,p+'.first'))
        if (i+1)%donor.architecture['globalResidualBlockInterval']==0:
            g=torch.cat([h.mean((2,3)),h.amax((2,3))],dim=1)
            h=h+donor.linear(F.relu(donor.linear(g,p+'.global.hidden')),p+'.global.output')[:,:,None,None]
        h=F.relu(donor.conv(h,p+'.second'));x=F.relu(x+donor.conv(h,p+'.expand'))
    return x.permute(0,2,3,1).reshape(-1,81,48).contiguous()

class GoBridge(nn.Module):
    def __init__(self,hidden=2560,width=64):
        super().__init__();self.width=width
        self.query=nn.Linear(hidden,width,bias=False)
        self.key=nn.Linear(66,width,bias=False);self.value=nn.Linear(66,width,bias=False)
        self.output=nn.Linear(width,hidden,bias=False);nn.init.zeros_(self.output.weight)
        p=torch.arange(81);pos=torch.cat([F.one_hot(p//9,9),F.one_hot(p%9,9)],-1).float()
        self.register_buffer('positions',pos)
        self.memory=None;self.spans=None;self.calls=0

    def attend(self,x,memory):
        x32=x.float();x32=F.layer_norm(x32,(x32.shape[-1],))
        mem=torch.cat([F.layer_norm(memory.float(),(48,)),self.positions.expand(memory.shape[0],-1,-1)],-1)
        q=self.query(x32);k=self.key(mem);v=self.value(mem)
        a=(q@k.transpose(-1,-2)/math.sqrt(self.width)).softmax(-1)
        return self.output(a@v)

    def hook(self,module,args,out):
        if self.memory is None:return out
        self.calls+=1;x=args[0]
        # Explicitly disable outer packed autocast for FP32 bridge parameters.
        with torch.autocast(device_type=x.device.type,enabled=False):
            if x.ndim==3:
                if self.spans is not None or x.shape[0]!=self.memory.shape[0]:raise ValueError('Native bridge batch/context mismatch')
                correction=self.attend(x,self.memory)
            elif x.ndim==2:
                if self.spans is None or len(self.spans)!=len(self.memory):raise ValueError('Packed bridge requires exact question spans')
                pieces=[];cursor=0
                for j,(start,end) in enumerate(self.spans):
                    if start!=cursor or end<=start:raise ValueError('Invalid packed ownership')
                    pieces.append(self.attend(x[start:end][None],self.memory[j:j+1])[0]);cursor=end
                if cursor!=len(x):raise ValueError('Unowned tokens')
                correction=torch.cat(pieces)
            else:raise ValueError('Unexpected Qwen MLP input shape')
        return (out+correction.to(out.dtype)).to(out.dtype)

    def clear(self):self.memory=None;self.spans=None

class HiddenToGo(nn.Module):
    def __init__(self, hidden=2560, width=64):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(81, width) / math.sqrt(width))
        self.key = nn.Linear(hidden, width, bias=False)
        self.value = nn.Linear(hidden, width, bias=False)
        self.planes = nn.Linear(width, 12)

    def forward(self, hidden, prompt_mask):
        if hidden.ndim != 3 or prompt_mask.shape != hidden.shape[:2]:
            raise ValueError('Expected native hidden states and exact prompt mask')
        if prompt_mask.dtype != torch.bool or not prompt_mask.any(-1).all():
            raise ValueError('Each request requires nonempty prompt ownership')
        x = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        k, v = self.key(x), self.value(x)
        scores = (self.queries[None] @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(~prompt_mask[:, None, :], -torch.inf)
        z = self.planes(scores.softmax(-1) @ v)
        # First eleven donor planes are binary; the final komi plane is signed.
        # Soft predictions remain differentiable through the frozen expert.
        z = torch.cat([z[..., :11].sigmoid(), z[..., 11:].tanh()], -1)
        return z.reshape(-1, 9, 9, 12)

class HiddenGoRoundTrip(nn.Module):
    def __init__(self, donor, hidden=2560, width=64):
        super().__init__()
        self.donor = donor.eval().requires_grad_(False)
        self.encoder = HiddenToGo(hidden, width)
        self.writer = GoBridge(hidden, width)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)
        self.clear()

    def begin(self, prompt_mask, condition='normal'):
        """Call once for every independent request/batch; excludes answer suffix."""
        self.clear()
        if prompt_mask.ndim != 2 or prompt_mask.dtype != torch.bool:
            raise ValueError('Expected boolean native prompt mask')
        if not prompt_mask.any(-1).all():
            raise ValueError('Empty prompt')
        self.prompt_mask = prompt_mask.clone()
        if condition not in ('normal', 'shuffled', 'zero'):
            raise ValueError('Unknown causal control')
        self.condition = condition
        self.enabled = True

    def clear(self):
        self.enabled = False
        self.prompt_mask = None
        self.memory = None
        self.predicted_features = None
        self.write_mask = None
        self.gate_logits = None
        self.gate_value = None
        self.writer.clear()

    def read_hook(self, module, args, output):
        if not self.enabled:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.ndim != 3:
            raise ValueError('Packed route requires separate ownership qualification')
        b, t, _ = h.shape
        if self.memory is None:
            if self.prompt_mask.shape != (b, t):
                raise ValueError('Prefill shape differs from declared prompt ownership')
            with torch.autocast(device_type=h.device.type, enabled=False):
                self.predicted_features = self.encoder(h, self.prompt_mask)
                # No no_grad: expert weights are frozen, but input gradients matter.
                self.memory = donor_hidden(self.donor, self.predicted_features)
            positions = torch.arange(t, device=h.device)[None].expand(b, -1)
            last = positions.masked_fill(~self.prompt_mask, -1).amax(-1)
            with torch.autocast(device_type=h.device.type, enabled=False):
                final = h[torch.arange(b, device=h.device), last].float()
                self.gate_logits = self.gate(F.layer_norm(final, (final.shape[-1],)))
                self.gate_value = self.gate_logits.sigmoid()
            if self.condition == 'shuffled':
                if b < 2:
                    raise ValueError('Shuffled control requires multiple requests')
                self.memory = self.memory.roll(1, 0)
            if self.condition == 'zero':
                self.memory = self.memory * 0
            # Full-prompt information must never be injected into earlier tokens.
            self.write_mask = positions >= last[:, None]
        else:
            if t != 1 or b != self.memory.shape[0]:
                raise ValueError('Only cached one-token decode may reuse expert memory')
            self.write_mask = torch.ones((b, 1), dtype=torch.bool, device=h.device)
        return output

    def write_hook(self, module, args, output):
        if not self.enabled:
            return output
        if self.memory is None:
            raise ValueError('Expert writer ran before the Qwen reader')
        x = args[0]
        if x.ndim != 3 or x.shape[:2] != self.write_mask.shape:
            raise ValueError('Writer batch/sequence ownership mismatch')
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = self.writer.attend(x, self.memory)
            delta = delta * self.write_mask[..., None] * self.gate_value[:, None, :]
        return output + delta.to(output.dtype)

    def install(self, layers, read_layer, write_layer):
        if not 0 <= read_layer < write_layer < len(layers):
            raise ValueError('Reader must precede writer within the same Qwen forward')
        return (layers[read_layer].register_forward_hook(self.read_hook),
                layers[write_layer].mlp.register_forward_hook(self.write_hook))

class AlignedHiddenToGo(HiddenToGo):
    """Same numerical forward as007; expose log attention for auxiliary training."""
    def forward(self, hidden, prompt_mask):
        if hidden.ndim != 3 or prompt_mask.shape != hidden.shape[:2] or prompt_mask.dtype != torch.bool or not prompt_mask.any(-1).all():
            raise ValueError('Exact nonempty prompt ownership required')
        x = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        k, v = self.key(x), self.value(x)
        scores = (self.queries[None] @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(~prompt_mask[:, None, :], -torch.inf)
        self.attention_log_probs = scores.log_softmax(-1)
        z = self.planes(scores.softmax(-1) @ v)
        z = torch.cat([z[..., :11].sigmoid(), z[..., 11:].tanh()], -1)
        return z.reshape(-1, 9, 9, 12)

class AlignedRoundTrip(HiddenGoRoundTrip):
    def __init__(self, donor, hidden=2560, width=64):
        super().__init__(donor, hidden, width)
        self.encoder = AlignedHiddenToGo(hidden, width)

class ReceiverRoundTrip(AlignedRoundTrip):
    receiver_stage = False

    def read_hook(self, module, args, output):
        if self.receiver_stage:
            # Preserve source tensor flags/linear dispatch, but construct no
            # sender graph. Only the downstream writer participates in training.
            with torch.no_grad():
                return super().read_hook(module, args, output)
        return super().read_hook(module, args, output)

class AddressedWriter(GoBridge):
    def clear(self):
        super().clear()
        self.attention_log_probs = None
        self.last_attention = None

    def attend(self, x, memory):
        # Preserve the exact qualified forward operations and state keys.
        x32=x.float();x32=F.layer_norm(x32,(x32.shape[-1],))
        mem=torch.cat([F.layer_norm(memory.float(),(48,)),self.positions.expand(memory.shape[0],-1,-1)],-1)
        q=self.query(x32);k=self.key(mem);v=self.value(mem)
        scores=q@k.transpose(-1,-2)/math.sqrt(self.width)
        a=scores.softmax(-1)
        self.attention_log_probs=scores.log_softmax(-1)
        self.last_attention=a.detach()
        return self.output(a@v)

class AddressedRoundTrip(ReceiverRoundTrip):
    def __init__(self, donor, hidden=2560, width=64):
        super().__init__(donor,hidden,width)
        self.writer=AddressedWriter(hidden,width)

class SpatialWriter(AddressedWriter):
    def __init__(self,hidden=2560,width=64):
        super().__init__(hidden,width)
        self.address=nn.Linear(hidden,19)
        nn.init.zeros_(self.address.weight);nn.init.zeros_(self.address.bias)
        with torch.no_grad():self.address.bias[-1]=-2.
        self.capture_callback=None

    def clear(self):
        super().clear()
        self.prompt_mask=None
        self.address_bias=None

    def spatial(self,hidden):
        # hidden is already layer-normalized FP32 actual Qwen MLP input.
        z=self.address(hidden)
        raw=(z[:,:9,None]+z[:,None,9:18]).reshape(-1,81)
        return raw*z[:,18:].sigmoid(),z[:,18],raw

    def attend(self,x,memory):
        x32=F.layer_norm(x.float(),(x.shape[-1],))
        mem=torch.cat([F.layer_norm(memory.float(),(48,)),self.positions.expand(memory.shape[0],-1,-1)],-1)
        q=self.query(x32);k=self.key(mem);v=self.value(mem)
        scores=q@k.transpose(-1,-2)/math.sqrt(self.width)
        if self.address_bias is None:
            assert self.prompt_mask is not None and self.prompt_mask.shape==x.shape[:2]
            last=torch.arange(x.shape[1],device=x.device)[None].expand_as(self.prompt_mask).masked_fill(~self.prompt_mask,-1).amax(-1)
            idx=torch.arange(x.shape[0],device=x.device)
            selected=x32[idx,last]
            self.address_bias,gate,_=self.spatial(selected)
            if self.capture_callback is not None:
                self.capture_callback(selected.detach().clone(),scores[idx,last].detach().clone())
        scores=scores+self.address_bias[:,None,:]
        a=scores.softmax(-1);self.last_attention=a.detach();self.attention_log_probs=scores.log_softmax(-1)
        return self.output(a@v)

class SpatialRoundTrip(AddressedRoundTrip):
    def __init__(self,donor,hidden=2560,width=64):
        super().__init__(donor,hidden,width);self.writer=SpatialWriter(hidden,width)
    def write_hook(self,module,args,output):
        if self.enabled and self.writer.address_bias is None:self.writer.prompt_mask=self.prompt_mask
        return super().write_hook(module,args,output)
