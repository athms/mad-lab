import torch
import torch.nn.functional as F
from einops import rearrange


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)

def fftconv(u, h, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    h_f = torch.fft.rfft(h, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=h.dtype), n=fft_size)
    if len(u.shape) > 3: h_f = h_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * h_f, n=fft_size, norm='forward')[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)
    
def fftconv_heads(k, h, D, q, v, head_dim=1):
    seqlen = k.shape[-1]
    fft_size = 2 * seqlen
    kv = (rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
            * rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=head_dim))  # b d1 d2 h l
    kv_f = torch.fft.rfft(kv.to(dtype=h.dtype), n=fft_size) / fft_size
    h_f = torch.fft.rfft(h, n=fft_size)  
    y = torch.fft.irfft(kv_f * h_f, n=fft_size, norm='forward')[..., :seqlen]  # b d1 d2 h l
    out = y + kv * D.unsqueeze(-1)  # b d1 d2 h l
    q = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
    if head_dim > 1:
        out = mul_sum(out, q)
        return rearrange(out, 'b d2 h l -> b (h d2) l').to(dtype=k.dtype)
    else:
        return rearrange(out * q, 'b 1 1 h l -> b h l').to(dtype=k.dtype)
