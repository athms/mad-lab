import torch
import torch.nn.functional as F
from einops import rearrange


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)

def fftconv(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, 'b H -> b H 1')).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)
    
def fftconv_heads(k, ssm_kernel, D, q, v, head_dim=1, ssm_kernel_rev=None):
    seqlen = k.shape[-1]
    fft_size = 2 * seqlen
    kv = (rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
            * rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=head_dim))  # b d1 d2 h l
    kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
    ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
    if ssm_kernel_rev is not None:
        ssm_kernel_rev_f = torch.fft.rfft(ssm_kernel_rev, n=fft_size)  # h L+1
        ssm_kernel_f = ssm_kernel_f + ssm_kernel_rev_f.conj()
    y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm='forward')[..., :seqlen]  # b d1 d2 h l
    out = y + kv * D.unsqueeze(-1)  # b d1 d2 h l
    q = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=head_dim)
    if head_dim > 1:
        out = mul_sum(out, q)
        return rearrange(out, 'b d2 h l -> b (h d2) l').to(dtype=k.dtype)
    else:
        return rearrange(out * q, 'b 1 1 h l -> b h l').to(dtype=k.dtype)