# adapted from:
# https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/src/model.py

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load


def time_mixer_rwkv6_wrapped_bf16(
    dim: int = 128,
    max_length: int = 1_280,
    head_dim: int = 16,
    dim_att: int = 128,
    head_dim_divisor: int = 8,
    n_layer: int = 1,
    layer_id: int = 0,
    use_jit: bool = False,
    *args, **kwargs
):
    """
    Wrapper to ensure that cuda kernel is only loaded when 
    we actually create an instance of the Time Mixer.
    """

    if not use_jit:
        def __nop(ob):
            return ob
        MyModule = nn.Module
        MyFunction = __nop
    else:
        MyModule = torch.jit.ScriptModule
        MyFunction = torch.jit.script_method

    if 'TUNE_ORIG_WORKING_DIR' in os.environ:
        base_path = os.getenv("TUNE_ORIG_WORKING_DIR")
    else:
        base_path = ''

    wkv6_cuda = load(
        name="wkv6",
        sources=[
            os.path.join(
                base_path,
                "mad",
                "model",
                "layers",
                "rwkv",
                "cuda",
                "wkv6_op.cpp"
            ),
            os.path.join(
                base_path,
                "mad",
                "model",
                "layers",
                "rwkv",
                "cuda",
                "wkv6_cuda.cu"
            ),
        ],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={head_dim}",
            f"-D_T_={int(max_length)}"
        ]
    )
        
    class WKV_6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.bfloat16
                assert k.dtype == torch.bfloat16
                assert v.dtype == torch.bfloat16
                assert w.dtype == torch.bfloat16
                assert u.dtype == torch.bfloat16
                assert head_dim == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ew = (-torch.exp(w.float())).contiguous()
                ctx.save_for_backward(r, k, v, ew, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.bfloat16
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
        return WKV_6.apply(B, T, C, H, r, k, v, w, u)

    class TimeMixer_RWKV6(MyModule):
        def __init__(self,
            dim: int = 128,
            head_dim: int = 16,
            dim_att: int = 128,
            head_dim_divisor: int = 8,
            n_layer: int = 1,
            layer_id: int = 0,
            # dtype=torch.bfloat16,
            *args, **kwargs
        ):
            super().__init__()
            self.layer_id = layer_id

            self.head_dim = head_dim
            self.n_head = dim_att // self.head_dim
            assert dim_att % self.n_head == 0

            with torch.no_grad():
                ratio_0_to_1 =  0 if n_layer<2 else layer_id / (n_layer - 1) #layer_id / (n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, dim)
                for i in range(dim):
                    ddd[0, 0, i] = i / dim

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
                self.time_maa_w1 = nn.Parameter(torch.zeros(dim, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
                self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, dim).uniform_(-1e-4, 1e-4))

                # fancy time_decay
                decay_speed = torch.ones(dim_att)
                for n in range(dim_att):
                    decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1,1,dim_att))

                TIME_DECAY_EXTRA_DIM = 64
                self.time_decay_w1 = nn.Parameter(torch.zeros(dim, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
                self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, dim_att).uniform_(-1e-4, 1e-4))

                tmp = torch.zeros(dim_att)
                for n in range(dim_att):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_dim))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(dim, dim_att, bias=False)
            self.key = nn.Linear(dim, dim_att, bias=False)

            self.value = nn.Linear(dim, dim_att, bias=False)
            self.output = nn.Linear(dim_att, dim, bias=False)
            self.gate = nn.Linear(dim, dim_att, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, dim_att, eps=(1e-5)*(head_dim_divisor**2))

        @MyFunction
        def jit_func(self, x):
            B, T, C = x.size()

            xx = self.time_shift(x) - x

            xxx = x + xx * self.time_maa_x
            xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
            xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
            mw, mk, mv, mr, mg = xxx.unbind(dim=0)

            xw = x + xx * (self.time_maa_w + mw)
            xk = x + xx * (self.time_maa_k + mk)
            xv = x + xx * (self.time_maa_v + mv)
            xr = x + xx * (self.time_maa_r + mr)
            xg = x + xx * (self.time_maa_g + mg)

            r = self.receptance(xr)
            k = self.key(xk)
            v = self.value(xv)
            g = F.silu(self.gate(xg))

            ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
            w = self.time_decay + ww

            return r, k, v, g, w

        @MyFunction
        def jit_func_2(self, x, g):
            B, T, C = x.size()
            x = x.view(B * T, C)
            
            x = self.ln_x(x).view(B, T, C)
            x = self.output(x * g)
            return x

        def forward(self, x):
            x = x.to(torch.bfloat16)
            B, T, C = x.size()
            H = self.n_head

            r, k, v, g, w = self.jit_func(x)
            x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

            return self.jit_func_2(x, g)

    return TimeMixer_RWKV6(
        dim=dim,
        max_length=max_length,
        head_dim=head_dim,
        dim_att=dim_att,
        head_dim_divisor=head_dim_divisor,
        n_layer=n_layer,
        layer_id=layer_id,
        *args, **kwargs
    ).to(torch.bfloat16)


def channel_mixer_rwkv6_wrapped(
    dim: int = 128,
    dim_inner: int = 512,
    n_layer: int = 1,
    layer_id: int = 0,
    use_jit: bool = False,
    *args, **kwargs
):
    """
    Wrapper to ensure that cuda kernel is only loaded when
    we actually create an instance of the Channel Mixer.
    """

    if not use_jit:
        def __nop(ob):
            return ob
        MyModule = nn.Module
        MyFunction = __nop
    else:
        MyModule = torch.jit.ScriptModule
        MyFunction = torch.jit.script_method

    class ChannelMixer_RWKV6(MyModule):
        def __init__(self,
            dim: int = 128,
            dim_inner: int = 512,
            n_layer: int = 1,
            layer_id: int = 0,
            *args, **kwargs
        ):
            super().__init__()
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, dim)
                for i in range(dim):
                    ddd[0, 0, i] = i / dim
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            self.key = nn.Linear(dim, dim_inner, bias=False)
            self.receptance = nn.Linear(dim, dim, bias=False)
            self.value = nn.Linear(dim_inner, dim, bias=False)

        @MyFunction
        def forward(self, x):
            xx = self.time_shift(x) - x
            xk = x + xx * self.time_maa_k
            xr = x + xx * self.time_maa_r

            k = self.key(xk)
            k = torch.relu(k) ** 2
            kv = self.value(k)
            return torch.sigmoid(self.receptance(xr)) * kv

    return ChannelMixer_RWKV6(
        dim=dim,
        dim_inner=dim_inner,
        n_layer=n_layer,
        layer_id=layer_id,
        *args, **kwargs
    )


if __name__ == '__main__':
    x = torch.randn(2, 128, 128).cuda().to(torch.bfloat16)
    cmixer = channel_mixer_rwkv6_wrapped().cuda().to(torch.bfloat16)
    tmixer = time_mixer_rwkv6_wrapped_bf16().cuda().to(torch.bfloat16)
    y1 = tmixer(x)
    y2 = cmixer(y1)
    