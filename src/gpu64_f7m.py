# KLAAR-VOOR-RUN — 64-token GPU-run op node2 (f7m mini, col_f7m)
# Doel: één forward-pass van 64 tokens over de mini-stack (lagen 0-7) met fp16-col.
# Dit is GEEN pool-stop: de run draait op node2 met GPU-peek terwijl de pool rust.
import json, os, time
import numpy as np
import torch

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

COL = os.path.expanduser("~/colibri/col_f7m.col")
META = json.load(open(os.path.expanduser("~/colibri/col_f7m.meta.json")))
DEV = "cuda:0"

def w(name):
    off = META["offsets"][name]; shp = META["shapes"][name]
    n = int(np.prod(shp))
    with open(COL, "rb") as f:
        f.seek(off * 2); buf = f.read(n * 2)
    return torch.frombuffer(bytearray(buf), dtype=torch.float16).reshape(shp).to(DEV)

log("col meta:", META["tensors"], "tensors", META["total_gb"], "GB")
# embedding-lookup: 64 tokens id's 1..64
emb = w("model.language_model.embed_tokens.weight")  # [V, H]
H = emb.shape[1]; log("H =", H, "V =", emb.shape[0])
ids = torch.arange(1, 65, device=DEV)
x = emb[ids].float()  # [64, H]
log("x:", x.shape, "norm", float(x.norm()))

import math
def rmsnorm(t, wgt, eps=1e-6):
    v = t.pow(2).mean(-1, keepdim=True)
    return (t * torch.rsqrt(v + eps)).to(torch.float16).float() * wgt.float()

def rope_apply(q, k, dim=64):
    # NEOX rope via complex: freqs 1/10000^(i/dim)
    inv = torch.tensor([10000.0 ** (-2.0 * i / dim) for i in range(dim // 2)], device=DEV)
    pos = torch.arange(q.shape[1], device=DEV).float()
    ang = pos[:, None] * inv[None, :]                    # [S, dim/2]
    def rot(t):
        tc = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2).contiguous())
        tc = tc * torch.view_as_complex(torch.stack([torch.cos(ang), torch.sin(ang)], -1).to(torch.complex64))
        return torch.view_as_real(tc).flatten(-2)
    return rot(q), rot(k)

def softmax(t, dim=-1): return torch.softmax(t, dim)

for L in range(8):
    p = f"model.language_model.layers.{L}."
    xn = rmsnorm(x, w(p + "input_layernorm.weight"))
    # attention
    xq = xn @ w(p + "self_attn.q_a_proj.weight").T
    xq = xq @ w(p + "self_attn.q_b_proj.weight").T        # [64, n_heads*128]
    xkv = xn @ w(p + "self_attn.kv_a_proj_with_mqa.weight").T
    xk_raw = xkv @ w(p + "self_attn.kv_b_proj.weight").T  # [64, n_kv*128]
    # rope
    xq4, xk4 = rope_apply(xq, xk_raw)
    xq4 = xq4 @ w(p + "self_attn.o_proj.weight").T
    x = x + xq4

    xn2 = rmsnorm(x, w(p + "post_attention_layernorm.weight"))
    # MoE
    gate_w = w(p + "mlp.gate.weight")                     # [E, H]
    logits = xn2 @ gate_w.T                               # [64, E]
    probs = softmax(logits)
    topk = torch.topk(probs, k=8, dim=-1)                 # 8 experts/token
    norm_top = topk.values / topk.values.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for e in range(gate_w.shape[0]):
        sel = (topk.indices == e)
        if not sel.any(): continue
        tok = sel.any(-1)
        xw = xn2[tok]
        a = w(p + f"mlp.experts.{e}.gate_proj.weight")
        b = w(p + f"mlp.experts.{e}.up_proj.weight")
        d = w(p + f"mlp.experts.{e}.down_proj.weight")
        inter = torch.nn.functional.silu(xw @ a.T) * (xw @ b.T)
        contrib = inter @ d.T
        weight = (norm_top[tok][:, sel[tok]] if norm_top.dim() > 1 else norm_top[tok]).sum(-1, keepdim=True)
        out[tok] += contrib * weight
    x = x + out
    log(f"laag {L} klaar — x-norm {float(x.norm()):.3f}")

h = x
hw = w("lm_head.weight")
logits = (h @ hw.T).float()
toks = logits.argmax(-1)
log("64 tokens door 8 lagen — argmax-token-reeks:", toks.tolist())
log("EINDE-GPU-RUN OK" if torch.isfinite(logits).all() else "NaN in logits")
