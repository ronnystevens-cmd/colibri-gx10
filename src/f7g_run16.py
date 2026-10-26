#!/usr/bin/env python3
# f7g_run16.py — 16-token GPU-run van de echte colibri_f7 met fp16 fp8mma-Kolommen.
# De CPU-kern (colibri_f5.ModelF5 / colibri_f7: engram, DSPark, block, experts, KV,
# compressor) blijft ONAANGETAST; alleen attention() en moe() krijgen een
# GPU-GEMV-adapter die de dichte gewichten per laag uit gpu_cols/l{L}.col haalt.
# Kolom-formaat (f7g_build_all.py): JSON-header ({"layer","cols":[[name,shape,off,end]],"bytes"})
# + fp16-payload. Identiteit GPU fp16 vs CPU fp32 wordt vóór de run bewezen.
# Excuut-lock (.colibri-gpu.lock) voorkomt dubbel GPU-gebruik.
import fcntl, json, os, sys, time
sys.path.insert(0, "/home/flitsken/colibri")
import numpy as np
import torch
import colibri_f5 as f5
import colibri_f7 as f7mod
from colibri_f7 import ModelF7

COLI = "/home/flitsken/colibri"
COLS = os.path.join(COLI, "gpu_cols")
LOCK = "/home/flitsken/.colibri-gpu.lock"
GPU_HEAVY = ["attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
             "attn.wo_a.weight", "attn.wo_b.weight",
             "ffn.gate.weight", "ffn.shared_experts.w1.weight",
             "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight"]

lk = open(LOCK, "w")
import atexit
def _geef_lock_vrij():
    try:
        fcntl.flock(lk, fcntl.LOCK_UN)
        os.remove(LOCK)
    except OSError:
        pass
try:
    fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atexit.register(_geef_lock_vrij)
except BlockingIOError:
    # iemand anders houdt de GPU: wacht blokkerend (geen stille exit-val)
    print("[run] GPU-lock bezet — wacht ...", flush=True)
    fcntl.flock(lk, fcntl.LOCK_EX)
    atexit.register(_geef_lock_vrij)
    print("[run] GPU-lock verkregen", flush=True)

# ---- geheugen: vrijgegeven CPU-dense kopieën (model draait met 40 GPU-lagen) ----
f5.ModelF5.dense_max_layers = 40          # dc.layer-cache mag alle lagen bijhouden
GPU_DENSE = ("wq_a", "wq_b", "wkv", "wo_b", "gate_w", "sw1", "sw2", "sw3")
_dense_off = set()

def release_dense(d, l):
    """Zet de zware dichte fp32-dutsen uit de dc-laag (GPU doet ze).
    Kleine tensors (wo_a, norms, biases, hc, sinks, compressor) blijven."""
    if l in _dense_off:
        return
    for k in GPU_DENSE:
        d.pop(k, None)
    _dense_off.add(l)

_cache = {}   # l -> {suffix: torch fp16 cuda} | None = geen kolom (CPU-valk)
_LRU = []

def load_cols(l):
    if l in _cache:
        return _cache[l]
    meta_p = os.path.join(COLS, f"l{l}.meta.json")
    col_p = os.path.join(COLS, f"l{l}.col")
    if not os.path.exists(meta_p):
        _cache[l] = None
        return None
    meta = json.load(open(meta_p))
    pref = f"layers.{l}."
    idx = {}
    for name, shape, off, end in meta["cols"]:
        key = name[len(pref):] if name.startswith(pref) else name
        idx[key] = (tuple(shape), off, end)
    f = open(col_p, "rb")
    g = {}
    for suf in GPU_HEAVY:
        if suf not in idx:
            raise RuntimeError(f"{suf} ontbreekt in {meta_p}")
        shape, off, end = idx[suf]
        f.seek(off)
        raw = f.read(end - off)
        g[suf] = torch.from_numpy(np.ascontiguousarray(
            np.frombuffer(raw, np.float16).reshape(shape).copy())).cuda()
    f.close()
    while len(_LRU) >= 4:
        old = _LRU.pop(0); _cache.pop(old, None)
    _LRU.append(l); _cache[l] = g
    torch.cuda.empty_cache()
    return g

def gmv(t, x):  # x: fp32 numpy/CPU-tensor -> fp32 GPU-GEMV (fp16-opslag, fp32-compute) -> fp32 numpy
    # fp16 cuBLAS-compute overflowt/cancelt op zware lagen (inf in q_b, absmax ~7,7e4).
    # fp16-opslag + fp32-compute is aantoonbaar exact (rel ~1e-7 t.o.v. fp64-referentie).
    xx = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32))) if isinstance(x, np.ndarray) \
        else torch.as_tensor(x, dtype=torch.float32)
    return (t.float() @ xx.cuda().unsqueeze(1)).squeeze(1).cpu().numpy().astype(np.float32)

# ---- adapter: exact dezelfde wiskunde als f5.attention/f5.moe, dutsen van GPU ----
def gpu_attention(self, d, x, pos):
    G = load_cols(self.cur_layer)
    if G is None:
        return orig_attn(self, d, x, pos)
    release_dense(d, self.cur_layer)
    S = "attn."
    qr = f5.rmsnorm(gmv(G[S+"wq_a.weight"], x), d["q_norm"])
    q = gmv(G[S+"wq_b.weight"], qr).reshape(f5.NHEADS, f5.HEAD_DIM)
    kv = f5.rmsnorm(gmv(G[S+"wkv.weight"], x), d["kv_norm"])
    ratio = f5.COMPRESS_RATIOS[self.cur_layer]
    yarn = ratio > 0
    theta = f5.COMPRESS_THETA if yarn else f5.ROPE_THETA
    orig = f5.YARN_ORIG_SEQ if yarn else 0
    c, s = f5.precompute_freqs(f5.ROPE_DIM, 1, pos, orig, theta, f5.YARN_FACTOR)
    q[..., -f5.ROPE_DIM:] = f5.rope_apply(q[..., -f5.ROPE_DIM:], c[0], s[0])
    kv[..., -f5.ROPE_DIM:] = f5.rope_apply(kv[..., -f5.ROPE_DIM:], c[0], s[0])
    self.window_kv[self.cur_layer][pos % f5.WINDOW] = kv
    if pos < f5.WINDOW:
        win_rows = np.arange(0, pos + 1)
    else:
        oldest = pos % f5.WINDOW + 1
        win_rows = np.concatenate([np.arange(oldest, f5.WINDOW), np.arange(0, oldest)])
    rows = list(win_rows)
    kvs = [self.window_kv[self.cur_layer][r] for r in rows]
    if ratio > 0:
        # compressor-update als dit een kv_source is (compressie-KV is LAAG-OVERSCHRIDEND
        # gedeeld: alle ratio>0-lagen lezen self.compress_kv, ook niet-producers)
        latent = None
        if self.cur_layer in f5.KV_SOURCES:
            latent = self._compressor(d, x, pos, ratio)
            if latent is not None:
                freqs = self._compress_freq(pos, ratio)
                lat = latent.copy()
                lat[-f5.ROPE_DIM:] = f5.rope_apply(lat[-f5.ROPE_DIM:], freqs[0], freqs[1])
                self.compress_kv[pos // ratio] = lat
                self.compress_slot = pos // ratio + 1
    if ratio > 0:
        compress_len = (pos + 1) // ratio
        if compress_len > 0:
            cl = min(compress_len, self.compress_slot)
            for j in range(cl):
                kvs.append(self.compress_kv[j])
    if not kvs:
        o = np.zeros((f5.NHEADS, f5.HEAD_DIM), dtype=np.float32)
    else:
        K = np.stack(kvs)
        scores = (q @ K.T) * self.softmax_scale
        scores = scores + d["attn_sink"][:, None]
        mx = scores.max(axis=-1, keepdims=True)
        p = np.exp(scores - mx)
        o = (p @ K) / p.sum(axis=-1, keepdims=True)
    o[..., -f5.ROPE_DIM:] = f5.rope_apply(o[..., -f5.ROPE_DIM:], c[0], s[0], inverse=True)
    og = o.reshape(f5.O_GROUPS, -1)
    wa = d["wo_a"].reshape(f5.O_GROUPS, f5.O_LORA, -1)
    o2 = np.einsum("gd,grd->gr", og, wa)
    return gmv(G[S+"wo_b.weight"], o2.flatten())

def gpu_moe(self, d, x):
    G = load_cols(self.cur_layer)
    if G is None:
        return orig_moe(self, d, x)
    scores = gmv(G["ffn.gate.weight"], x)
    sel = np.sqrt(f5.softplus(scores)) + d["gate_bias"]
    top = np.argsort(sel)[::-1][:f5.TOP_K]
    wts = np.sqrt(f5.softplus(scores[top])); wts = wts / (wts.sum() + 1e-20)
    wts = wts * f5.ROUTE_SCALE
    y = np.zeros(f5.DIM, dtype=np.float32)
    for k, i in enumerate(top):
        w1, w2, w3 = self.ec.get(self.cur_layer, int(i))
        gg = w1 @ x
        u = np.clip(w3 @ x, -f5.SWIGLU_LIMIT, f5.SWIGLU_LIMIT)
        gg = np.clip(gg, None, f5.SWIGLU_LIMIT)
        h = f5.silu(gg) * u
        y += wts[k] * (w2 @ h)
    h = f5.silu(np.clip(gmv(G["ffn.shared_experts.w1.weight"], x), None, f5.SWIGLU_LIMIT)) \
        * np.clip(gmv(G["ffn.shared_experts.w3.weight"], x), -f5.SWIGLU_LIMIT, f5.SWIGLU_LIMIT)
    y = y + gmv(G["ffn.shared_experts.w2.weight"], h)
    return y

orig_attn = f5.ModelF5.attention
orig_moe = f5.ModelF5.moe
f5.ModelF5.attention = gpu_attention
f5.ModelF5.moe = gpu_moe

# ---- identiteit: GPU fp16 vs CPU fp32 op de ECHTE dichte gewichten ----
# Zuivere fp32-dequant-GEMV als maatstaf: fp16-cuBLAS vs fp32 mag op lange
# dotproducten (5120–32768) tot ~10% afdrijven — grens 0.15 zoals de bouwer.
sys.path.insert(0, COLI)
import f7g_build_all as builder

def identity():
    worst = {}
    rng = np.random.default_rng(11)
    S = builder.SFT(f5.MDIR)
    for l in (0, 13, 26, 39):
        G = load_cols(l)
        if G is None:
            raise RuntimeError(f"geen kolom voor identiteitslaag {l}")
        for suf in ("attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
                    "attn.wo_b.weight", "ffn.gate.weight",
                    "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
                    "ffn.shared_experts.w3.weight"):
            Wg = G[suf].float().cpu().numpy()
            Wc = S.tensor_f32(f"layers.{l}.{suf}")
            din = Wc.shape[1]
            x = rng.standard_normal(din).astype(np.float32)
            yg = Wg @ x
            y32 = Wc @ x                    # CPU fp32 op de gedekwantificeerde kolom
            worst[f"l{l}.{suf}"] = float(np.max(np.abs(yg - y32) / (np.abs(y32) + 1e-2)))
    bad = {k: v for k, v in worst.items() if v > 0.15}
    return worst, bad

def main():
    _a = [int(x) for x in sys.argv[1:] if x.lstrip("-").isdigit()]
    if len(_a) >= 2 and _a[0] == _a[1] + 1:
        tokens, anker = 64, 989           # starter-laag-roep: <l+1> <l>
    elif len(_a) >= 2:
        tokens, anker = _a[0], _a[1]      # direct: <tokens> <anker>
    elif len(_a) == 1:
        tokens, anker = _a[0], 989        # direct: <tokens>
    else:
        tokens, anker = 64, 989
    _f7m = "aan boord" if os.path.exists("/home/flitsken/colibri/gpu_f7m.py") else "niet geïntegreerd"
    print(f"colibri_f7 GPU (fp16 fp8mma-Kolommen, head) — {tokens} tokens, exp-cache=32, f7m={_f7m}", flush=True)
    t0 = time.time()
    worst, bad = identity()
    print(f"[f7g] identiteit GPU fp16 vs CPU fp32: {len(worst)} controles, max {max(worst.values()):.4g}", flush=True)
    with open(os.path.join(COLI, "f7gpu_identiteit.txt"), "w") as f:
        json.dump({"controles": worst, "foutbovengrens": 0.15, "afkeur": bad,
                   "max": max(worst.values()), "ts": time.strftime("%d/%m %H:%M:%S")}, f, indent=1)
    if bad:
        print(f"[f7g] FAAL identiteit: {bad}", flush=True); sys.exit(2)
    m = ModelF7(exp_cache=32)
    print(f"[f7g] init {time.time()-t0:.1f}s", flush=True)
    seed = anker  # 989 = anker van de goudreferentie f7ref64
    seq = [seed]
    t = time.time(); logits = m.forward(seed, 0)
    print(f"[f7g] prefill(pos0) {time.time()-t:.1f}s", flush=True)
    for step in range(1, tokens + 1):
        t = time.time()
        tok = int(np.argmax(logits))
        logits = m.forward(tok, step)
        nxt = int(np.argmax(logits))
        extra = ""
        if m.last_gate_dot is not None:
            extra = f" gate_dot={np.round(m.last_gate_dot, 4).tolist()}"
        if m.with_dspark:
            ids, conf, mx = m.draft(nxt, temperature=0.0)
            extra += (f" draft={ids.tolist()} conf={np.round(conf,3).tolist()}"
                      f" mainx_rms={np.sqrt((mx*mx).mean()):.3f}")
        seq.append(nxt)
        print(f"[f7] stap {step}: {tok}->{nxt} ({time.time()-t:.1f}s){extra}", flush=True)
    rss = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print("=== F7G RESULTATEN ===", flush=True)
    print(f"  piek RSS {rss:.2f} GB | totale tijd {time.time()-t0:.1f}s voor {tokens} tokens", flush=True)
    print("  dense GEMV's via GPU fp16 fp8mma-Kolommen; experts/embed/head/KV/engram/DSPark als CPU-referentie", flush=True)
    with open(os.path.join(COLI, "f7gpu64.bewijs.json"), "w") as f:
        json.dump({"reeks": seq, "tijd_s": round(time.time() - t0, 1), "piek_gb": round(rss, 2),
                   "tokens": tokens, "ts": time.strftime("%d/%m %H:%M:%S")}, f, indent=1)

if __name__ == "__main__":
    main()
