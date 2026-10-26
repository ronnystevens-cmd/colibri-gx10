#!/usr/bin/env python3
"""
colibri_f5.py — DeepSeek-V4.1-Flash-1017 VOLLE 40-laag decode-loop (f5).
PRINCIPE: dense core compact in RAM (fp8/bf16 bytes + scales, ~9,4 GB), experts
per laag/token rechtstreeks van de NVMe (fp16 LRU-cache over 384 experts/laag).

Schaal: 40 backbone-lagen; attention met compress_ratio (KV-compressie):
  - laag 0,1: ratio 0 (window-only, base rope_theta 10000)
  - laag 2..19: ratio 2, laag 20..39: ratio 1 (YaRN rope, theta 160000, factor 16)
  - kv_source [2,8,14,20] produceren gecomprimeerde KV (Compressor); alle ratio>0
    lagen lezen de gedeelde compress-KV. indexer/topk = f7 (hier dense pragmatisch).
Hyper-connections (hc_mult 4, pre/post/comb + sinkhorn) per laag (uit PoC).
Engram (1,14) en DSPark/MTP (37-39) = gestubt (identiteit) in f5.

Gebruik (node2): /home/flitsken/colibri-venv/bin/python colibri_f5.py [--tokens N]
  --tokens N: decode-stappen na de seed-token (default 8)
  --cold     : forceer NVMe-herlezing van experts elke stap (geen expert-cache)
  --exp-cache N : expert-fp16-cache-capaciteit in aantal experts (default 192)
"""
import json, struct, os, sys, time, argparse, collections
import numpy as np

MDIR = "/home/flitsken/models/DeepSeek-V4.1-Flash-1017"

# ---- architectuur (uit inference_config.json / manifest) ----
DIM      = 5120
NHEADS   = 64
HEAD_DIM = 512
ROPE_DIM = 64
Q_LORA   = 1280
O_LORA   = 1024
O_GROUPS = 8
WINDOW   = 128
HC       = 4
HC_EPS   = 1e-6
N_ROUTED = 384
TOP_K    = 6
MOE_INT  = 2304
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
EPS      = 1e-20
VOCAB    = 129280
N_LAYERS = 40

ROPE_THETA      = 10000.0
COMPRESS_THETA  = 160000.0
YARN_ORIG_SEQ   = 65536
YARN_FACTOR     = 16.0
BETA_FAST       = 32
BETA_SLOW       = 1

COMPRESS_RATIOS = [0,0] + [2]*18 + [1]*20          # lagen 0..39
KV_SOURCES      = {2,8,14,20}
# index_source en candidate-blocks zijn f7 (Indexer niet geladen in f5)

FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], dtype=np.float32)

def e4m3_table():
    tab = np.zeros(256, dtype=np.float32)
    tab[0x7F] = np.nan
    tab[0xFF] = np.nan
    for b in range(256):
        if b in (0x7F, 0xFF):
            continue
        s = (b >> 7) & 1
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = m * (2.0 ** -6) / 8.0
        else:
            v = (1 + m / 8.0) * (2.0 ** (e - 7))
        tab[b] = -v if s else v
    return tab

E4M3 = e4m3_table()

def bf16_to_f32(b):
    u = b.view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)

def e8m0_to_f32(e):
    return np.exp2(e.astype(np.float64) - 127.0).astype(np.float32)


class SFT:
    """Pure-stdlib safetensors lezer (uit PoC). Tensor-bytes worden per naam
    van de file gelezen (dat IS de NVMe-streaming). Dispatcht op opslagdtype."""
    def __init__(self, mdir=MDIR):
        self.mdir = mdir
        self.idx = json.load(open(os.path.join(mdir, "model.safetensors.index.json")))
        self.wm = self.idx["weight_map"]
        self.hdrs = {}
        self.fh = {}
        self._bytes = 0
        self._reads = 0

    def _hdr(self, sh):
        if sh not in self.hdrs:
            with open(os.path.join(self.mdir, sh), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
                h["__data_start__"] = 8 + n
                self.hdrs[sh] = h
        return self.hdrs[sh]

    def raw(self, name):
        sh = self.wm[name]
        h = self._hdr(sh)
        m = h[name]
        dtype = m["dtype"]
        shape = list(m["shape"])
        off, end = m["data_offsets"]
        ln = end - off
        if sh not in self.fh:
            self.fh[sh] = open(os.path.join(self.mdir, sh), "rb")
        self.fh[sh].seek(h["__data_start__"] + off)
        buf = self.fh[sh].read(ln)
        self._bytes += len(buf)
        self._reads += 1
        return np.frombuffer(buf, dtype=np.uint8), dtype, shape

    def bf16(self, name):
        b, dt, shp = self.raw(name)
        return bf16_to_f32(b.view(np.uint16).reshape(shp))

    def f32(self, name):
        b, dt, shp = self.raw(name)
        return b.view(np.float32).reshape(shp).copy()

    def fp8(self, name):
        b, dt, shp = self.raw(name)
        w = E4M3[b.reshape(shp)].copy()            # [R,K] fp32
        sb, sdt, sshape = self.raw(name.replace("weight", "scale"))
        s = e8m0_to_f32(sb.reshape(sshape))        # [R/32,K/32]
        s = np.repeat(np.repeat(s, 32, axis=0), 32, axis=1)
        return w * s

    def fp4(self, name):
        b, dt, shp = self.raw(name)                # [R, K/2] bytes
        R, Khalf = shp
        K = Khalf * 2
        b2 = b.reshape(R, Khalf)
        low = b2 & 0x0F
        high = (b2 >> 4) & 0x0F
        vals = np.empty((R, Khalf, 2), dtype=np.float32)
        vals[..., 0] = np.take(FP4_TABLE, low)
        vals[..., 1] = np.take(FP4_TABLE, high)
        w = vals.reshape(R, K)
        sb, sdt, sshape = self.raw(name.replace("weight", "scale"))
        s = e8m0_to_f32(sb.reshape(sshape))        # [R, K/32]
        s = np.repeat(s, 32, axis=1)
        return w * s

    def dequant(self, name):
        """Algemene dispatch op opslagdtype: F32/BF16/F8_E4M3/I8(fp4)-packed."""
        b, dt, shp = self.raw(name)
        dt = dt.upper()
        if dt == "F32":
            return b.view(np.float32).reshape(shp).copy()
        if dt == "BF16":
            return bf16_to_f32(b.view(np.uint16).reshape(shp))
        if "F8" in dt or "E4M3" in dt or dt == "U8":
            w = E4M3[b.reshape(shp)].copy()
            sb, sdt, sshape = self.raw(name.replace("weight", "scale"))
            s = e8m0_to_f32(sb.reshape(sshape))
            r = shp[0] // sshape[0]; k = shp[1] // sshape[1]
            s = np.repeat(np.repeat(s, r, axis=0), k, axis=1)
            return w * s
        if dt == "I8":   # fp4 gepackt
            R, Khalf = shp
            b2 = b.reshape(R, Khalf)
            low = b2 & 0x0F; high = (b2 >> 4) & 0x0F
            vals = np.empty((R, Khalf, 2), dtype=np.float32)
            vals[..., 0] = np.take(FP4_TABLE, low)
            vals[..., 1] = np.take(FP4_TABLE, high)
            w = vals.reshape(R, Khalf * 2)
            sb, sdt, sshape = self.raw(name.replace("weight", "scale"))
            s = e8m0_to_f32(sb.reshape(sshape))
            s = np.repeat(s, 32, axis=1)
            return w * s
        raise ValueError(f"onbekend dtype {dt} voor {name}")


# ---------- hulpfuncties ----------
def rmsnorm(x, w, eps=EPS):
    return x * np.reciprocal(np.sqrt(np.mean(x*x) + eps)) * w

def precompute_freqs(dim, seqlen, start_pos, original_seq_len, base, factor):
    """(c, s): cos/sin per paar-dim voor posities [start_pos, start_pos+seqlen)."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * np.log(original_seq_len / (rotations * 2 * np.pi)) / (2 * np.log(base))
        low = max(int(np.floor(corrected_dim(BETA_FAST))), 0)
        high = min(int(np.ceil(corrected_dim(BETA_SLOW))), dim - 1)
        ramp = ((np.arange(dim // 2, dtype=np.float64) - low) / max(high - low, 1e-3)).clip(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    pos = np.arange(start_pos, start_pos + seqlen, dtype=np.float64)[:, None]
    ang = pos * freqs[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

def rope_apply(x, c, s, inverse=False):
    xr = x.reshape(*x.shape[:-1], ROPE_DIM // 2, 2)
    v0, v1 = xr[..., 0].copy(), xr[..., 1].copy()
    if not inverse:
        nr = v0 * c - v1 * s; ni = v0 * s + v1 * c
    else:
        nr = v0 * c + v1 * s; ni = -v0 * s + v1 * c
    return np.stack([nr, ni], axis=-1).reshape(*x.shape[:-1], ROPE_DIM)

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def hc_mixes(x, fn, scale, base):
    """x [HC*DIM]; fn [24, HC*DIM]; scale [3]; base [24] -> (pre, post, comb)."""
    xf = x.astype(np.float32)
    rsqrt = 1.0 / np.sqrt(np.mean(xf*xf) + EPS)
    mixes = (fn @ xf) * rsqrt
    pre = 1.0 / (1.0 + np.exp(-(mixes[0:HC] * scale[0] + base[0:HC]))) + HC_EPS
    post = 2.0 * (1.0 / (1.0 + np.exp(-(mixes[HC:2*HC] * scale[1] + base[HC:2*HC]))))
    comb = mixes[2*HC:] * scale[2] + base[2*HC:]
    comb = comb.reshape(HC, HC)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True)) + HC_EPS
    comb = comb / (comb.sum(axis=-1, keepdims=True) + HC_EPS)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    for _ in range(19):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    return pre, post, comb


class DenseCore:
    """Compacte dense weights per laag RESIDENT in RAM (~9,4 GB totaal).

    Compact = opgeslagen bytes + scales (uint8) zoals ze op de NVMe staan;
    eenmalig per laag gelezen. Per gebruik worden ze naar fp32 gedequantizeerd
    (transient) zodat het residente voetafdruk compact blijft. Een kleine
    fp32-LRU (max_layers, default 4) dempt herhaalde dequant van dezelfde laag.
    Dit geeft de '~9,4 GB dense core in RAM' claim waar EN houdt het ontwerp
    trouw: dense resident, enkel experts streamen van NVMe."""
    def __init__(self, sft: SFT, max_layers=4):
        self.sft = sft
        self.max_layers = max_layers
        self.compact = {}        # l -> {naam: (raw_bytes_uint8, dtype, shape)}
        self._fp32 = collections.OrderedDict()
        self.compact_bytes = 0

    def _has(self, name):
        return name not in ("comp_wgate",)  # alleen ratio>1; afhandeling in layer()

    def load_compact(self, l):
        """Lees de compacte (bytes+scale) tensoren van laag l eenmalig in RAM."""
        if l in self.compact:
            return self.compact[l]
        sft = self.sft
        L = f"layers.{l}."
        names = [f"{L}attn.wq_a.weight", f"{L}attn.q_norm.weight", f"{L}attn.wq_b.weight",
                 f"{L}attn.wkv.weight", f"{L}attn.kv_norm.weight", f"{L}attn.wo_a.weight",
                 f"{L}attn.wo_b.weight", f"{L}attn.attn_sink", f"{L}attn_norm.weight",
                 f"{L}ffn_norm.weight", f"{L}ffn.gate.weight", f"{L}ffn.gate.bias",
                 f"{L}ffn.shared_experts.w1.weight", f"{L}ffn.shared_experts.w2.weight",
                 f"{L}ffn.shared_experts.w3.weight", f"{L}hc_attn_fn", f"{L}hc_attn_base",
                 f"{L}hc_attn_scale", f"{L}hc_ffn_fn", f"{L}hc_ffn_base", f"{L}hc_ffn_scale"]
        if l in KV_SOURCES:
            names += [f"{L}attn.compressor.wkv.weight", f"{L}attn.compressor.norm.weight"]
            if COMPRESS_RATIOS[l] > 1:
                names += [f"{L}attn.compressor.wgate.weight"]
        cc = {}
        for n in names:
            sh = sft.wm[n]
            h = sft._hdr(sh)
            m = h[n]
            off, end = m["data_offsets"]
            with open(os.path.join(sft.mdir, sh), "rb") as f:
                f.seek(h["__data_start__"] + off)
                buf = f.read(end - off)
            cc[n] = (np.frombuffer(buf, dtype=np.uint8), m["dtype"], list(m["shape"]))
            self.compact_bytes += end - off
            # Gekwantiseerde tensoren (F8_E4M3 / I8-fp4) hebben een .scale-sibling
            # (block-schalen, E8M0). Alleen als die in de index bestaat — normen
            # (bf16/f32) en niet-kwantgewichten hebben geen scale-pad.
            sname = n[:-len("weight")] + "scale"
            if n.endswith(".weight") and sname in sft.wm:
                sh2 = sft.wm[sname]
                h2 = sft._hdr(sh2)
                m2 = h2[sname]
                off2, end2 = m2["data_offsets"]
                with open(os.path.join(sft.mdir, sh2), "rb") as f:
                    f.seek(h2["__data_start__"] + off2)
                    buf2 = f.read(end2 - off2)
                cc[sname] = (np.frombuffer(buf2, dtype=np.uint8), m2["dtype"], list(m2["shape"]))
                self.compact_bytes += end2 - off2
        self.compact[l] = cc
        return cc

    def _dequant1(self, raw):
        """Eén tensor uit compacte bytes -> fp32 (schaal meegenomen)."""
        buf, dtype, shp = raw
        dt = dtype.upper()
        if dt == "F32":
            return buf.view(np.float32).reshape(shp).copy()
        if dt == "BF16":
            u = buf.view(np.uint16).reshape(shp)
            return bf16_to_f32(u)
        if "F8" in dt or "E4M3" in dt:
            w = E4M3[buf.reshape(shp)].copy()
            sbuf = raw[0]  # placeholder; scales worden apart opgeslagen
            raise NotImplementedError("schaal-tensor handling hoort via separate schaal")
        raise NotImplementedError(dt)

    def _dequant_pair(self, cc, weight_name):
        """weight+scale paar uit compacte bytes -> fp32."""
        b, dt, shp = cc[weight_name]
        dt = dt.upper()
        if dt == "BF16":
            u = b.view(np.uint16).reshape(shp)
            return bf16_to_f32(u)
        if dt == "F32":
            return b.view(np.float32).reshape(shp).copy()
        # F8_E4M3 of I8 (fp4 packed): scale tensor lezen
        scale_name = weight_name.replace("weight", "scale")
        sb, sdt, sshp = cc[scale_name]
        if dt == "I8":   # fp4 gepackt
            R, Khalf = shp
            b2 = b.reshape(R, Khalf)
            low = b2 & 0x0F; high = (b2 >> 4) & 0x0F
            vals = np.empty((R, Khalf, 2), dtype=np.float32)
            vals[..., 0] = np.take(FP4_TABLE, low)
            vals[..., 1] = np.take(FP4_TABLE, high)
            w = vals.reshape(R, Khalf * 2)
            s = e8m0_to_f32(sb.reshape(sshp))
            s = np.repeat(s, 32, axis=1)
            return w * s
        # FP8 (E4M3)
        w = E4M3[b.reshape(shp)].copy()
        s = e8m0_to_f32(sb.reshape(sshp))
        r = shp[0] // sshp[0]; k = shp[1] // sshp[1]
        s = np.repeat(np.repeat(s, r, axis=0), k, axis=1)
        return w * s

    def layer(self, l):
        """fp32-dict voor laag l, gedequantizeerd uit resident compacte bytes
        (kleine LRU dempt herhaalde dequant)."""
        if l in self._fp32:
            self._fp32.move_to_end(l)
            return self._fp32[l]
        cc = self.load_compact(l)
        L = f"layers.{l}."
        def g(short):
            return self._dequant_pair(cc, L + short)
        d = {}
        d["wq_a"]  = g("attn.wq_a.weight")
        d["q_norm"]= g("attn.q_norm.weight")
        d["wq_b"]  = g("attn.wq_b.weight")
        d["wkv"]   = g("attn.wkv.weight")
        d["kv_norm"]= g("attn.kv_norm.weight")
        d["wo_a"]  = g("attn.wo_a.weight")
        d["wo_b"]  = g("attn.wo_b.weight")
        d["attn_sink"] = cc[L + "attn.attn_sink"][0].view(np.float32).reshape(cc[L+"attn.attn_sink"][2]).copy()
        d["attn_norm"] = g("attn_norm.weight")
        d["ffn_norm"]  = g("ffn_norm.weight")
        d["gate_w"]   = g("ffn.gate.weight")
        d["gate_bias"]= cc[L + "ffn.gate.bias"][0].view(np.float32).reshape(cc[L+"ffn.gate.bias"][2]).copy()
        d["sw1"] = g("ffn.shared_experts.w1.weight")
        d["sw2"] = g("ffn.shared_experts.w2.weight")
        d["sw3"] = g("ffn.shared_experts.w3.weight")
        d["hc_attn_fn"] = cc[L+"hc_attn_fn"][0].view(np.float32).reshape(cc[L+"hc_attn_fn"][2]).copy()
        d["hc_attn_base"] = cc[L+"hc_attn_base"][0].view(np.float32).reshape(cc[L+"hc_attn_base"][2]).copy()
        d["hc_attn_scale"] = cc[L+"hc_attn_scale"][0].view(np.float32).reshape(cc[L+"hc_attn_scale"][2]).copy()
        d["hc_ffn_fn"] = cc[L+"hc_ffn_fn"][0].view(np.float32).reshape(cc[L+"hc_ffn_fn"][2]).copy()
        d["hc_ffn_base"] = cc[L+"hc_ffn_base"][0].view(np.float32).reshape(cc[L+"hc_ffn_base"][2]).copy()
        d["hc_ffn_scale"] = cc[L+"hc_ffn_scale"][0].view(np.float32).reshape(cc[L+"hc_ffn_scale"][2]).copy()
        if l in KV_SOURCES:
            d["comp_wkv"] = g("attn.compressor.wkv.weight")
            d["comp_norm"]= g("attn.compressor.norm.weight")
            if COMPRESS_RATIOS[l] > 1:
                d["comp_wgate"] = g("attn.compressor.wgate.weight")
        self._fp32[l] = d
        if len(self._fp32) > self.max_layers:
            self._fp32.popitem(last=False)
        return d


class ExpertCache:
    """fp16-LRU over (laag, expert) -> (w1,w2,w3 fp16). Koud = NVMe herlezen."""
    def __init__(self, sft: SFT, capacity=192, cold=False):
        self.sft = sft
        self.capacity = capacity
        self.cold = cold
        self.store = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, l, i):
        key = (l, i)
        if not self.cold and key in self.store:
            self.store.move_to_end(key)
            self.hits += 1
            return self.store[key]
        L = f"layers.{l}.ffn.experts.{i}."
        w1 = self.sft.fp4(L + "w1.weight")
        w2 = self.sft.fp4(L + "w2.weight")
        w3 = self.sft.fp4(L + "w3.weight")
        self.misses += 1
        if not self.cold:
            self.store[key] = (w1.astype(np.float16), w2.astype(np.float16), w3.astype(np.float16))
            if len(self.store) > self.capacity:
                self.store.popitem(last=False)
        return w1, w2, w3


class ModelF5:
    def __init__(self, exp_cache=192, cold=False, dense_max_layers=40):
        self.sft = SFT(MDIR)
        self.dc = DenseCore(self.sft, max_layers=dense_max_layers)
        self.ec = ExpertCache(self.sft, capacity=exp_cache, cold=cold)
        self.window_kv = {l: np.zeros((WINDOW, HEAD_DIM), dtype=np.float32)
                          for l in range(N_LAYERS)}
        # gedeelde gecomprimeerde KV-cache (slots; ratio1 vult elke slot, ratio2 om de 2)
        self.compress_kv = np.zeros((8192, HEAD_DIM), dtype=np.float32)
        self.compress_slot = 0
        # compressor toestand per kv_source voor ratio>1 (groeps-buffers)
        self.kv_state = {l: np.zeros((COMPRESS_RATIOS[l], HEAD_DIM), dtype=np.float32)
                         for l in KV_SOURCES if COMPRESS_RATIOS[l] > 1}
        self.score_state = {l: np.full((COMPRESS_RATIOS[l], HEAD_DIM), -np.inf, dtype=np.float32)
                            for l in KV_SOURCES if COMPRESS_RATIOS[l] > 1}
        # embedding + head (bf16 in checkpoint; fp32 hier voor logits in fp32)
        t0 = time.time()
        self.embed = self.sft.bf16("embed.weight")
        self.head  = self.sft.bf16("head.weight")
        self.norm_w = self.sft.bf16("norm.weight")
        self.softmax_scale = HEAD_DIM ** -0.5
        print(f"[f5] embed/head/norm geladen: {time.time()-t0:.2f}s, "
              f"NVMe-bytes {self.sft._bytes/1e9:.2f} GB", flush=True)
        self.sft._bytes = 0

    # ---- attention per laag (1 decode-token) ----
    def attention(self, d, x, pos):
        qr = rmsnorm(d["wq_a"] @ x, d["q_norm"])                     # [1280]
        q = (d["wq_b"] @ qr).reshape(NHEADS, HEAD_DIM)               # [64,512]
        kv = rmsnorm(d["wkv"] @ x, d["kv_norm"])                     # [512]
        ratio = COMPRESS_RATIOS[self.cur_layer]
        yarn = ratio > 0
        theta = COMPRESS_THETA if yarn else ROPE_THETA
        orig = YARN_ORIG_SEQ if yarn else 0
        c, s = precompute_freqs(ROPE_DIM, 1, pos, orig, theta, YARN_FACTOR)
        q[..., -ROPE_DIM:] = rope_apply(q[..., -ROPE_DIM:], c[0], s[0])
        kv[..., -ROPE_DIM:] = rope_apply(kv[..., -ROPE_DIM:], c[0], s[0])
        self.window_kv[self.cur_layer][pos % WINDOW] = kv

        # window-posities (get_window_topk_idxs, decode-tak)
        if pos < WINDOW:
            win_rows = np.arange(0, pos + 1)
        else:
            oldest = pos % WINDOW + 1
            win_rows = np.concatenate([np.arange(oldest, WINDOW), np.arange(0, oldest)])
        rows = list(win_rows)
        kvs = [self.window_kv[self.cur_layer][r] for r in rows]

        if ratio > 0:
            # compressor-update als dit een kv_source is
            latent = None
            if self.cur_layer in KV_SOURCES:
                latent = self._compressor(d, x, pos, ratio)
                if latent is not None:
                    # latent staat voor het eerste token van de groep: positie j*ratio
                    freqs = self._compress_freq(pos, ratio)
                    lat = latent.copy()
                    lat[-ROPE_DIM:] = rope_apply(lat[-ROPE_DIM:], freqs[0], freqs[1])
                    self.compress_kv[pos // ratio] = lat
                    self.compress_slot = pos // ratio + 1
            compress_len = (pos + 1) // ratio
            if compress_len > 0:
                cl = min(compress_len, self.compress_slot)
                for j in range(cl):
                    kvs.append(self.compress_kv[j])

        if not kvs:
            o = np.zeros((NHEADS, HEAD_DIM), dtype=np.float32)
        else:
            K = np.stack(kvs)                                        # [n, 512]
            scores = (q @ K.T) * self.softmax_scale                  # [64, n]
            scores = scores + d["attn_sink"][:, None]                # per-head sink bias
            mx = scores.max(axis=-1, keepdims=True)
            p = np.exp(scores - mx)
            o = (p @ K) / p.sum(axis=-1, keepdims=True)              # [64,512]
        o[..., -ROPE_DIM:] = rope_apply(o[..., -ROPE_DIM:], c[0], s[0], inverse=True)
        og = o.reshape(O_GROUPS, -1)                                 # [8, 4096]
        wa = d["wo_a"].reshape(O_GROUPS, O_LORA, -1)                 # [8,1024,4096]
        o2 = np.einsum("gd,grd->gr", og, wa)                         # [8,1024]
        return d["wo_b"] @ o2.flatten()                              # [5120]

    def _compress_freq(self, pos, ratio):
        """freqs voor het groepeerst-token: positivie (pos+1-ratio)."""
        yarn = True
        p0 = pos + 1 - ratio
        return precompute_freqs(ROPE_DIM, 1, max(p0, 0), YARN_ORIG_SEQ, COMPRESS_THETA, YARN_FACTOR)

    def _compressor(self, d, x, pos, ratio):
        """Compressor.forward (decode, 1 token). ratio1: latent per step.
        ratio>1: pool wanneer de groep kompleet is ((pos+1)%ratio==0)."""
        if ratio == 1:
            return rmsnorm(d["comp_wkv"] @ x, d["comp_norm"], EPS)
        kv = d["comp_wkv"] @ x
        sc = d["comp_wgate"] @ x
        slot = pos % ratio
        self.kv_state[self.cur_layer][slot] = kv
        self.score_state[self.cur_layer][slot] = sc
        if (pos + 1) % ratio != 0:
            return None
        st = self.score_state[self.cur_layer]
        mx = st.max(axis=1, keepdims=True)
        w = np.exp(st - mx); w = w / w.sum(axis=1, keepdims=True)
        lat = (self.kv_state[self.cur_layer] * w).sum(axis=0)
        return rmsnorm(lat, d["comp_norm"], EPS)

    # ---- MoE per laag (top-6 + shared, experts van NVMe) ----
    def moe(self, d, x):
        scores = d["gate_w"] @ x                                     # [384] bf16wijten->f32
        sel = np.sqrt(softplus(scores)) + d["gate_bias"]
        top = np.argsort(sel)[::-1][:TOP_K]
        wts = np.sqrt(softplus(scores[top]))
        wts = wts / (wts.sum() + 1e-20)
        wts = wts * ROUTE_SCALE
        y = np.zeros(DIM, dtype=np.float32)
        for k, i in enumerate(top):
            w1, w2, w3 = self.ec.get(self.cur_layer, int(i))
            g = w1 @ x
            u = np.clip(w3 @ x, -SWIGLU_LIMIT, SWIGLU_LIMIT)
            g = np.clip(g, None, SWIGLU_LIMIT)
            h = silu(g) * u
            y += wts[k] * (w2 @ h)
        h = silu(np.clip(d["sw1"] @ x, None, SWIGLU_LIMIT)) * np.clip(d["sw3"] @ x, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        y = y + (d["sw2"] @ h)
        return y

    # ---- block ----
    def block(self, d, x, pre_mix, pos):
        residual = x
        ap, apo, ac = hc_mixes(x.flatten(), d["hc_attn_fn"], d["hc_attn_scale"], d["hc_attn_base"])
        xin = np.sum(pre_mix[:, None] * x, axis=0)
        h = rmsnorm(xin, d["attn_norm"])
        self.cur_layer = self.block_layer
        h = self.attention(d, h, pos)
        x = apo[:, None] * h[None, :] + (ac @ residual)
        residual = x
        fp, fpo, fc = hc_mixes(x.flatten(), d["hc_ffn_fn"], d["hc_ffn_scale"], d["hc_ffn_base"])
        xin = np.sum(ap[:, None] * x, axis=0)
        h = rmsnorm(xin, d["ffn_norm"])
        h = self.moe(d, h)
        x = fpo[:, None] * h[None, :] + (fc @ residual)
        return x, fp

    # ---- forward (1 token) ----
    def forward(self, token, pos):
        """Eén generatie-stap: embed(token) -> 40 lagen -> logits.

        De KV-caches (window per laag + gedeelde compress-KV + compressor-
        toestand) PERSISTEREN over stappen heen (dedicated decode). Elke stap
        hervat vanuit de nieuwe token-embedding met een verse one-hot pre_mix —
        precies zoals generate.py elke stap `model(x[:, i:i+1], i)` aanroept."""
        h = self.embed[token].astype(np.float32)                 # [5120]
        h = np.broadcast_to(h[None, :], (HC, DIM)).copy()        # [HC,DIM]
        pre_mix = np.zeros(HC, dtype=np.float32); pre_mix[0] = 1.0
        for l in range(N_LAYERS):
            self.block_layer = l
            d = self.dc.layer(l)
            h, pre_mix = self.block(d, h, pre_mix, pos)
        # einde: hc_pre + final norm + head
        h1 = np.sum(pre_mix[:, None] * h, axis=0)                # collapse
        logits = self.head @ rmsnorm(h1, self.norm_w)            # [VOCAB]
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--exp-cache", type=int, default=192)
    ap.add_argument("--seed", type=int, default=1776)
    ap.add_argument("--dense-max-layers", type=int, default=8, help="LRU-cap dense fp32-lagen")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"colibri_f5 — DeepSeek-V4.1-Flash-1017 volle {N_LAYERS}-laag decode, "
          f"{args.tokens} tokens, cold={args.cold}, exp-cache={args.exp_cache}", flush=True)
    t0 = time.time()
    m = ModelF5(exp_cache=args.exp_cache, cold=args.cold, dense_max_layers=args.dense_max_layers)
    t_load = time.time() - t0
    # compact-bytes boeking zit in dc.compact_bytes via dequant-copies; print ram
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB (linux: KB->? /1e6)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"[f5] model init: {t_load:.2f}s | embed/head/norm cached | "
          f"compact dense bytes (40 lagen, incl. embed/head)~{m.dc.compact_bytes/1e9:.2f} GB", flush=True)

    # seed-token: pos 0 (prefill 1 token)
    seed = int(rng.integers(0, VOCAB))
    t_pre = time.time()
    logits = m.forward(seed, 0)
    t_prefill = time.time() - t_pre
    print(f"[f5] prefill (1 token, alle 40 lagen): {t_prefill:.2f}s", flush=True)
    print(f"[f5] compact dense bytes resident: {m.dc.compact_bytes/1e9:.2f} GB "
          f"| fp32-LRU lagen na prefill: {len(m.dc._fp32)}", flush=True)

    # decode-lus
    sft_bytes0 = m.sft._bytes
    t_start = time.time()
    token = int(np.argmax(logits))
    for step in range(1, args.tokens + 1):
        logits = m.forward(token, step)
        token = int(np.argmax(logits))
    dt = time.time() - t_start
    tok_s = args.tokens / dt
    nvme = (m.sft._bytes - sft_bytes0) / 1e6
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"\n=== F5 RESULTATEN ===")
    print(f"decode {args.tokens} tokens (warm dense cache, {args.exp_cache} exp-LRU): "
          f"{dt:.2f}s -> {tok_s:.2f} tok/s")
    print(f"expert-cache: hits={m.ec.hits} misses={m.ec.misses} "
          f"(hit-rate {m.ec.hits/(m.ec.hits+m.ec.misses)*100:.1f}%)")
    print(f"expert-NVMe bytes in decode: {nvme:.1f} MB -> {nvme/args.tokens:.1f} MB/token")
    print(f"piek RSS: {rss_gb:.2f} GB  | dense fp32-lagen in LRU: {len(m.dc._fp32)}")
    print(f"total including prefill: {(args.tokens+1)/(t_prefill+dt):.2f} tok/s")

if __name__ == "__main__":
    main()
