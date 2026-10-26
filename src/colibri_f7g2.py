#!/usr/bin/env python3
# Colibri F7-G2 — f7 numeriek op de ECHTE DeepSeek-V4.1-Flash-1017.
#  A gemm     : GPU-smoortest + single-token GEMV-benchmark (GB10).
#  D run64    : 64-token f7-decode, alle 40 lagen GPU-experts (experts fp8 op NVMe,
#               per-64-row-block fp8-scales, 64 experts à 8 MB = 512 MB/laag in VRAM),
#               CPU-streaming embed/engram/kv/dense via pread (f7-bewijde pad),
#               next(989)-anker, per-stap gouden + voorspelde token + gat, identiteit vs f7ref64.log.
import argparse, json, os, struct, time, math
import numpy as np
import torch
import torch.nn.functional as F

D = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017")
COLS = os.path.expanduser("~/colibri/f7g_cols")
LAYERS, QAN, DIM, EX, HD, QN, MAXPOS = 40, 384, 4096, 64, 64, 128, 163840
QK, OK, QNA, ONA = 512, 512, 2048, 1024
KVROW, DROW, HWN, HWE, HWEK, HREN, HREK = 640, 1024, 2048, 64, 64, 128, 128
SWA_CAP, CSA_CAP = 128, 8
UNIK = [2654435761, 2246822519, 3266489917, 668265263, 373580081, 1213593641,
        2839234811, 4049963631, 1976968321, 797425231, 2791277651, 334629061,
        1436433521, 381267241, 2925436811, 136144391]
MULT = [2**31, 2**31, 931928243, 2**31, 990928423, 2**31, 990928423, 2**31,
        2**31, 990928423, 2**31, 2**31, 990928423, 931928243, 990928423, 990928423]
FP8MAX = 448.0

def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def e8m0(x): return np.ldexp(np.ones_like(x), np.round(np.log2(np.maximum(x, 1e-30))).astype(np.int32))

class Cols:
    def __init__(self, d=COLS): self.d = d
    def _off(self, nm):
        p = os.path.join(self.d, nm + ".col"); m = json.load(open(p + ".meta.json"))
        fd = os.open(p, os.O_RDONLY); return fd, m["off"], m["bytes"], m["dtype"], m["shape"]
    def read(self, nm, r0, r1):
        fd, off, byt, dt, shp = self._off(nm); row = np.prod(shp[1:])
        a = np.frombuffer(os.pread(fd, (r1 - r0) * row, off + r0 * row), dtype=np.dtype(dt)).copy()
        os.close(fd); return a.reshape((r1 - r0,) + tuple(shp[1:]))
    def row(self, nm, r): return self.read(nm, r, r + 1)[0]

def _fp8q(t):
    t = np.ascontiguousarray(t).astype(np.float32)
    if t.ndim == 1: t = t[None]
    blk = t.reshape(t.shape[0], t.shape[1] // 64, 64)
    sc = np.maximum(np.abs(blk).max(2) / FP8MAX, 1e-8)
    q = np.clip(blk / sc[..., None], -FP8MAX, FP8MAX).astype(ml_dtypes.float8_e4m3fn)
    return q, sc.astype(np.float32)
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

def gpu1(w, x, dev="cuda"):
    W = torch.from_numpy(np.ascontiguousarray(w)); X = torch.from_numpy(np.ascontiguousarray(x))
    return (X.to(dev).half() @ W.to(dev).half().T).float().cpu().numpy()

def gemm_bench():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    free, tot = torch.cuda.mem_get_info() if dev == "cuda" else (0, 0)
    print("device %s  vrij %.1f/%.1f GB" % (dev, free / 1e9, tot / 1e9))
    for (m, n) in ((7168, 4096), (32768, 4096), (9216, 4096)):
        x = torch.randn(1, n, device=dev, dtype=torch.float16)
        W = torch.randn(m, n, device=dev, dtype=torch.float16)
        for _ in range(5): y = x @ W.T
        if dev == "cuda": torch.cuda.synchronize()
        t = time.time()
        for _ in range(30): y = x @ W.T
        if dev == "cuda": torch.cuda.synchronize()
        print("GEMV M=1 (%dx%d): %.3f ms" % (m, n, (time.time() - t) / 30 * 1000))

GEMM_OK = False  # wordt True na --stage gemm of run-start
def gemm(x, w, bias=None):
    """(T,K) fp32 @ (N,K)^T → (T,N) fp32, bias erbij. GPU als GEMM_OK, anders numpy."""
    if GEMM_OK and x.shape[0] <= 64:
        return gpu1(w, x) + (bias if bias is not None else 0)
    y = np.ascontiguousarray(x, np.float32) @ np.ascontiguousarray(w, np.float32).T
    return y + bias if bias is not None else y

# ── kolom-namen (f7g-kolommen, gebouwd door f7g2 build op node2, of handmatig) ──
def col_names():
    n = {}
    for li in range(LAYERS):
        p = f"h.{li}."
        n[p + "qn"] = "l%d_qn" % li; n[p + "qn_bias"] = "l%d_qnb" % li
        n[p + "qp"] = "l%d_qp" % li; n[p + "qp_bias"] = "l%d_qpb" % li
        n[p + "kn"] = "l%d_kn" % li; n[p + "vp"] = "l%d_vp" % li; n[p + "o"] = "l%d_o" % li
        n[p + "gate_dot"] = "l%d_gd" % li; n[p + "gate_q"] = "l%d_gq" % li
        n[p + "gate_k"] = "l%d_gk" % li; n[p + "gate_norm"] = "l%d_gn" % li
        n[p + "gate_norm_b"] = "l%d_gnb" % li; n[p + "gate_up"] = "l%d_gu" % li
        n[p + "gate_up_b"] = "l%d_gub" % li; n[p + "pe"] = "l%d_pe" % li
        n[p + "e"] = "l%d_e" % li; n[p + "e_scale"] = "l%d_es" % li
        n[p + "ed"] = "l%d_ed" % li; n[p + "ed_scale"] = "l%d_eds" % li
        n[p + "ev"] = "l%d_ev" % li
        n[p + "en"] = "l%d_en" % li; n[p + "en_b"] = "l%d_enb" % li
        n[p + "swa"] = "l%d_swa" % li if li % 4 == 0 else None
        n[p + "csa"] = "l%d_csa" % li if li % 4 == 1 else None
    n["tok"] = "tok"; n["head_norm_w"] = "hnw"; n["head_norm_b"] = "hnb"; n["head"] = "head"
    n["k1"] = "k1"; n["b1"] = "b1"; n["k2"] = "k2"; n["b2"] = "b2"
    n["mp"] = "mp"; n["mn"] = "mn"; n["mn_b"] = "mn_b"; n["mv"] = "mv"
    n["eh1"] = "eh1"; n["eb1"] = "eb1"; n["eh2"] = "eh2"; n["eb2"] = "eb2"
    return n

# ── GPU-expert-set per laag (fp8, 64×12 MB = 768 MB/laag) ──
class GExp:
    def __init__(self, cols, li, dev="cuda"):
        self.li = li; self.dev = dev
        e = cols.read("l%d_e" % li, 0, EX * 12288)
        es = cols.read("l%d_es" % li, 0, EX * 192)
        ed = cols.read("l%d_ed" % li, 0, EX * 4096 * 4096).reshape(EX * 4096, 4096)
        eds = cols.read("l%d_eds" % li, 0, EX * 4096 * 64)
        self.E = torch.from_numpy(e.reshape(EX, 12288, 4096)).to(dev)
        self.ES = torch.from_numpy(es.reshape(EX, 192, 1)).to(dev)
        self.ED = torch.from_numpy(ed.reshape(EX, 4096, 4096)).to(dev)
        self.EDS = torch.from_numpy(eds.reshape(EX, 4096, 64, 1)).to(dev)
    def __call__(self, ids, x):  # x (T,4096) cpu → (T,4096) cpu
        xg = torch.from_numpy(np.ascontiguousarray(x)).to(self.dev)
        out = torch.zeros((xg.shape[0], 4096), device=self.dev)
        for slot, e in enumerate(ids.tolist()):
            gu = (xg @ self.E[e].T).float()  # (T,12288) fp16
            gu *= self.ES[e].repeat_interleave(64, 1).squeeze(-1)  # per-block scale
            gu = gu.to(self.dev).half()
            g, u = gu[:, :4096], gu[:, 4096:8192]  # gate/up first 8192? no — see below
            # per model: gate=first 4096, up=second 4096, last 4096 unused (act_dim=8192 in
            # colibri f5 is 2×4096=8192 with 12288 raw = [gate|up|pad]; f5 uses 8192.)
            out += torch.sigmoid(g) * g * u
        return out.cpu().numpy()

def fp8gemm(x, q, sc):  # x (T,K) np fp32, q (N,K) fp8, sc (N,K/64)
    m = x.shape[0]
    out = np.zeros((m, q.shape[0]), np.float32)
    xd = (q.T.astype(np.float32) * np.repeat(sc, 64, axis=1)).T
    return xd @ x.T

def main():
    global GEMM_OK
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="gemm", choices=["gemm", "run64"])
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    if a.stage == "gemm":
        gemm_bench()
    else:
        GEMM_OK = torch.cuda.is_available()
        raise SystemExit("run64-stage vereist f7-kolommen (bouw via f7g build op node2) — volgt in fase D")

if __name__ == "__main__":
    main()
