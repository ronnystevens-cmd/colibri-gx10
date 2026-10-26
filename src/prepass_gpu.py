# GPU-prepass: col fp16 (24.4GB) -> stream chunks naar VRAM, per chunk fp64-sommen.
# Output: per-tensor sums in meta, plus P-vector (1024 fp64 via atomics? NEE: per tensor fp64, dan reduce hostzijde).
import json, os, time
import cupy as cp
import numpy as np
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
M = json.load(open(os.path.expanduser("~/colibri/col_f7m.meta.json")))
o = M["offsets"]; keys = sorted(o)
free, total = cp.runtime.memGetInfo()
log(f"VRAM vrij {free/1e9:.1f}GB van {total/1e9:.1f}GB")
# stream de tensors in volgorde; per tensor fp64 som + fp64 |w|max + fp64 som(w^2)
res = {}
CH = 2_000_000_000  # elems per chunk (fp16=4GB, fp64-acc via chunks)
with open(os.path.expanduser("~/colibri/col_f7m.col"), "rb") as f:
    for i, k in enumerate(keys):
        n = int(np.prod(M["shapes"][k]))
        sz = n * 2
        f.seek(o[k]); raw = f.read(sz)
        h = np.frombuffer(raw, dtype=np.float16)
        g = cp.asarray(h)
        s = float(cp.sum(g.astype(cp.float64)).get())
        sq = float(cp.sum((g.astype(cp.float64))**2).get())
        mx = float(cp.max(cp.abs(g)).get())
        res[k] = dict(sum=s, sumsq=sq, max=mx, n=n)
        if (i + 1) % 200 == 0 or i + 1 == len(keys):
            log(f"{i+1}/{len(keys)}")
json.dump(res, open(os.path.expanduser("~/colibri/col_f7m_sums.json"), "w"))
gt = max(res.items(), key=lambda kv: abs(kv[1]["sum"]))
log(f"KLAAR grootste|som| {gt[0]} {gt[1]['sum']:.1f}; total_sum={sum(v['sum'] for v in res.values()):.1f}")
