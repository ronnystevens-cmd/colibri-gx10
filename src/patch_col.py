# Repareer de 5 corrupte F32-tensors in col_f7m.col in-place.
import json, os
import numpy as np
MODEL = os.path.expanduser("~/models/ds-mini")
COLB = os.path.expanduser("~/colibri/col_f7m.col")
META = os.path.expanduser("~/colibri/col_f7m.meta.json")
keys = ["layers.0.hc_attn_fn", "layers.0.ffn.gate.bias", "layers.0.ffn.gate.bias_vl",
        "layers.0.attn.attn_sink", "layers.0.hc_attn_base"]
M = json.load(open(META))
o = M["offsets"]
import glob
fixed = {}
for shf in sorted(glob.glob(os.path.join(MODEL, "*.safetensors"))):
    with open(shf, "rb") as f:
        hl = int.from_bytes(f.read(8), "little"); hdr = json.loads(f.read(hl))
        for k in keys:
            if k in hdr:
                e = hdr[k]; f.seek(8+hl+e["data_offsets"][0])
                w = np.frombuffer(f.read(e["data_offsets"][1]-e["data_offsets"][0]), dtype=np.float32)
                out = w.astype(np.float16)
                assert np.isfinite(out).all(), k
                with open(COLB, "r+b") as cf:
                    cf.seek(o[k]); cf.write(out.tobytes())
                fixed[k] = (float(np.abs(out).max()), int(np.isinf(out).sum()))
print("fixed:", fixed)
# verifieer
with open(COLB, "rb") as f:
    for k in keys:
        n = int(np.prod(M["shapes"][k]))
        f.seek(o[k]); h = np.frombuffer(f.read(n*2), dtype=np.float16)
        print("VERIFY", k, "inf", int(np.isinf(h).sum()), "max", float(np.abs(h).max()))
# nieuwe globale som (alleen deze 5 + vorige som bijwerken = volledig herberekenen is 27min; doe steekproef)
import time
t0 = time.time()
tot = np.float64(0); n_t = 0
with open(COLB, "rb") as f:
    for k in sorted(o):
        nn = int(np.prod(M["shapes"][k]))
        f.seek(o[k]); h = np.frombuffer(f.read(nn*2), dtype=np.float16)
        tot += float(np.float64(h.astype(np.float64)).sum()); n_t += 1
print(f"TOTAL sum={float(tot):.4e} finite={np.isfinite(tot)} tensors={n_t} ({time.time()-t0:.0f}s)")
