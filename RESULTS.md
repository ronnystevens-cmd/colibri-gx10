# Results — measured, reproducible

All timings are wall-clock on an ASUS DGX Spark (GB10, 121 GB unified, one node).
Every run is deterministic and can be re-verified from `results/`.

## Summary table

| Phase | What | Status | Key numbers | Evidence |
|---|---|---|---|---|
| **F4** | Single-layer PoC (layer 0) | ✅ done | decode ~1.9 tok/s (expert cache), dense 0.55 s | `colibri_f5.py` |
| **F5** | Full 40-layer decode loop | ✅ done, EXIT=0 | dense core 6.84 GB resident, ~0.02 tok/s CPU, 4.5 GB NVMe/token | `f5` logs |
| **F7** | Engram + DSPark/MTP correctness | ✅ numerically proven | engram hash **exact**, gate Δ = 0.0, NVMe engram live | `f7ref64.goud.log` |
| **F7G** | GPU acceleration (GB10) | ⚠️ weight-identity ✅, chain-identity ❌ | GEMV 0.252/1.485 ms → ~617 tok/min projection; 32 weight-controls max 2.457e-06; 64-token run 3135 s — but token chain diverges from CPU | `f7gpu64.log`, `f7_cpu_seed989.log` |

## Key measurements

### Dense core (all 40 layers, resident)
- **6.84 GB** in RAM (fp4/fp8 bytes + scales)
- **~9.4 GB** with embeddings, head, norms
- Loaded in **~11 s** (incl. index parsing)

### NVMe expert streaming
- **~4.5 GB / token** — 40 layers × 6 active experts × ~17.7 MB fp4 each
- Random-access `pread` (experts scattered across 48 shards, 476 GiB total)
- NVMe is **not** the bottleneck at 0.02 tok/s — CPU dequant/matmul is

### Engram (the "brain" layers 1 & 14)
- Rolling-hash n-gram indexer, hash IDs **exactly equal** to reference `NgramHashState`
- Gate math (fp64 golden): **Δ = 0.00e+00**
- Real NVMe hash-table lookup live (gate_dot: `[0,0,0,0]` → trained values after fix)

### GPU phase (GB10, CUDA-torch 2.9.0+cu130)
- fp16 GEMV calibration: experts **0.252 ms**, engram **1.485 ms**
- → projection **~617 tok/min** (theoretical, compute-only, excludes NVMe floor)
- 16-token GPU run: **32 dense-column weight-controls max 2.457e-06**, peak RSS **17.81 GB**
- 64-token GPU run completed: 3135 s total, peak RSS 19.81 GB, ~49 s/step

### ⚠ GPU vs CPU chain divergence (verified cross-check)
Same seed (989), same model, same input:
- **CPU core (fp32)**: 989→989→989→… (stable attractor loop)
- **GPU run (fp16 dense)**: 989→62876→122476→117821→… (diverges)

Weight-level identity is proven (≤2.46e-06 across 32 dense-column controls), but the
**full autoregressive token chain does not match**. Cause hypothesis: fp16-vs-fp32
accumulation over 40 layers shifts logits enough that argmax flips early. This is
reported as an **open research item** — not hidden as a success.

Evidence: `results/f7_cpu_seed989.log` (CPU), `results/f7gpu64.bewijs.json` (GPU).

## The physical floor (honest)
- 4.5 GB NVMe per token → at realistic 1-3 GB/s sustained, **~1-3 tok/s ceiling**
  even with perfect GPU kernels
- CPU-numpy currently delivers **~0.02 tok/s** — compute-bound, far below the floor
- Conclusion: the gap to close is compute (GPU), then IO (io_uring/alignment)

## Determinism & correctness checks
- Byte-identical reruns (EXIT=0)
- GPU fp16 vs CPU fp32 identity: **max 2.457e-06** (32 tensor controls)
- Engram gate fp64: **Δ = 0.0**
- FP8 E4M3 handling verified (exp=15 valid in this checkpoint — see `docs/model-format.md`)
