# Contributing

This is a **proof-of-concept with real numbers**. We publish the slow parts too —
honesty over hype. All help that moves the needle on real hardware is welcome.

## High-value contributions

### 1. GPU kernels (biggest win)
Current bottleneck: fp4→fp16 dequant + GEMV on CPU-numpy (~0.02 tok/s).
Target: CUDA/tilelang/Triton kernels for:
- fp4 (E2M1) dequant + 5120×2304 GEMV, batched over 6 experts
- fp8 (E4M3) dense + engram GEMVs
- KV-compressed attention (MLA) kernel
- GEMV calibration reference: experts 0.252 ms, engram 1.485 ms (fp16, GB10)

### 2. NVMe / IO tuning
- io_uring async pread (currently synchronous pread)
- Expert block alignment + larger coalesced reads
- Prefetch next-layer experts during current-layer compute
- 4.5 GB/token is the ceiling — anything that cuts bytes or overlaps IO wins

### 3. Smarter caching
- LRU/FIFO currently; a learned or heuristic policy tuned on real routing
  distributions would cut NVMe traffic measurably

### 4. Generalization
- Port `colibri_f5.py` to other MoE checkpoints (Qwen-MoE, Mixtral, V3/V4)
- Validation harness: decode 64 tokens, compare against the original stack
  (identity ≤ 1e-5 is our bar)

## Acceptance criteria for results
- **Reproducible**: include the exact command, env (numpy/torch versions), hardware
- **Verified**: compare against the golden reference (see `results/`) — show the
  identity/Δ values, not just "it runs"
- **Honest**: if it's slower, say so. We benchmark reality, not PR.

## Project structure
See README «Repository layout». New files go in `src/`, new numbers in `results/`.

## License
MIT for this repo (unless a file says otherwise). Do **not** redistribute model
weights — point to the original source instead.
