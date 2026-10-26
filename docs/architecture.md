# Architecture — the colibri principle

## Problem
A 511 GB MoE model vs. a 121 GB unified-memory node. You cannot hold the experts.

## Insight
MoE models split weights into two very different parts:

1. **Dense core** — attention, shared experts, output projection, norms, embeddings.
   Small (a few GB) but used for *every token, every layer*.
2. **Routed experts** — 384+1 per layer. Huge, but only **6 of 390** are active per
   token per layer.

Holding the experts resident is pure waste: 98.5% of them are idle on any given
token. So: **keep the dense core hot, stream the 6 active experts per layer from
NVMe on demand.**

## Data flow per token (decode)

```
for layer in 0..39:
    dense attention (resident RAM/VRAM)          ──── linear, cheap
    route → 6 active expert ids                    ──── gate (BF16, resident)
    for each active expert:
        read fp4 weights (random pread, NVMe)      ──── ~8 MB / expert
        dequant fp4→fp16 (CPU or GPU)
    MoE combine → residual
```

- **~4.5 GB read per token** (40 layers × 6 experts × ~17.7 MB fp4)
- Random-access pread (not sequential) — experts are scattered across 48 shards
- Expert cache: FIFO/LRU keeps recently used experts hot (real workloads reuse them)

## Why this works on one node

| Requirement | Reality |
|---|---|
| Dense core fits in RAM | 6.84 GB for 40 layers ✅ |
| NVMe can deliver ~4.5 GB/token | NVMe ~1-3 GB/s sustained → ceiling ~1-3 tok/s |
| Expert reuse during a conversation | high — routing is bursty, cache hits |

## The honest ceiling
NVMe bandwidth (~4.5 GB/token) sets a **physical ceiling** in the single-digit
tok/s range. GPU dequant/matmul (the active workstream) closes the compute gap;
io_uring / block-aligned packing closes the bandwidth gap. Expect single-digit
tok/s, not 100+.

## Generalization
Any MoE model where `dense_core ≪ RAM` but `total_weights ≫ RAM` qualifies:
- DeepSeek-V3 / V4 families (same MLA + MoE shape)
- Qwen-MoE, Mixtral, etc. (if fp4/8-quantized)
The packing, E4M3 finding, and streaming machinery are model-agnostic.
