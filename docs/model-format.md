# Model format notes — DeepSeek-V4.1-Flash-1017 (reverse-engineered, verified)

> Everything here was verified against the model's own `config.json`,
> `model.safetensors.index.json`, and reference implementation files. It is the
> practical knowledge needed to read this checkpoint from disk without the
> original inference stack.

## Key dimensions

- `hidden_size`: 5120, `vocab_size`: 129280, `num_hidden_layers`: 40
- 64 heads × 512 (MLA), `q_lora`: 1280, `o_lora`: 1024, `o_groups`: 8
- Context: 1M (YaRN ×16), vision-capable
- MoE: 384 routed + 1 shared expert, **6 active per token**, `moe_inter`: 2304,
  `route_scale`: 1.5

## Attention KV compression (`compress_ratios`)

`compress_ratios [0, 0, 2×18, 1×20, 0, 0, 0]` — this is **KV compression of the
attention**, NOT the expert-input dimension (a common misreading):

- Layer 0/1: sliding-window only (ratio 0)
- Layers 2–19: ratio 2 (compressed KV)
- Layers 20–39: ratio 1
- `kv_source_layers [2,8,14,20]` produce the compressed KV cache
- `index_source [2,8,14,20,24,28,32,36]`

The "2560" seen in some `w1` byte sizes is **fp4 packing** (2 values/byte), not a
dimension change. Experts are uniform 2304→5120 in all layers.

Other config: `hc_mult` 4 (hyper-connections, 4 copies of the residual stream,
sinkhorn 20 iters). Engram layers: [1, 14]; DSPark/MTP target: [37, 38, 39];
MTP: 3 layers (40–42).

## Weight formats

### Experts — FP4 (E2M1), 2-per-byte packed
- 4-bit E2M1 nibbles, `low = b & 0x0F`, `high = (b >> 4) & 0x0F`
- FP4 lookup table over the 16 values
- E8M0 block scales, 1 scale per 32 columns (16 bytes)
- Byte-shape = half of the logical fp16 dimension

### Dense tensors — F8 (E4M3) + E8M0
- Attention / shared expert / output projection
- Block scales: 32×32 blocks

### ⚠ Empirical E4M3 finding (important!)
This checkpoint uses **`exp=15` as a VALID exponent** (0x78 / 0xF8 = ±256,
**not** inf/NaN). Only `0x7F` and `0xFF` are NaN.

Default E4M3FN tables (which treat `e=15` → NaN) produce **~1.5% NaN** in dense
tensors — which corrupts gate scores. If you dequantize this model and get NaNs,
this is why. Use a custom table with exp=15 valid.

### Gate
- BF16 [384, 5120] + bias / bias_vl F32 [384]
- sqrtsoftplus + bias for routing selection; sqrtsoftplus (no bias) for weighting

## Tensor layout / packing
- Safetensors: header read once per shard; tensor bytes fetched on demand from NVMe
- Each `.weight` has a sibling `.scale` tensor (E8M0 block scales); load both when
  dequantizing — a missing `.scale` raises `KeyError` in F8/fp4 paths

## Verified reference matches
- Engram `NgramHashState`: primes via Miller-Rabin ≡ sympy.isprime, offsets =
  cumsum(primes[:-1]), multipliers seeded 10007·layer from the compressed vocab
  (identity map), XOR rolling without modulo reduction (int64 wrap), cumulative
  `blocked` (pad stays pad), start-blocking. Hash IDs exactly equal to the real
  state over 12 positions.
- Engram gate: per-copy rstd, signed-sqrt, sigmoid, q·k product. fp64 golden
  comparison Δ = 0.0.
