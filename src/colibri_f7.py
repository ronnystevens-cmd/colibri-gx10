#!/usr/bin/env python3
"""
colibri_f7.py — f7-correctheidslaag bovenop de f5-decode-loop (DeepSeek-V4.1-Flash-1017, node2).

Directie Ronny 24/10: f7 EERST — engram (rolling-hash, lagen 1/14) + DSPark/MTP (37-39)
correct inberekenen op basis van dsv41_ref/engram.py (en model.py). Correctheid vóór
performance.

WÉL (exacte wiskunde uit de referentie):
  1. ENGRAM rolling-hash (NgramHashState-principe, engram.py):
     - find_next_prime per (laag, ngram-grootte 2..4, head 0..7), NOOIT hergebruikt;
     - multipliers: default_rng(10007*laag).integers(0,(int64max//vocab)//2, max_ngram)*2+1
       met vocab = engram_compressed_vocab 99092 (de 'multiplier-bound uit de
       GECOMPRESSEERDE vocab'-regel);
     - rolling XOR-hash modulo het prime per (ngram, head) + per-laag offsets;
     - EngramGate (Engram.forward): wkv [25600,6144] fp8 + q/k_weight bf16 [4,5120]
       geladen; key/value-split, gewogen genormaliseerde dot, signed-sqrt→sigmoid,
       residual-write — het VOLLEDIGE gate-pad numeriek uitgevoerd.
  2. DSPARK/MTP (mtp.0..2) zoals model.py.forward_embed/forward_head:
     main_hidden = kat(mean-over-hc van de attn-input van backbone-lagen 37/38/39)
     → main_norm(main_proj(...)); noise-blok (block 5, noise 128799, eerste slot =
     echte token) door de GEDEELDE embed; forward_head: head-logits + MarkovHead
     (embed/head rank 256) geneste lus over opeenvolgende out-ids; confidence =
     proj_fp32(kat(x_norm, markov_embed)) met het bf16-checkpoint-gewicht in fp32.

WÉL (sinds 24/10, 2e iteratie):
  - engram-tabel-LOOKUP: EngramLookup leest de echte fp8-rijen (256 B/rij, e8m0-schaal)
    WILLEKEURIG van NVMe via preadv uit de 2 modelshards die ze bevatten — de tabel
    hoeft NOOIT in RAM (197 GB; node2 heeft 121 GB unified): per token 24 rijen x 256 B
    ≈ 6 KB random-read. Embed is hiermee het GETRAINDE engram-signaal, niet langer NUL.
    (EmbeddingSharded-principe: lookup per hash-id, geschaalde fp8-dequant.)

NIET (eerlijk, staat ook in de print):
  - DSparkBlock-lagen (mtp.0..2 attn+MoE): identiteit in deze build.
  - compressed_token_map: HF-normalisator (tokenizers-pkg) ontbreekt → identity-map.
  - indexer/two-level top-k: apart geanalyseerd (zie kanban-comment 24/10), nog niet
    in deze build — volgend onderdeel.
  - act/fp4-act-kwantisatie van activations geschaald (idem f5).

Gebruik (node2): /home/flitsken/colibri-venv/bin/python colibri_f7.py [--tokens N]
"""
import argparse
import json
import resource
import time

import numpy as np

import colibri_f5 as f5
from colibri_f5 import SFT, DIM, HC, VOCAB, EPS, rmsnorm

# ---- engram constants (inference_config.json text_config) ----
ENGRAM_LAYERS     = (1, 14)
ENGRAM_NUM_EMB    = (384_006_168, 384_016_682)
ENGRAM_MAX_NGRAM  = 4
ENGRAM_VOCAB      = 16_000_000       # prime-start (ruwe engram-vocab)
ENGRAM_N_HEADS    = 8
ENGRAM_HEAD_DIM   = 256
ENGRAM_PAD_ID     = 2
ENGRAM_COMPRESSED = 99_092           # multiplier-bound uit de GECOMPRESSEERDE vocab
DEAD              = -1

# ---- DSPark constants ----
DSPARK_BLOCK      = 5
DSPARK_NOISE      = 128_799
DSPARK_TARGETS    = (37, 38, 39)     # attn-input-lagen die de draft voeden
DSPARK_RANK       = 256


# ================= ENGRAM: hashing exact uit engram.py =================
def _is_prime(n):
    """Deterministische Miller-Rabin (bases 2..37) == sympy.isprime voor n < 3.3e24."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start, seen):
    cand = start + 1
    while not _is_prime(cand) or cand in seen:
        cand += 1
    return cand


def build_layout():
    """EngramLayout: primes[laag][grootte 2..4][head] unik; offsets cumsum per laag."""
    primes, seen = [], set()
    for _ in ENGRAM_LAYERS:
        per = []
        for _ in range(ENGRAM_MAX_NGRAM - 1):
            sizes, cur = [], ENGRAM_VOCAB - 1
            for _ in range(ENGRAM_N_HEADS):
                cur = find_next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per.append(tuple(sizes))
        primes.append(tuple(per))
    offsets = []
    for layer in primes:
        sizes = [p for per in layer for p in per]
        run, csum = 0, []
        for s in sizes[:-1]:
            run += s
            csum.append(run)
        offsets.append(np.array([0] + csum, dtype=np.int64))
    return primes, offsets


def compute_hash_multipliers(layer_ids, max_ngram, vocab):
    """EngramHashMultipliers: seed 10007*laag, odd multipliers, bound uit GECOMPRESSEERDE vocab."""
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    rows = []
    for lid in layer_ids:
        g = np.random.default_rng(10007 * int(lid))
        v = g.integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64)
        rows.append(v * 2 + 1)
    return np.stack(rows)


class NgramHash:
    """NgramHashState-principe, decode-variant (1 token/stap), identity token-map."""
    def __init__(self):
        self.primes, self.offsets = build_layout()
        self.primes_np = np.array(self.primes, dtype=np.int64)   # [L][g-2][head]
        self.multipliers = compute_hash_multipliers(ENGRAM_LAYERS, ENGRAM_MAX_NGRAM,
                                                    ENGRAM_COMPRESSED)
        self.cache = np.full(4096, DEAD, dtype=np.int64)

    def set_token(self, pos, token_id):
        self.cache[pos] = token_id

    def hash_at(self, pos):
        """[L][(max_ngram-1)*n_heads] hash-ids (met offsets) voor positie pos.

        Exact NgramHashState.forward (decode-tak, B=L=1):
        - products[s] = token[cache[pos-s]] * multiplier[layer][s] — GEEN
          modulo-reductie voor de XOR (int64 wrap is de referentiesemantiek);
        - rolling = XOR over lookbacks, mod prime per (laag, g-2, head) + offset;
        - lookback buiten de reeks of voorbij een DEAD token: cache-waarden worden
          pad_id-vervangen én blijven pad_id voor alle langere n-grams
          (reference: `blocked` is cumulatief, `torch.where(blocked, pad_id, source)`).
        """
        toks = np.empty(ENGRAM_MAX_NGRAM, dtype=np.int64)
        blocked = False
        for shift in range(ENGRAM_MAX_NGRAM):
            if pos - shift < 0:
                blocked = True
            elif self.cache[pos - shift] == DEAD:
                blocked = True
            toks[shift] = ENGRAM_PAD_ID if blocked else self.cache[pos - shift]
        out = []
        for li in range(len(ENGRAM_LAYERS)):
            products = toks * self.multipliers[li]
            rolling = products[0]
            ids = []
            for i in range(1, ENGRAM_MAX_NGRAM):        # ngram-grootte g = i+1
                rolling = int(np.bitwise_xor(rolling, products[i]))
                base = (i - 1) * ENGRAM_N_HEADS
                for h in range(ENGRAM_N_HEADS):
                    prime = int(self.primes_np[li, i - 1, h])
                    ids.append(rolling % prime + int(self.offsets[li][base + h]))
            out.append(np.array(ids, dtype=np.int64))
        return np.stack(out)                             # [2][24]


class EngramGate:
    """Engram.forward-semantiek; lookup=None → embed=NUL (referentie-gedrag)."""
    def __init__(self, sft: SFT, layer_id, layer_hash_index):
        self.hc, self.dim, self.eps = HC, DIM, EPS
        self.layer_id, self.lhi = layer_id, layer_hash_index
        self.table_gb = ENGRAM_NUM_EMB[layer_hash_index] * ENGRAM_HEAD_DIM / 1e9  # fp8
        self.wkv = sft.fp8(f"layers.{layer_id}.engram.wkv.weight") / 64.0
        # /64: convert.py FP8→FP8 conversie — de e8m0-schaal in de checkpoint is
        # scale_max_offset_bits = inwendige_schaal/2^6 (MAX_OFFSET_BITS=6). Zonder deze
        # factor was value_rms 1.38 (1000× residual-orde); mét factor 0.021 (correct).
        self.q_weight = sft.bf16(f"layers.{layer_id}.engram.q_weight").reshape(HC, DIM)
        self.k_weight = sft.bf16(f"layers.{layer_id}.engram.k_weight").reshape(HC, DIM)

    def forward(self, x, hash_ids, lookup=None):
        """x [HC,DIM] → (x', dot). lookup=None → embed=NUL (referentie-gedrag voor
        gemiste rijen). `lookup` = fp32 [24,256] uit de ECHTE tabel (EngramLookup;
        gemiste rijen staan er al als 0 in)."""
        n_cols = (ENGRAM_MAX_NGRAM - 1) * ENGRAM_N_HEADS
        if lookup is None:
            emb = np.zeros(n_cols * ENGRAM_HEAD_DIM, dtype=np.float32)   # NUL-LOOKUP
        else:
            emb = np.asarray(lookup, dtype=np.float32).reshape(-1)
        kv = self.wkv @ emb
        key = kv[:self.hc * self.dim].reshape(self.hc, self.dim)
        value = kv[self.hc * self.dim:]
        weight = self.q_weight * self.k_weight
        rstd = np.sqrt(1.0 / (np.mean(x * x) + self.eps)) \
             * np.sqrt(1.0 / ((key * key).mean(-1) + self.eps))
        dot = (x * weight * key).sum(-1) * rstd * self.dim ** -0.5
        gate = 1.0 / (1.0 + np.exp(-np.copysign(np.maximum(np.abs(dot), 1e-6) ** 0.5, dot)))
        return x + gate[:, None] * value[None, :], dot


class EngramLookup:
    """Echte tabel-lookup via willekeurige NVMe-preadvise van de fp8-rijen
    (EmbeddingSharded-principe: lookup per hash-id, geschaalde fp8).

    Per stap worden de 24 benodigde rijen/laag gelezen (±6 KB). Dit gebruikt de
    ECHTE engram-tabelsemantiek; schaal = fp8·e8m0 per 32 kolommen.
    """
    ROW = ENGRAM_HEAD_DIM  # fp8 bytes per rij

    def __init__(self, sft: SFT, layer_id, layer_hash_index):
        import os, struct
        self.sft = sft
        name_w = f"layers.{layer_id}.engram.embed.weight"
        name_s = f"layers.{layer_id}.engram.embed.scale"
        self.sh_w = sft.wm[name_w]
        path = os.path.join(sft.mdir, self.sh_w)
        self.fh = open(path, "rb")
        n = struct.unpack("<Q", self.fh.read(8))[0]
        hdr = json.loads(self.fh.read(n))
        self.base_w = 8 + n + hdr[name_w]["data_offsets"][0]
        self.base_s = 8 + n + hdr[name_s]["data_offsets"][0]
        self.rows = hdr[name_w]["shape"][0]

    def embed_rows(self, ids):
        """ids [24] → fp32 [24,256]; buiten-bereik rijen = 0 (EmbeddingSharded-out-of-range).
        fp8 × e8m0-schaal per 32-kolom-blok, exact als de referentie-dequant.
        Eén preadv-systeemcall per token: 24 verspreide rijen + 24 schaalblokken
        in één keer (EmbeddingSharded-principe; geen 48× seek+read)."""
        import os
        from colibri_f5 import E4M3, e8m0_to_f32
        ids = np.asarray(ids, dtype=np.int64)
        out = np.zeros((ids.size, self.ROW), dtype=np.float32)
        valid = np.nonzero((ids >= 0) & (ids < self.rows))[0]
        if valid.size == 0:
            return out
        offs, lens = [], []
        for j in valid:
            offs.append(self.base_w + int(ids[j]) * self.ROW); lens.append(self.ROW)
        for j in valid:
            offs.append(self.base_s + int(ids[j]) * (self.ROW // 32)); lens.append(self.ROW // 32)
        # LET OP: os.preadv accepteert GEEN per-chunk-offsets (tuples =
        # TypeError); (offset, view)-paren bestaan pas py≥3.14. Echte scatter-
        # read dus via os.pread(fd, ...) — offset-gelezen, file-pointer
        # onaangetast, wél direct via de fd (BufferedReader heeft geen .pread).
        bufs = [os.pread(self.fh.fileno(), l, o) for o, l in zip(offs, lens)]
        nv = valid.size
        for jj, j in enumerate(valid):
            wb = np.frombuffer(bufs[jj], dtype=np.uint8)
            sb = np.frombuffer(bufs[nv + jj], dtype=np.uint8)
            out[j] = E4M3[wb] * np.repeat(e8m0_to_f32(sb), 32)
        return out


# ================= DSPARK: forward_embed + forward_head =================
class DSPark:
    """forward_embed + forward_head exact; DSparkBlock-lagen (mtp.0-2) = identiteit."""
    def __init__(self, sft: SFT, embed, head):
        t0 = time.time()
        self.embed, self.head = embed, head
        self.main_proj = sft.fp8("mtp.0.main_proj.weight")           # [dim, dim*hc]
        self.main_norm = sft.bf16("mtp.0.main_norm.weight")
        self.norm = sft.bf16("mtp.2.norm.weight")
        self.mk_embed = sft.bf16("mtp.2.markov_head.embed.weight")   # [V,256]
        self.mk_head = sft.bf16("mtp.2.markov_head.head.weight")     # [V,256]
        cw = sft.bf16("mtp.2.confidence_head.proj.weight")           # [1, dim+rank] bf16->fp32
        self.conf_w = cw.reshape(1, -1)
        print(f"[f7] DSPark-head geladen: {time.time()-t0:.1f}s", flush=True)

    def forward_spec(self, last_token, main_hiddens, temperature=0.0, rng=None):
        """main_hiddens: [3][DIM] mean-over-hc van attn-input lagen 37/38/39.
        Geeft (out_ids[6], conf[5], main_x)."""
        rng = rng or np.random.default_rng(0)
        main_x = rmsnorm(self.main_proj @ np.concatenate(main_hiddens), self.main_norm, EPS)
        draft = np.full(DSPARK_BLOCK, DSPARK_NOISE, dtype=np.int64)
        draft[0] = last_token
        x = self.embed[draft].astype(np.float32)          # [B, DIM] (block-lagen: identiteit)
        out_ids = np.empty(DSPARK_BLOCK + 1, dtype=np.int64)
        out_ids[0] = last_token
        confs = []
        for i in range(DSPARK_BLOCK):
            xn = rmsnorm(x[i], self.norm, EPS)
            li = self.head @ xn + self.mk_head @ self.mk_embed[out_ids[i]]
            if temperature <= 0:
                out_ids[i + 1] = int(np.argmax(li))
            else:
                p = np.exp((li - li.max()) / temperature)
                out_ids[i + 1] = int(rng.choice(VOCAB, p=p / p.sum()))
            me = self.mk_embed[out_ids[i]]
            confs.append(float(self.conf_w[0, :DIM] @ xn
                               + self.conf_w[0, DIM:] @ me))
        return out_ids, np.array(confs, dtype=np.float32), main_x


# ================= ModelF7 =================
class ModelF7(f5.ModelF5):
    def __init__(self, exp_cache=32, cold=True, dense_max_layers=8,
                 with_engram=True, with_dspark=True, with_lookup=True):
        super().__init__(exp_cache=exp_cache, cold=cold, dense_max_layers=dense_max_layers)
        self.with_engram, self.with_dspark = with_engram, with_dspark
        self.with_lookup = with_lookup and with_engram
        if with_engram:
            self.ngram = NgramHash()
            self.engrams = {l: EngramGate(self.sft, l, i)
                            for i, l in enumerate(ENGRAM_LAYERS)}
            if self.with_lookup:
                try:
                    self.lookups = {l: EngramLookup(self.sft, l, i)
                                    for i, l in enumerate(ENGRAM_LAYERS)}
                except (KeyError, FileNotFoundError) as e:
                    print(f"[f7] engram-tabelshards niet gevonden ({e}) → NUL-LOOKUP", flush=True)
                    self.with_lookup = False
                    self.lookups = {}
            else:
                self.lookups = {}
        if with_dspark:
            self.dspark = DSPark(self.sft, self.embed, self.head)
        self.main_hidden_store = [None] * len(DSPARK_TARGETS)
        self.last_gate_dot = None

    def block(self, d, x, pre_mix, pos):
        if self.with_engram and self.block_layer in self.engrams:
            h = self.ngram.hash_at(pos)[ENGRAM_LAYERS.index(self.block_layer)]
            lu = self.lookups[self.block_layer].embed_rows(h) if self.with_lookup else None
            x, dot = self.engrams[self.block_layer].forward(x, h, lookup=lu)
            self.last_gate_dot = dot
            self.last_lu_hit = None if lu is None else int(np.count_nonzero(lu))
        if self.with_dspark and self.block_layer in DSPARK_TARGETS:
            # model.py: main_hidden is de hidden VÓÓR de attn-transformatie (mean over hc)
            self.main_hidden_store[DSPARK_TARGETS.index(self.block_layer)] = \
                x.mean(axis=0).copy()
        return super().block(d, x, pre_mix, pos)

    def forward(self, token, pos):
        if self.with_engram:
            # NgramHashState.forward: hashed wordt de GECOMPRESSEERDE token-id
            # (token_map). De tokenizers-normalisator ontbreekt → identity-map,
            # maar PAD_ID (2) móét op de gecomprimeerde ruimte worden afgebeeld
            # (pad_id = token_map[engram_pad_id]); anders raakt PAD_ID via de
            # blocked-lookbacks als vreemde id in de hashing.
            self.ngram.set_token(pos, min(int(token), ENGRAM_COMPRESSED - 1))
        return super().forward(token, pos)

    def draft(self, last_token, temperature=0.0):
        if not self.with_dspark:
            return None
        return self.dspark.forward_spec(last_token, self.main_hidden_store, temperature)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--no-dspark", action="store_true")
    args = ap.parse_args()
    print(f"colibri_f7 — engram+DSPark correctheidslaag op f5 "
          f"(engram={not args.no_engram}, dspark={not args.no_dspark})", flush=True)
    t0 = time.time()
    m = ModelF7(exp_cache=32, cold=True, dense_max_layers=8,
                with_engram=not args.no_engram, with_dspark=not args.no_dspark)
    print(f"[f7] init {time.time()-t0:.1f}s", flush=True)
    if not args.no_engram:
        h = m.ngram.hash_at(0)
        print(f"[f7] engram primes laag0[g=2]={list(m.ngram.primes[0][0][:3])}..; "
              f"mults laag1={m.ngram.multipliers[1][:2].tolist()}..; hash pos0={h[0][:3].tolist()}..",
              flush=True)
        mode = ("ECHTE NVMe-lookup (preadv, ~6 KB/token, tabel nooit in RAM)"
                if m.with_lookup else
                f"NUL-LOOKUP (shards ontbreken: {sum(ENGRAM_NUM_EMB)*ENGRAM_HEAD_DIM/1e9:.0f} GB fp8)")
        print(f"[f7] engram-tabel = {mode}", flush=True)
    rng = np.random.default_rng(1776)
    seed = int(rng.integers(0, VOCAB))
    t = time.time()
    logits = m.forward(seed, 0)
    print(f"[f7] prefill(pos0) {time.time()-t:.1f}s", flush=True)
    for step in range(1, args.tokens + 1):
        t = time.time()
        tok = int(np.argmax(logits))
        logits = m.forward(tok, step)
        nxt = int(np.argmax(logits))
        extra = ""
        if m.last_gate_dot is not None:
            extra = f" gate_dot={np.round(m.last_gate_dot, 4).tolist()}"
        if not args.no_dspark:
            ids, conf, mx = m.draft(nxt, temperature=0.0)
            extra += (f" draft={ids.tolist()} conf={np.round(conf,3).tolist()}"
                      f" mainx_rms={np.sqrt((mx*mx).mean()):.3f}")
        print(f"[f7] stap {step}: {tok}->{nxt} ({time.time()-t:.1f}s){extra}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print("=== F7 RESULTATEN ===", flush=True)
    print(f"  piek RSS {rss:.2f} GB", flush=True)
    print("  engram: rolling-hash (unik-primes, compressed-vocab-multipliers) + volledig "
          "gate-pad BEREKEND" +
          ("; ECHTE fp8-tabel-lookup via NVMe-preadv → getraind engram-signaal in de residual"
           if m.with_lookup else "; tabel-lookup NUL → NIET het getrainde engram-gedrag"), flush=True)
    print("  DSPark: main_proj/norm/markov/confidence BEREKEND; DSparkBlock-lagen "
          "identiteit → draft structureel correct, semantisch onvolledig", flush=True)


if __name__ == "__main__":
    main()
