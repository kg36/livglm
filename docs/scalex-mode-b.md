# Lossless ScaleX Mode B for GLM-5.3-Flash

Status: implemented and production-qualified on `perf/scalex-mode-b`. All 42
served expert layers in the local checkpoint are converted and validated.

## Representation

ScaleX does not requantize the routed experts. The three packed MXFP4 weight
tensors remain byte-identical. Only each expert's three U8 E8M0 scale tensors
are concatenated in gate/down/up order and encoded losslessly:

- one palette bit per scale when the two dominant exponent bytes suffice;
- two palette bits when four values produce the smaller record;
- sorted `(uint32 position, uint8 value)` exceptions; and
- raw storage as a worst-case fallback.

The layer container embeds the exact original safetensors prefix, original
file length and SHA-256. Its record order is:

```text
encoded gate+down+up scales | gate weight | down weight | up weight
```

That permits one native `preadv` per cache miss. Restoration reconstructs the
original interleaved safetensors payload and must match its complete SHA-256
before atomic promotion.

Mode B keeps the compressed record in each ExpertSSD slot. The existing
`mlx-io-glm` Metal QMV decodes one 512-scale tile into threadgroup memory while
performing the MXFP4 dot product. Packed weights and outputs remain exact.
The small uint16 tile directory is built by the native refill worker in this
first implementation; no persistent prefix sidecar is required.

## Measured GLM geometry

The read-only scan covered all 12,096 experts in main layers 3-44:

| Metric | Result |
|---|---:|
| Raw scale bytes | 9,512,681,472 |
| Encoded scale bytes | 1,603,900,029 |
| Encoded/raw scale ratio | 16.861% |
| One-bit experts | 12,067 |
| Two-bit experts | 29 |
| Average encoded scales/expert | 132,598 bytes |
| Complete expert-payload ratio | 95.109% |
| Main routed payload | 143.244 GiB |
| Main routed payload-only saving | 7.366 GiB |
| Actual container filesystem saving | 7.358 GiB |

The memory solver uses each layer's own maximum encoded record. A global
worst-case stride would unnecessarily lose one cache row. With startup MXFP4
resident weights and `--memory 30`:

| Cache | Slots/layer | Cache allocation | Total plan |
|---|---:|---:|---:|
| Native scales | 41 | 21.441 GiB | 29.957 GiB |
| ScaleX Mode B | 43 | 21.458 GiB | 29.974 GiB |

Forty-four uniform slots do not fit the hard 30 GiB plan.

## Qualification completed

- Python one-bit, two-bit and raw codec round trips.
- Synthetic layer-container compression, corruption rejection and exact
  restoration.
- Native Mode-B direct-to-slot loading of compressed records and packed
  weights.
- Real Layer 3 expert projection QMV equality for gate, down and up.
- A non-mutating conversion of the complete real Layer 3 shard:
  3.586 GiB native to 3.407 GiB ScaleX, saving 0.179 GiB.
- Exact restoration of that shard to SHA-256
  `e7b183551ddf6cb05510d0ad848e44e59de5b0002ccde4f44b7a835e8aa6b47c`.
- Real Layer 3 top-8 native versus ScaleX output equality at the BF16 bit
  level.
- Transactional conversion of all 42 served expert shards, followed by a full
  90-shard/76,108-tensor validation with 42 reconstructed SHA-256 checks and
  three remote byte samples.
- Full-runtime FP32 residual inputs are lazily cast inside the MLX graph to the
  BF16 input precision required by the fixed ScaleX Metal primitive. Real
  single-token and two-token expert paths match the decoded MXFP4 oracle.

The production 20-token generation completed with the expected text and exact
ExpertSSD accounting.

## Performance verdict

At `--memory 30 --resident-mxfp4`, ScaleX increases the uniform cache from 41
to 43 slots/layer and reduces decode I/O from 28.052 to 26.049 GiB for the
canonical 20-token prompt. The decode hit rate rises from 64.7% to 65.5%.
Measured decode speed was 4.025 token/s versus the earlier 4.180 token/s native
result, so ScaleX alone is not accepted as the missing 20% performance gain.

A direct production-geometry microbenchmark measured the complete ScaleX
Gate/Up/SwiGLU/Down chain at 0.862 ms per layer versus 0.899 ms for FP32
`gather_qmm`; in-kernel decompression is not the regression. A traced run
measured 107.6 ms/token of route/upstream-GPU wait and 122.4 ms/token of exposed
SSD join. A trace-guided non-uniform 34-73 slot allocation saved only 37 misses
and measured 4.016 token/s, so that experiment was removed.

### MTP and width-two follow-on

The figures above isolate ScaleX Mode B before speculative decoding. The next
phase retained ScaleX and added the official layer-45 MTP drafter, a width-two
target verifier, fused top-8 width-two ExpertSSD kernels, width-two resident
matrix kernels, and LivSeek-style attention-only MTP cache catch-up. The target
still verifies every emitted token.

The production CLI command

```bash
./run.sh --memory 30 --resident-mxfp4 --max-tokens 60 "who are you"
```

generated 44 tokens before EOS at 4.890 token/s over the full decode and 5.225
token/s after the first 20 output tokens. It accepted 21 of 22 MTP drafts,
reported 60.7% decode ExpertSSD hit rate, and stayed below the requested hard
ceiling at 29.80 GiB peak MLX memory. This is the first measured 5+ token/s
steady-state result; the cold-start-inclusive full-decode number remains
separately visible rather than being folded into the steady figure.

## Commands and safety

Build and run the complete unit suite first:

```bash
./build.sh
```

Create and verify a non-mutating production-geometry layer artifact:

```bash
./converter --layer 3 --output /path/to/layer-3.scalex \
  /Users/kumargaurav/Documents/livglm/GLM53Flash
```

Convert or re-verify all served layers transactionally and resumably:

```bash
./converter /Users/kumargaurav/Documents/livglm/GLM53Flash
```

Each layer is written and reconstructed successfully before its original shard
is atomically replaced. An interruption can leave a mixed checkpoint; the
runtime rejects that state, while rerunning the converter resumes it. Restore
the byte-identical native representation with:

```bash
./converter --native /Users/kumargaurav/Documents/livglm/GLM53Flash
```

`scripts/validate.sh` understands ScaleX virtual tensors and hashes the exact
reconstructed safetensors image. `scripts/download.sh` refuses to run over any
converted shard so it cannot silently replace ScaleX data.
