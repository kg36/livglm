# LivGLM: GLM-5.3-Flash on Apple silicon

This small project prepares a LivSeek-oriented GLM-5.3-Flash checkpoint from
two pinned Hugging Face sources:

- the official Z.ai FP8 checkpoint supplies every non-routed tensor;
- the INCModel3 AutoRound checkpoint supplies all routed expert projections as
  group-32 MXFP4 (`weight_packed` + E8M0 `weight_scale`).

The result is intentionally a **custom composite checkpoint**. Its expert
tensor names and quantization differ from the official FP8 graph, so vanilla
Transformers compatibility is not claimed. The original Z.ai `config.json` is
kept verbatim, while `livseek-composite.json` records the actual provenance and
mixed quantization contract.

The downloader uses HTTP byte ranges and writes directly into the final
safetensors layout. It does not keep full copies of either source model. Work
is resumable at 256 MiB boundaries and no Git repository is required.

The runtime is a text-only MLX implementation. All 42 routed MoE layers are
required to use **ExpertSSD**: native MXFP4 expert records remain in the
composite safetensors files and `mlx-io-glm` reads misses directly into fixed,
wired per-layer slot arrays. A native Markov-LHD policy reuses experts across
tokens, while eight workers issue independent SSD reads in parallel. There is
no full resident expert-bank fallback and the runtime does not use ScaleX.
At startup, 497 active resident linear weights are converted in memory to
group-32 MXFP8; the official checkpoint is never rewritten. Use
`--resident-bf16` to retain the slower correctness-oracle representation.

For a larger exact-MXFP4 expert cache under the same hard memory ceiling, use
`--resident-mxfp4`. This quantizes only the resident linear matrices to MXFP4
at startup; routed expert files and the official checkpoint remain unchanged.
The option is faster in the measured 30 GiB short-decode workload, but has a
wider numerical error bound than the default MXFP8 path.

Default artifact location:

```text
/Users/kumargaurav/Documents/livglm/GLM53Flash
```

Commands:

```bash
# Uses MLX from the existing DSv4Flash environment for the numerical check.
./scripts/preflight.sh

# Build the plan and download/assemble the complete checkpoint.
./scripts/download.sh

# Audit all safetensors and produce SHA256SUMS.
./scripts/validate.sh
```

`download.sh` may be run again after interruption. Do not delete `.download/`
until validation has completed.

## Runtime

Build the Python 3.13 environment, the pinned `mlx-io-glm` overlay, and run unit
tests (this does not load or run the full model):

```bash
./build.sh
```

The command UX follows the DeepSeek project:

```bash
./run.sh --preflight
./run.sh --memory 24 --max-tokens 1 "Say hello in one word."
./run.sh --chat --memory 24
./run.sh --thinking --max-tokens 8 "Explain mHC briefly."
./run.sh --resident-bf16 --memory 30 --max-tokens 1 "Say hello."
./run.sh --resident-mxfp4 --memory 30 --max-tokens 20 "who are you"
./run.sh --memory 30 --max-tokens 25 --trace artifacts/perfetto/decode.json \
  --trace-decode-steps 24 "who are you"
```

`--memory` is a hard total MLX runtime ceiling, not an expert-cache allowance.
The runtime first reserves the selected resident graph and 3 GiB of execution
headroom, then converts only the remainder into equal-capacity ExpertSSD caches
across all 42 routed layers. With default MXFP8 resident execution,
`--memory 30` selects 33 slots per layer and a 29.553 GiB plan. The BF16 oracle
selects 19 slots under the same ceiling. Expert slots are allocated only after
the startup conversion, so the temporary BF16 graph cannot overlap the larger
cache. The prompt still runs one token at a time through recurrent KDA, and
total prompt-plus-output context is capped at 128 tokens.
DSA uses its real MLA projections but skips indexer scoring only inside the
mathematically equivalent short-context domain where every visible key fits
within the official 2,048-key selection budget. Vision, MTP, chunked prefill,
and ScaleX remain outside this runtime.

See [`docs/v1-impl.md`](docs/v1-impl.md) for the original correctness plan,
[`docs/v1-status.md`](docs/v1-status.md) for the first-token milestone, and
[`docs/native-expertssd-status.md`](docs/native-expertssd-status.md) for the
native I/O milestone. Current performance measurements are in
[`docs/mxfp8-performance.md`](docs/mxfp8-performance.md), and the first
correlated Perfetto/Metal bottleneck analysis is in
[`docs/perfetto-analysis.md`](docs/perfetto-analysis.md).
