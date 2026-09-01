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

The runtime is a text-only, correctness-first MLX implementation. All 42
routed MoE layers are required to use **ExpertSSD**: native MXFP4 expert records
remain in the composite safetensors files and only the eight experts selected
for the current token are loaded into each layer's bounded LRU. There is no
full resident expert-bank fallback and v1 does not use ScaleX.

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

Build the Python 3.13 environment and run unit tests (this does not load or run
the full model):

```bash
./build.sh
```

The command UX follows the DeepSeek project:

```bash
./run.sh --preflight
./run.sh --memory 24 --max-tokens 1 "Say hello in one word."
./run.sh --chat --memory 24
./run.sh --thinking --max-tokens 8 "Explain mHC briefly."
```

V1 deliberately runs the prompt one token at a time, uses recurrent KDA, and
fixes ExpertSSD capacity at eight per layer even when more memory is requested.
It caps the total prompt-plus-output context at 128 tokens. DSA uses its real MLA
projections but skips indexer scoring only inside the mathematically equivalent
short-context domain where every visible key fits within the official 2,048-key
selection budget. Vision, MTP, chunked prefill, ScaleX, and performance tuning
are deferred until after the first-token correctness run.

See [`docs/v1-impl.md`](docs/v1-impl.md) for the implementation and
qualification plan and [`docs/v1-status.md`](docs/v1-status.md) for the exact
tested/unrun boundary of the current code.
