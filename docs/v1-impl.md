# GLM-5.3-Flash ExpertSSD v1 implementation plan

> Historical plan: the first-token path described here was completed. See
> [`v1-status.md`](v1-status.md) for that milestone and
> [`native-expertssd-status.md`](native-expertssd-status.md) for the current
> native direct-to-slot implementation.

Status: implementation plan only; no runtime code has been written as part of this document.

Date: 2026-09-01

## 1. Decision

The first implementation will be a text-only, target-only GLM-5.3-Flash runtime in stock MLX. It will keep the official Z.ai non-routed weights resident and read the routed MXFP4 experts from the existing composite checkpoint on demand, using the same fundamental ExpertSSD seam as LivSeek's DeepSeek runtime.

The first milestone is deliberately narrow: process one short prompt, produce one finite `(1, 1, 154880)` logits tensor, choose one greedy token, and decode it. Performance work begins only after that path is repeatable and its component-level parity tests pass.

The v1 correctness path will:

- build only language layers 0 through 44;
- exclude the vision tower and MTP/NextN layer 45;
- construct no full routed-expert parameter tables;
- dequantize the official block-FP8 resident linears to BF16 once at startup;
- keep routed experts in their downloaded group-32 MXFP4 representation;
- use a synchronous Python `pread` reader and a bounded per-layer expert cache;
- run prefill one token at a time through the recurrent KDA path;
- use dense causal DSA under a hard short-context cap, but only after proving that this selects the same keys as the official indexer in that domain;
- use the official tokenizer and chat template; and
- force greedy decoding for the first-token qualification run.

This baseline intentionally does not depend on `mlx-io-glm`. The native fork becomes useful after the stock-MLX output is a correctness oracle.

## 2. Hardware reality and target profiles

The machine hosting this worktree currently reports:

| Item | Observed value |
| --- | --- |
| Machine | MacBook Pro `Mac17,9` |
| SoC | Apple M5 Pro, 15 CPU cores |
| Unified memory | 48 GiB |
| Internal SSD | approximately 1 TB |
| Free data-volume space at planning time | approximately 178 GiB |

The intended deployment machine discussed for this project is a 256 GB Mac Studio. These are not the same machine. The first-token profile will therefore be conservative enough to run on the present 48 GiB development Mac; the 256 GB Studio is a strict memory superset and should run the same profile unchanged before any larger-cache tuning.

Two named profiles should eventually exist:

| Profile | Purpose | Initial expert capacity | Initial context cap |
| --- | --- | ---: | ---: |
| `first-token-safe` | Correctness on the current 48 GiB host and the Studio | 8 experts per MoE layer | 128 tokens |
| `studio-256` | Post-correctness tuning on the 256 GB Studio | Start identically; tune only from measured traces | Do not increase until real DSA is qualified |

No additional full model or converted checkpoint should be written to disk. The existing model already occupies about 169.66 GiB, and the development machine has insufficient free space for a second full representation with safe operating headroom.

## 3. Frozen sources and references

### Runtime checkpoint

The runtime source is the already validated composite at:

```text
/Users/kumargaurav/Documents/livglm/GLM53Flash
```

Its two payload sources are frozen in `SOURCES.json` and `livseek-composite.json`:

| Role | Repository | Revision |
| --- | --- | --- |
| Official non-routed remainder | `zai-org/GLM-5.3-Flash` | `03eb5366286afd40d2221b1d9c63a6dd1ba4832e` |
| Routed MXFP4 experts | `INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound` | `8712b4a299e2cbb81c019d2c20084fb99cbc2d00` |
| BF16 numerical validation source | `zai-org/GLM-5.3-Flash-BF16` | `61f77a1e1a67c410650ce5017411337da0dcd11a` |

The assembled checkpoint has already passed full validation:

- 90 safetensors shards;
- 76,108 tensors;
- 182,163,075,960 tensor-payload bytes;
- 182,174,011,528 physical shard bytes;
- full SHA-256 recomputation for all 90 shards; and
- 12/12 routed-expert numerical preflight probes.

Runtime startup must require `VALIDATION.json` with `status: PASS`. It must read the public index, manifests, and safetensors headers; it must not depend on `.download/assembly-plan.json`, which is downloader state rather than a runtime contract.

### Mathematical graph reference

The primary graph reference is Hugging Face Transformers commit:

```text
69a7fb1aca7f2e7487294846be5859ebb6db9462
```

The relevant source is `src/transformers/models/glm5_next/modeling_glm5_next.py`. This is a specification donor, not code to trust blindly: GLM-5.3-Flash support was added only on 2026-08-26, and its upstream full-weight integration test was still skipped at the inspected revision. The official checkpoint also uses several checkpoint-native names and layouts that do not directly match the generated PyTorch module names. Every mapping must therefore be explicit and tested.

The small differential-test checkpoint is:

```text
inference-optimization/GLM-5.3-Flash-0.1B-A0.1B
revision 8311399447eba9c9b215e3209ab6f25e59c7d21e
```

It preserves the important topology in a manageable BF16 model: mHC, dense layers, MoE layers, KDA, DSA/indexer, and the multimodal wrapper. It is an oracle fixture, not a quality model.

### MLX and local structural reference

- Stock correctness runtime: MLX `v0.32.0` (`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`). This version exposes the working `mxfp4` `quantized_matmul` and `gather_qmm` paths already used by the preflight.
- Read-only ExpertSSD donor: `/Users/kumargaurav/Documents/LLMs/DSv4Flash/main` at commit `107bfe8d98b8e9ec5a78f8c47565daf54f0dc263`.
- Native dependency repository: `kg36/mlx-io-glm`. It had no `HEAD` at planning time and is not a v1 dependency. Its first usable revision must be pinned before native work starts.

The DeepSeek worktree remains strictly read-only. GLM code must be implemented independently in this repository; no symlinks, edits, generated files, or build products may be placed in the DeepSeek tree.

## 4. Exact v1 model contract

### Architecture

The text target has:

| Property | Value |
| --- | ---: |
| Hidden width | 4,096 |
| Main decoder layers | 45, numbered 0-44 |
| Dense MLP layers | 0-2 |
| Sparse MoE layers | 3-44, 42 layers total |
| Routed experts per sparse layer | 288 |
| Selected experts per token | 8 |
| Shared experts per sparse layer | 1 |
| Dense MLP width | 12,288 |
| Expert/shared MLP width | 2,048 |
| KDA layers | 34 |
| DSA layers | 11: 3, 7, 11, ..., 43 |
| mHC streams | 4 |
| Vocabulary | 154,880 |
| MTP layers | 1 additional layer, checkpoint layer 45; excluded from v1 |

The KDA configuration is 64 heads of width 128 with a causal convolution kernel of 4. The DSA/MLA configuration uses 64 heads, query LoRA rank 1,536, KV LoRA rank 512, no RoPE dimensions, and 256-dimensional key/value heads. The DSA indexer has 32 heads of width 128, pool size 4, and a 2,048-token selection budget.

The forward structure is:

```text
token ids
  -> resident BF16 embedding
  -> replicate into four mHC streams
  -> 45 decoder blocks
       attention site: mHC -> RMSNorm -> KDA or DSA -> mHC combine
       MLP site:       mHC -> RMSNorm -> dense MLP or MoE -> mHC combine
         MoE = resident FP32 router
             + ExpertSSD routed MXFP4 experts
             + resident shared expert
  -> unweighted mean of four streams
  -> final RMSNorm
  -> resident BF16 language head
  -> greedy token id
```

### Tensor partition

The runtime must derive this classification from config, names, the index, and headers. Counts are assertions, not selection logic.

| Partition | Tensors | Source payload | v1 treatment |
| --- | ---: | ---: | --- |
| Main text non-routed | 1,425 | 14.180311 GiB | Load resident; 179 FP8 scale tensors are absorbed during dequantization, yielding 1,246 runtime arrays |
| Main routed experts, layers 3-44 | 72,576 | 150.609375 GiB | Remain on SSD; demand-read through ExpertSSD |
| MTP layer 45 non-routed | 32 | 0.227124 GiB | Reject from v1 graph |
| MTP layer 45 routed experts | 1,728 | 3.585938 GiB | Reject from v1 graph |
| Vision tower | 347 | 1.049837 GiB | Reject from v1 graph |
| Total v1 source contract | 74,001 | 164.789686 GiB | Exactly once: resident or ExpertSSD |
| Total explicitly ignored | 2,107 | approximately 4.863 GiB | Only vision and MTP |

The contract gate fails on any missing, duplicate, or unclassified tensor. A new upstream tensor is not silently ignored.

### Checkpoint-to-runtime mapping rules

The implementation should keep checkpoint-native names where that avoids unnecessary transforms, but the mapping must be centralized and bijective. At minimum:

- strip or account for the `model.language_model` wrapper consistently;
- map `hc_attn_{fn,base,scale}` into the attention mHC component;
- map `hc_ffn_{fn,base,scale}` into the FFN mHC component;
- preserve the separate `q_conv1d`, `k_conv1d`, and `v_conv1d` weights, or explicitly concatenate them into the reference combined convolution layout;
- keep `mlp.gate.weight` and `mlp.gate.e_score_correction_bias` in FP32 routing math;
- map each expert's `gate_proj`, `up_proj`, and `down_proj` by name, never by assumed physical order; and
- absorb each `*.weight_scale_inv` into its paired resident FP8 weight and do not retain it as a runtime parameter.

The loader must generate both sets—expected runtime tensors and classified source tensors—and prove the mapping is one-to-one before allocating large arrays.

## 5. ExpertSSD source contract

GLM's expert layout is better for direct reads than the current DeepSeek source layout, but it is different and must not inherit DeepSeek's two-range assumption.

For every one of the 12,384 experts in layers 3-45, the six MXFP4 tensors are in one shard and occupy one exact contiguous range:

| Component | Source dtype and logical role |
| --- | --- |
| `down_proj.weight_packed` | U8-packed MXFP4 weight |
| `down_proj.weight_scale` | U8 E8M0 group-32 scales |
| `gate_proj.weight_packed` | U8-packed MXFP4 weight |
| `gate_proj.weight_scale` | U8 E8M0 group-32 scales |
| `up_proj.weight_packed` | U8-packed MXFP4 weight |
| `up_proj.weight_scale` | U8 E8M0 group-32 scales |

One record is exactly 13,369,344 bytes (12.75 MiB). Main-target v1 uses 12,096 such records in layers 3-44.

The source planner must:

1. Resolve tensor-to-shard membership from `model.safetensors.index.json`.
2. Read each referenced safetensors header and calculate absolute offsets as `8 + header_length + data_offset`.
3. Verify the six tensors' dtype and shapes.
4. Sort the six ranges by physical offset and coalesce only exact adjacency.
5. Assert one range and one shard for the current artifact, while retaining a generic multi-range representation so a future layout change fails safely rather than corrupting reads.
6. Record source and MLX destination metadata for each component.
7. Assert uniform 13,369,344-byte payloads and zero read amplification.

The MLX representation is byte preserving:

- packed U8 weights are viewed as `uint32`, reducing the last stored dimension by four;
- scale bytes remain `uint8`; and
- `mx.quantized_matmul`/`mx.gather_qmm` use `transpose=True`, `group_size=32`, `bits=4`, and `mode="mxfp4"`.

The synchronous reader keeps at most the 42 main expert-shard file descriptors open and uses a short-read-safe `os.pread` loop. It must issue exactly one useful-range request per missed expert for this artifact. Reader metrics must distinguish logical misses, system read calls, bytes requested, bytes returned, and short-read retries.

## 6. Correctness-first resident representation

The official remainder uses block FP8 E4M3 weights with F32 `weight_scale_inv` tensors for 179 main-text linears. Stock MLX 0.32 does not expose that safetensors FP8 dtype as a normal array dtype, so v1 will dequantize these weights to BF16 once at startup.

For a weight block, the intended transform is:

```text
BF16_weight[i, j] = BF16(FP32(E4M3FN_byte[i, j])
                         * weight_scale_inv[i // 128, j // 128])
```

All 179 FP8 weights have exactly one F32 scale partner. The pair geometry must be checked against 128-by-128 blocks before conversion. The 179 scale arrays are discarded after their paired weight has been materialized.

The E4M3FN byte codec is safety-critical. Before any full load it must pass:

- an exhaustive all-256-byte comparison with PyTorch's `float8_e4m3fn`, treating NaN encodings explicitly;
- block-broadcast shape tests, including non-square linears;
- BF16 rounding tests; and
- sampled tile comparisons against the pinned BF16 GLM source, preferably one tile from every one of the 179 FP8 weights, with numerical thresholds recorded in a report.

Resident loading is streamed one tensor at a time, preferably shard by shard:

1. mmap one composite shard read-only;
2. copy BF16/F32 tensors or dequantize one FP8/scale pair;
3. attach the resulting array to the already constructed module;
4. force `mx.eval` so the result no longer depends on a lazy source view;
5. release temporary CPU/MLX arrays; and
6. unmap the shard as soon as all of its resident tensors are loaded.

No converted resident artifact is written. The projected resident representation is exactly 17,842,600,184 bytes (16.617216 GiB), versus 15,225,993,464 source bytes (14.180311 GiB). This includes the separate 1,210 MiB embedding and 1,210 MiB language head.

## 7. Minimum correct graph for the first token

### mHC

mHC cannot be approximated or removed. Hidden state has shape `[batch, sequence, 4, 4096]`. Each attention and FFN site must reproduce the reference mapping:

- unweighted RMS normalization of the flattened four streams in FP32;
- learned projection to pre, post, and 4-by-4 combine terms;
- sigmoid pre weights plus `hc_eps`;
- twice-sigmoid post weights;
- row softmax followed by the configured 20 Sinkhorn iterations for the combine matrix;
- weighted stream collapse before the sublayer; and
- post placement plus transposed combine-matrix residual update.

The final HyperHead is an unweighted mean over the four streams, followed by the final RMSNorm.

### KDA: recurrent-only initially

The official reference contains chunk-prefill and recurrent decode algorithms. V1 uses the recurrent algorithm for every prompt token, starting from zero state. This is slower but removes a large independent kernel from the first-token critical path.

For each of the 34 KDA layers, cache:

- the last three samples needed by each q/k/v depthwise convolution; and
- a FP32 recurrent matrix with shape `[1, 64, 128, 128]`.

At each token:

1. project q, k, and v;
2. update the three causal convolution states and apply SiLU;
3. calculate the forget gate and beta gate;
4. L2-normalize q and k in FP32;
5. apply the recurrent Kimi Delta Attention update in FP32; and
6. apply the gated RMSNorm and output projection.

The recurrent state update must be differential-tested after every token, not merely at final logits. Chunk KDA is deferred until after first-token qualification.

### DSA: exact short-context dense equivalence

For batch size one, no padding, token-at-a-time execution, and a history no longer than `index_topk`, every complete four-token pool is selected and the visible tail is appended. The selected key set is therefore all causal keys. In that restricted domain, the official indexed mask is exactly dense causal attention.

V1 may exploit that equivalence only under these enforced conditions:

- batch size exactly one;
- no padding;
- token-at-a-time forward;
- prompt/cache length no greater than the hard v1 cap of 128; and
- a passing mask-equivalence test at lengths around pool boundaries.

The model must still implement the real DSA/MLA projections, expanded K/V cache, FP32 softmax, and output projection. Only indexer scoring and sparse-mask construction are bypassed. Indexer weights may be loaded to preserve the tensor contract but are semantically inactive under the cap.

Tests must compare the official indexer-produced mask with the dense causal mask for lengths `1, 3, 4, 5, 63, 64, 127, 128` using the full config, and corresponding in-range lengths using the tiny config. Any mismatch blocks the shortcut and requires implementing the full indexer before the full-model run.

Longer contexts must fail with a clear error. The real k-pool indexer and sparse attention are required before lifting the cap.

### Dense and shared MLPs

All dense and shared MLPs use limited SwiGLU:

```text
gate = min(gate, 10)
up   = clip(up, -10, 10)
out  = down_proj(silu(gate) * up)
```

The gate has no lower clamp. This detail is shared with the routed experts and must have a dedicated test.

### Router and routed reduction

Routing is calculated in FP32:

1. `router_logits = linear(hidden, gate.weight)`;
2. `scores = sigmoid(router_logits)`;
3. add `e_score_correction_bias` only for expert choice;
4. apply the group mask (a no-op for the present `n_group=topk_group=1`, but keep the validated general formula);
5. select eight experts;
6. gather the original sigmoid scores, not the bias-adjusted choice scores;
7. normalize by their sum plus `1e-20`; and
8. multiply by the routed scale 2.5.

The eight routed outputs are multiplied by those weights and reduced, then the resident shared-expert output is added. Expert ordering and tie behavior must be deterministic. Integer expert IDs and normalized route weights are part of the diagnostic trace.

### First-token ExpertSSD cache

Each of the 42 MoE layers owns an independent LRU with capacity eight. Since v1 processes one token at a time and routes exactly eight experts, one layer's working set always fits. A general wide-prefill/chunked path is not needed for the first milestone.

On a miss:

1. perform the one exact expert read;
2. split its six tensor views by the source plan;
3. construct immutable MLX U32/U8 arrays;
4. force evaluation before the source bytes leave scope; and
5. insert or evict through the per-layer LRU.

For compute, stack or gather only the eight selected cached experts for the current layer and invoke stock MLX `gather_qmm`. No 288-row expert allocation may exist, even transiently. Add an evaluation barrier at a conservative layer or token boundary during bring-up so MLX's lazy graph cannot retain evicted expert arrays.

## 8. Memory and I/O budget

### Memory

The `first-token-safe` projection is:

| Component | Projected allocation |
| --- | ---: |
| BF16/F32 resident arrays | 16.617216 GiB |
| 8 expert records x 42 layers | 4.183594 GiB |
| KDA FP32 recurrent state | 136 MiB |
| KDA convolution state | approximately 5-7 MiB |
| DSA expanded K/V cache at 128 tokens | 88 MiB |
| DSA auxiliary/index state at 128 tokens | less than 1 MiB |
| MLX inactive cache limit | 512 MiB initially |
| Activations, tokenizer, Python, and conversion temporaries | measured, expected below 2-3 GiB |

Expected steady active memory is roughly 21-24 GiB. The first full run has these stop gates:

- active MLX memory no higher than 24 GiB after resident load and a full eight-slot warm cache;
- process peak RSS no higher than 30 GiB on the 48 GiB host;
- no monotonic growth across two identical warm forwards; and
- at least 8 GiB of system headroom after steady state.

These are engineering gates, not promises derived only from arithmetic. If measurements exceed them, stop and find the retained allocation before increasing limits. The 256 GB Studio should still run this exact conservative profile first.

For comparison, larger future capacities would cost:

| Capacity per MoE layer | Expert-slot memory |
| ---: | ---: |
| 8 | 4.183594 GiB |
| 16 | 8.367188 GiB |
| 24 | 12.550781 GiB |
| 32 | 16.734375 GiB |
| 40 | 20.917969 GiB |

Capacity is a post-correctness tuning variable. Do not assume that the 256 GB machine should cache all experts; all 42 main banks would still consume 150.609375 GiB and would defeat the purpose of the SSD design.

### I/O

One cold token can request at most:

```text
42 layers x 8 experts x 13,369,344 bytes
= 4,492,099,584 bytes
= 4.183594 GiB
```

This baseline is expected to be slow. A T-token cold prompt has a pessimistic `T x 4.183594 GiB` routed-read ceiling before cache hits. The objective is observable correctness, not tokens per second.

The first-token report must include:

- prompt token count;
- cold and warm wall time;
- expert hits, misses, evictions, logical reads, system read calls, and bytes;
- per-layer unique routes;
- resident load bytes and time;
- active/peak MLX memory and process RSS; and
- output token ID and decoded text.

## 9. Implementation work packages and hard gates

Work proceeds in this order. A failed gate is debugged at its own layer; later work does not paper over it.

### M0. Reproducible environment and provenance

Deliverables:

- a dedicated GLM virtual environment rather than reuse of the DeepSeek environment;
- pinned stock MLX 0.32.0 runtime dependencies;
- a separate reference/test extra containing the pinned Transformers commit and CPU PyTorch requirements;
- a machine-readable source identity report containing config, index, composite-manifest, validation, and dependency identities; and
- a startup preflight for available memory and disk.

Gates:

- runtime imports without `mlx-io-glm` or PyTorch;
- reference tools are optional and isolated;
- the config declares `Glm5NextForConditionalGeneration` and `glm5_next_text`; and
- model validation status and pinned source revisions match this document.

### M1. Metadata-only target and source plans

Implement the model contract, resident source plan, and expert source plan without loading tensor payloads.

Gates:

- exactly 74,001 v1 tensors classified;
- exactly 2,107 allowed ignored tensors classified as vision or MTP;
- 1,425 main non-routed tensors, including exactly 179 FP8/scale pairs;
- 42 main MoE layers, 288 experts each, and six tensors per expert;
- all 12,096 v1 expert records resolve to one shard and one exact 13,369,344-byte range;
- no source range exceeds its file; and
- planned useful bytes equal payload bytes with no enclosing-span amplification.

This milestone should run in seconds and is the first defense against an incompatible checkpoint.

### M2. PyTorch reference fixtures

Build a small reference harness around the pinned Transformers source and tiny checkpoint. Because checkpoint-native mHC, separate convolution, and per-expert layouts may not match generated module names directly, create an explicit test-only state adapter and audit its mapping.

Capture deterministic fixtures for:

- RMSNorm and limited SwiGLU;
- mHC pre/post/combine tensors and stream updates;
- one-token KDA outputs, convolution states, and recurrent states over several tokens;
- DSA/MLA projections and short-context mask equivalence;
- router logits, selected IDs, and normalized weights;
- dense MLP, shared expert, and routed expert reductions;
- one decoder layer of each attention/MLP combination; and
- tiny-model per-layer hidden states and final logits for fixed token IDs.

Gates:

- every tiny checkpoint tensor is mapped once or explicitly classified as out of the text-only target;
- integer IDs and masks match exactly;
- the E4M3 codec matches all 256 reference byte values;
- FP32 component comparisons meet `rtol=2e-4, atol=2e-4` unless a documented operation requires a wider bound;
- BF16 component comparisons have cosine similarity at least 0.999 and `rtol/atol` no worse than 2e-2; and
- the tiny end-to-end greedy token ID matches exactly.

Thresholds may be tightened after observing the first fixture. They may not be loosened merely to get the full model running; any exception requires a named, understood numerical source.

### M3. Pure-MLX text graph with mandatory ExpertSSD

Implement the tiny graph with the same mandatory ExpertSSD seam as the full
runtime. Tests may use an in-memory synthetic reader behind that seam, but no
production graph may construct an ordinary resident routed-expert bank. This
keeps architecture tests small without creating a fallback that could silently
defeat the deployment design. Native MXFP4 is the only routed-expert format in
v1; ScaleX is not involved.

Gates:

- batch-one tokenwise execution passes all M2 fixtures;
- KDA state matches after each token;
- DSA dense equivalence passes all boundary lengths;
- dense layers 0-2 and MoE layers 3-4 are selected from config rather than hard-coded test topology;
- every sparse layer rejects a non-ExpertSSD backend;
- no vision or MTP object is constructed; and
- a fixed tiny prompt produces the reference greedy token.

Only after this gate should the large checkpoint be touched by the runtime.

### M4. Full resident loader

Attach the metadata plans and streamed resident loader to the full model graph, still without executing a routed layer.

Gates:

- exactly 1,246 runtime resident arrays are produced from 1,425 source tensors;
- every one of the 179 FP8 scale arrays is consumed exactly once;
- all resident module parameters are filled and no unexpected placeholder remains;
- no expert payload is read;
- projected and measured resident bytes agree;
- active MLX memory stays at or below the resident projection plus a small measured tolerance; and
- all mmaps and temporary conversion arrays are released.

Run isolated projection probes against BF16-source tiles before advancing.

### M5. Functional ExpertSSD in isolation

Implement the source reader, per-layer LRU, and routed SwitchGLU independently of the complete model.

Test real experts from early, middle, and late main layers, including single-expert, eight-expert, cache-hit, eviction, and injected-short-read cases.

Gates:

- the loaded six arrays are byte-identical to their checkpoint ranges;
- each miss reads exactly 13,369,344 useful bytes from one logical range;
- MXFP4 matmul matches the existing single-projection preflight behavior;
- limited SwiGLU and weighted top-eight reduction match the reference implementation;
- a repeated warm call issues zero new reads and returns the same output; and
- capacity never exceeds eight in the first-token profile.

### M6. Full first-token qualification

Run progressively:

1. one explicit token ID through all 45 main layers;
2. a short bare text prompt;
3. a text-only message rendered with the official chat template using `clear_thinking=true`; and
4. the same canonical chat prompt twice, once cold and once warm.

The canonical command is:

```bash
./run.sh --memory 24 --max-tokens 1 \
  "Reply briefly: what is the capital of France?"
```

The first-token gate passes only when:

- all prior milestone reports pass;
- the output logits have shape `(1, 1, 154880)` and are finite;
- the greedy token ID is in range and decodes through the official tokenizer;
- cold and warm runs select the same token and the same per-layer routes;
- warm execution performs no expert reads if the canonical prompt's working set remains cached, or any remaining reads are explained by the eight-slot capacity trace;
- no vision or layer-45 range is accessed;
- no 288-expert array exists;
- memory and graph-retention gates pass; and
- `reports/v1-first-token.json` records all provenance, output, memory, routing, and I/O metrics.

Do not make a semantic-quality claim from this gate. A particular word such as `Paris` is useful evidence but is not a hard acceptance criterion for a mixed-quantization custom checkpoint. The hard claim is that the intended graph and weights execute deterministically enough to emit a token.

### M7. Immediate post-token correctness

Before optimization, extend the same baseline to five greedy tokens and verify cache progression, stop-token handling, and repeated cold/warm token identity. Then implement and qualify the real DSA indexer before allowing context beyond 128.

Only after M7 should performance profiling drive native or memory-layout changes.

## 10. Proposed code organization

The implementation should remain small and keep downloader concerns separate from runtime concerns:

```text
src/glm53flash/
  contract.py             # config/index/header classification and invariants
  resident_plan.py        # source-to-runtime names, layouts, and byte budgets
  resident_reader.py      # streamed copies and block-FP8 -> BF16
  fp8.py                  # E4M3FN codec and 128x128 block dequantization
  expert_source.py        # exact expert ranges and MLX views
  expert_reader.py        # synchronous pread reader and metrics
  expert_ssd.py           # bounded per-layer LRU and MXFP4 routed compute
  model_config.py         # normalized text-only config
  mhc.py                  # mHC mapping and HyperHead
  kda.py                  # recurrent KDA and convolution cache
  dsa.py                  # MLA plus guarded short-context dense path
  moe.py                  # router, shared expert, ExpertSSD seam
  model.py                # 45-layer target-only language graph
  cache.py                # explicit KDA and DSA cache structures
  tokenizer.py            # official tokenizer/chat-template wrapper
  runtime.py              # source validation, load sequence, generation
  chat_cli.py             # narrow smoke entry point

tests/
  test_contract.py
  test_resident_plan.py
  test_fp8.py
  test_expert_source.py
  test_expert_reader.py
  test_expert_ssd.py
  test_mhc_reference.py
  test_kda_reference.py
  test_dsa_reference.py
  test_router_reference.py
  test_tiny_model_reference.py
  test_full_smoke.py       # opt-in, large-model marker

scripts/
  reference_fixtures.sh
  smoke.sh
```

Exact filenames can change if a simpler organization emerges, but the seams should remain separate. In particular, model code must not parse safetensors offsets, and the reader must not know routing policy.

## 11. Diagnostics and failure localization

The first implementation should favor observability over speed. A `--trace-layer N` mode should be able to report, without dumping proprietary tensor payloads:

- input/output shape, dtype, finite status, norm, min, and max;
- mHC pre/post/combine summaries;
- attention kind and cache shapes;
- KDA recurrent-state norm or DSA KV length;
- router top-eight IDs and weights;
- expert hits/misses and bytes; and
- layer output checksum or a small deterministic numerical fingerprint.

On a non-finite value or parity failure, abort at the earliest layer. Do not continue generation with NaNs, missing experts, a truncated read, or an unfilled parameter.

Reports should be JSON and include:

- git commit and dirty status of this repository;
- model source revisions and manifest hashes;
- MLX, Python, NumPy, tokenizer, and reference dependency versions;
- hardware model, unified memory, and OS version;
- command/profile and prompt-token IDs;
- memory counters at startup, post-resident-load, post-cold-forward, and post-warm-forward; and
- route/cache/I/O counters by layer and in total.

## 12. Risk register and controlled fallbacks

| Risk | Detection | Required response |
| --- | --- | --- |
| Very new or incomplete upstream GLM graph | Tiny differential tests fail or reference loader cannot map weights | Fix the explicit test adapter and compare mathematical components; do not copy a serving implementation blindly |
| Checkpoint/runtime name mismatch | Mapping is not bijective | Stop in M1/M2; no large allocation |
| Incorrect E4M3 interpretation or scale direction | Exhaustive codec or BF16 tile test fails | Stop before resident load; validate against PyTorch/compressed-tensors semantics |
| Incorrect mHC stream axis or Sinkhorn order | mHC fixture diverges | Fix at component level; do not loosen final-logit thresholds |
| KDA recurrence or convolution-cache error | Per-token state fixture diverges | Fix recurrent path before any chunk-prefill work |
| DSA dense shortcut is not equivalent | Mask boundary test fails | Implement the actual indexer before full-model smoke |
| Wrong MXFP4 view or projection order | Expert projection/SwitchGLU tests fail | Stop in M5; compare exact source offsets and preflight code |
| Lazy MLX graph retains evicted experts | Active memory grows across layers/tokens | Add explicit evaluation/materialization boundaries and inspect ownership |
| 48 GiB host pressure | RSS/active-memory gate fails or system starts swapping | Abort, clear MLX cache, reduce unrelated workload; do not silently use swap as capacity |
| FP8-to-BF16 startup is too slow | Resident timing report | Accept for first token; optimize only after correctness, without writing another checkpoint |
| ExpertSSD is too slow | I/O metrics show expected one-range misses | Accept for first token; native direct-to-slot and overlap are later stages |
| Output is finite but implausible | Tiny parity passes but full output is incoherent | Compare early/middle/late layer fingerprints, routes, and resident dequant tiles; seek an external full-model oracle before claiming quality |
| Native fork changes token output | Stock/native A-B token trace differs | Stock path remains the oracle; native path is rejected until identical |

If resident BF16 memory unexpectedly does not fit the development host, the first fallback is to run the same baseline on the 256 GB Studio. Requantizing resident weights, paging the embedding/head, or adding a block-FP8 Metal matmul are optimizations and must not be smuggled into the correctness milestone.

## 13. Explicit v1 non-goals

- No vision input or vision tower.
- No MTP/NextN layer 45, speculative decoding, or drafter.
- No contexts above 128 until real DSA is implemented and tested.
- No batched requests or padding.
- No chunk-prefill KDA kernel.
- No async I/O, prefetching, route prediction, warm-start policy, or Metal shared events.
- No native `mlx-io-glm` dependency.
- No converted expert slabs or resident checkpoint.
- No in-memory conversion of routed experts to another quantization.
- No performance/TPS claim from the first-token milestone.
- No modifications to the DeepSeek repository.

## 14. Optimization order after the baseline

Once M7 passes, optimize in an output-preserving order and compare every change against the stock baseline's token IDs, route trace, and logits fingerprint:

1. Profile KDA, resident matrix multiplies, routed compute, Python synchronization, and SSD wait separately.
2. Implement the real DSA indexer and lift the context cap gradually.
3. Add chunked KDA prefill while retaining recurrent decode, with state parity at chunk boundaries.
4. Tune per-layer cache capacity on the actual 256 GB Studio from measured hit/miss traces; do not assume uniform capacity is optimal.
5. Initialize `mlx-io-glm` from a pinned MLX 0.32.0-compatible base and add an explicit symbol/version handshake.
6. Replace immutable Python miss arrays with reusable native expert slots.
7. Read the single contiguous GLM expert record directly into its six final slot views.
8. Parallelize independent misses and add a native completion primitive before Metal consumes a slot.
9. Overlap routed misses with the resident shared expert and other independent work.
10. Consider resident block-FP8 execution, in-memory resident requantization, or embedding/head paging only with a separate numerical and token-identity qualification.
11. Add longer-generation memory/endurance tests before publishing throughput.

The native fork must be installed through an explicit isolated overlay, as in the DeepSeek project. It must never silently replace the stock MLX environment, and the stock synchronous reader must remain available as a diagnostic fallback.

## 15. Definition of done for v1

V1 is complete when all of the following are true:

- the model source and 74,001-tensor target contract pass;
- the tiny BF16 graph matches the pinned reference at component and greedy-token level;
- resident block-FP8 decoding is independently validated;
- the full model constructs without vision, MTP, or stock routed-expert allocations;
- a batch-one prompt of at most 128 tokens traverses all 45 main layers;
- routed experts are read from the existing composite through one exact range per miss;
- the runtime emits and decodes one finite greedy token;
- a repeat run is deterministic at token and route level;
- active and peak memory remain inside the 48 GiB safety gates;
- no duplicate model artifact is created; and
- the provenance, memory, route, I/O, and output report is saved.

Anything beyond that—more tokens, longer context, lower latency, a larger cache, native I/O, or MTP—is valuable follow-on work, but it is not allowed to obscure whether the first GLM token came from the intended graph and checkpoint.

## 16. Source links

- Official model: <https://huggingface.co/zai-org/GLM-5.3-Flash>
- Official Z.ai project: <https://github.com/zai-org/GLM-5>
- Pinned Transformers graph source: <https://github.com/huggingface/transformers/blob/69a7fb1aca7f2e7487294846be5859ebb6db9462/src/transformers/models/glm5_next/modeling_glm5_next.py>
- Tiny architecture fixture: <https://huggingface.co/inference-optimization/GLM-5.3-Flash-0.1B-A0.1B/tree/8311399447eba9c9b215e3209ab6f25e59c7d21e>
- MLX: <https://github.com/ml-explore/mlx/tree/v0.32.0>
- LivGLM: <https://github.com/kg36/livglm>
- Future native fork: <https://github.com/kg36/mlx-io-glm>
