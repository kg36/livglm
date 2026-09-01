# MXFP8 resident performance phase

Date: 2026-09-01

## Result

The default runtime now converts 497 active resident linear weights to native
group-32 MXFP8 during startup. Routed experts remain in their downloaded native
MXFP4 format on SSD. ScaleX is not used, and no converted checkpoint is written
to disk.

The following command exceeds the initial 3 token/s target on the 48 GiB
development Mac while respecting a hard 30 GiB MLX ceiling:

```bash
./run.sh --memory 30 --max-tokens 20 "who are you"
```

It produced the same 20-token decoded text as the BF16 baseline:

```text
Hi! I'm Claude, an AI assistant made by Anthropic. I'm here to help with
```

| Metric | BF16 user baseline | MXFP8 default | Change |
| --- | ---: | ---: | ---: |
| Prefill | 1.006 token/s | 2.704 token/s | 2.69x |
| Decode | 1.360 token/s | 3.525 token/s | 2.59x |
| End-to-end output | 0.669 token/s | 1.769 token/s | 2.64x |
| Decode cache hit rate | 51.6% | 60.8% | +9.2 points |
| Decode expert I/O | 38.487 GiB | 31.165 GiB | -19.0% |
| Decode SSD wait | 3.206 s | 2.715 s | -15.3% |
| Decode route rendezvous | 8.743 s | 2.315 s | -73.5% |
| MLX peak | 29.20 GiB | 26.77 GiB | -2.43 GiB |

A second fresh-process run produced the same text and identical aggregate
expert loads/hits/misses at 3.564 decode token/s, again peaking at 26.77 GiB.

The route rendezvous is the point where expert IDs become host-visible; it
includes unfinished resident GPU work upstream of the router. It is not router
kernel time. Its reduction is the clearest evidence that resident linear
traffic, rather than SSD throughput, was the principal bottleneck.

## Hard 30 GiB plan

The startup sequence deliberately avoids holding the larger expert cache at
the same time as the temporary BF16 resident graph:

1. construct ExpertSSD layer sources with slot allocation deferred;
2. load the official Z.ai resident remainder as the BF16 oracle graph;
3. quantize active `DeferredLinear` weights to group-32 MXFP8;
4. release allocator cache from replaced BF16 arrays; and
5. allocate and wire the final 33-slot ExpertSSD banks.

The resulting plan is:

| Component | GiB |
| --- | ---: |
| MXFP8 resident graph | 9.295302 |
| 33 slots/layer across 42 layers | 17.257324 |
| Runtime reserve | 3.000000 |
| Planned total | 29.552626 |
| Arithmetic headroom | 0.447374 |
| Measured active after load | 26.55 |
| Measured full-run peak | 26.77 |

The source conversion covers 16,230,907,904 bytes of resident BF16 linear
arrays and produces 8,369,061,888 bytes of MXFP8 weights plus E8M0 scales.
Dormant DSA-indexer weights, router weights, normalization parameters,
embeddings, and other non-linear arrays retain their existing representation.

## Precision boundary

MXFP8 is an in-memory runtime quantization of the official resident remainder;
it is not bit-identical to BF16. It was chosen over MXFP4 because a real
8192-by-4096 projection measured essentially the same kernel time for both,
while MXFP8 retained twice the weight precision. The qualification prompt kept
the same decoded output, but this is not a broad semantic-quality evaluation.

The original correctness path remains available:

```bash
./run.sh --resident-bf16 --memory 30 --max-tokens 1 \
  "Reply with exactly one word: hello"
```

That override restores the original 16.617 GiB resident graph, 19 slots/layer,
and produced `Hello` in the post-change verification run.

## Rejected experiments

- Fusing all 20 Sinkhorn iterations and stream collapse into 90 custom mHC
  kernels reduced Metal dispatches by 64% but regressed decode from the BF16
  baseline to 1.239 token/s and changed a few expert routes.
- Concatenating exact BF16 Q/K/V and gate/up matrices won isolated hot-cache
  microbenchmarks but regressed the matched full-model window from 801 ms to
  865 ms per forward.
- Fused RMSNorm was bit-identical in the tested BF16 geometry and improved a
  matched eight-forward window by about 2.1%; it is retained as a small,
  quality-preserving part of this phase.

These results are why the production change reduces resident storage traffic
without adding whole-graph compilation, aggressive kernel fusion, MXFP4
resident quantization, or ScaleX.
