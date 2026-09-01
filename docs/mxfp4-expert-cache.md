# MXFP4 resident / larger ExpertSSD-cache experiment

Date: 2026-09-01

## Result

`--resident-mxfp4` converts the same 497 resident linear matrices as the
default MXFP8 startup path to native group-32 MXFP4. It does not rewrite the
official Z.ai checkpoint, alter the downloaded native MXFP4 routed experts, or
use ScaleX. The saved resident memory is assigned to the exact ExpertSSD slot
banks under the same hard `--memory` plan.

At `--memory 30`, the option changes the plan from 9.30 GiB resident plus 33
slots/layer to 5.52 GiB resident plus 41 slots/layer:

| Metric | MXFP8 default | MXFP4 resident |
| --- | ---: | ---: |
| Resident graph | 9.295 GiB | 5.516 GiB |
| ExpertSSD slots/layer | 33 | 41 |
| Expert cache | 17.257 GiB | 21.441 GiB |
| Planned memory | 29.553 GiB | 29.957 GiB |
| Measured MLX peak | 26.77 GiB | 27.19 GiB |

The 30-token control was:

```bash
./run.sh --memory 30 --resident-mxfp4 --max-tokens 30 "who are you"
```

It generated the expected continuation and measured 3.905 decode token/s,
62.4% decode cache hits, 45.609 GiB decode expert I/O, 4.027 seconds of decode
SSD wait, and 2.900 seconds of decode route synchronization across 29 decode
forwards. A matched 20-token development run measured 3.870 token/s, 64.7%
decode hits, 28.052 GiB decode I/O, and a 27.18 GiB MLX peak. The independent
user confirmation reached **4.180 token/s** with the same text, routes,
64.7% decode hit rate, and 28.052 GiB decode I/O. Its earlier MXFP8 baseline
was 3.792 token/s, making the paired user-visible improvement 10.2%.

This is a useful approximately 10% local gain, not the 5 token/s milestone.
The option is not the default because a matching short continuation is not a
broad quality qualification and MXFP4 has a wider numerical error bound than
MXFP8.

## Rejected overlap variants

All variants retained exact native ExpertSSD demand reads and produced the
same short continuation, but none earned a production path:

| Variant | Decode token/s | Verdict |
| --- | ---: | --- |
| Full expert, serial control | 3.548 | Paired control |
| Gate/up then down staging | 2.377 | Regressed from extra reads and graph fragmentation |
| Metal-event staged graph | 2.280 | Regressed |
| Resident-hit work during reads | 3.534 | Neutral |
| Python completion-order streaming | 3.515 vs 3.494 control | Noise-sized |
| Per-expert Metal streams | 3.590 vs 3.525 control | Small/noisy; not retained |
| Batched gate/up under down reads, MXFP4 resident | 3.450 vs 3.870 control | Regressed |

The dependency is the key constraint: the real route is unavailable until the
upstream layer graph completes, and the next layer cannot start until all
current routed contributions are merged. Current-layer shared/resident expert
work is too small to hide enough of each demand-read tail.

## Predictive-prefetch boundary

A causal replay of the 24-token route tape tested expert-transition,
frequency, and same-position predictors. The best simple top-4 transition
predictor advised 43.4 GiB to cover only 0.3 GiB of real next-token misses:
0.6% precision and 0.7% miss recall. It must not be enabled.

The checkpoint does contain one MTP layer at
`model.language_model.layers.45` (`num_nextn_predict_layers: 1`), including
`enorm`, `hnorm`, `eh_proj`, attention, a routed MoE, and `shared_head` norm.
That makes an MTP-derived, non-evicting `F_RDADVISE` predictor the credible
next research phase. It must first measure held-out route precision and advice
bandwidth; the MTP layer is not itself an all-42-layer route oracle.

A 10-token diagnostic tested the cheaper MTP input anchor only: normalize the
known next-token embedding and previous target hidden state, apply the official
`eh_proj`, then evaluate all 42 target routers on that anchor. Route recall was
19.9%, but the important cache-miss result was only 10.2% recall at 5.9%
advice precision. It would have advised 25.5 GiB to warm 1.51 GiB that was
actually demanded. This anchor-only form also fails the bandwidth gate and was
not added to the runtime.
