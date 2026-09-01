# Perfetto and Metal trace analysis

Date: 2026-09-01

## Capture contract

The profiling path records only selected decode forwards. Model loading and
prefill are outside the Perfetto window. The application trace contains:

- the Python decode-step and model-forward envelopes;
- all 45 transformer layers, their attention/FFN stages, and all 42 MoE
  envelopes;
- router materialization barriers, cache slot planning, read issue, eight SSD
  worker lanes, read joins, and SSD-to-MoE flow arrows;
- MLX graph-construction spans and the final GPU synchronization;
- per-token Metal primitive/dispatch counters; and
- actual GPU-active intervals imported from a clock-correlated Instruments
  Metal System Trace.

The important semantic boundary is explicit in the trace: an `mlx_submit`
span is host-side graph construction or submission, not GPU execution.
`materialize_route` and `final_logits_eval_wait` are host waits for GPU work.
Only events on the `Metal GPU compute` lanes are actual GPU-active intervals.

The primary control command was:

```bash
./run.sh --memory 30 --max-tokens 25 \
  --trace artifacts/perfetto/glm53flash-25-token-perfetto-only.json \
  --trace-decode-steps 24 \
  "who are you"
```

It captured 24 decode forwards, 25,563 events, and produced the same 25-token
text and exact expert accounting as the paired Instruments run. Generated
captures live under `artifacts/perfetto/` and are deliberately ignored by Git.

## Observer effect

| Run | Decode rate | Purpose |
| --- | ---: | --- |
| User untraced baseline | 3.792 token/s | Production reference |
| Perfetto application trace only | 3.442 token/s | Primary timing analysis |
| Perfetto + Metal + filesystem + time profiler | 2.341 token/s | Structural GPU/I/O overlap evidence |

The application-only trace is about 9% below the user baseline. The full
Instruments probe set is deliberately invasive and its absolute token time
must not be used as a production estimate.

## Application trace result

The 24-forward control averaged 290.18 ms per token (3.446 token/s inside the
trace window):

| Critical-path component | Mean/token | Share of wall |
| --- | ---: | ---: |
| Route materialization / upstream GPU wait | 120.24 ms | 41.4% |
| Unhidden SSD join | 147.40 ms | 50.8% |
| Final logits GPU wait | 4.35 ms | 1.5% |
| Slot planning plus read issue | 2.04 ms | 0.7% |
| Remaining host/trace overhead | 16.15 ms | 5.6% |

The production numbers independently support the same decomposition. The
3.792 token/s run took about 264 ms/token, with approximately 104 ms/token of
decode route synchronization and 143 ms/token of decode SSD wait.

Each traced token had, on average:

- 136.9 expert misses and direct SSD reads;
- 1.705 GiB of logical expert payload;
- 12.28 GB/s active logical read bandwidth;
- 16,947 Metal dispatches and 29,696 primitive evaluations;
- 497 resident quantized matmuls, 123 gather-QMM evaluations, and 237 fused
  RMSNorm evaluations; and
- 45 transformer-layer and 42 MoE-layer envelopes.

The SSD worker read distribution was 4.38 ms p50, 6.94 ms p95, and 8.43 ms
maximum. Token wall time correlated 0.991 with SSD join time and 0.973 with
expert-miss count. SSD locality is therefore the main source of token-to-token
latency variation.

## Actual GPU and physical SSD activity

Instruments initialization consumed most of its 20-second recording, but the
trace contains complete, clock-correlated Metal and physical-disk coverage for
decode steps 0 through 4. Across those five invasive forwards:

| Activity | Active union | Share of observed wall |
| --- | ---: | ---: |
| Metal GPU compute | 737.77 ms | 34.5% |
| Physical data reads | 763.20 ms | 35.7% |
| GPU and physical reads active simultaneously | 30.15 ms | **1.4%** |

The Metal export contains 2,273 target-process compute intervals; 2,199 fall
inside the five complete forwards imported into Perfetto on two overlap lanes.
The physical-I/O export contains 19,459 data-read routines and 8.18 GiB in the
same window, with 805.24 ms total physical-read activity and 10.90 GB/s active
throughput. Kernel I/O routines were mostly 512 KiB, with 2.06 ms p50, 5.26 ms
p95, and 10.79 ms maximum latency.

The time profiler sampled about 1.03 running CPU cores on average. It charged
617 sampled milliseconds to `preadv` across the I/O workers, about 385 sampled
milliseconds to the main thread, and only 22 sampled milliseconds directly to
Python frame evaluation. Python bytecode is not the primary critical-path
consumer; it mostly coordinates GPU dependencies and SSD joins.

## Layer locality

The uniform 33-slot allocation is not equally effective. Layers 3 through 10
accounted for 1,025 of 3,286 misses (31.2%) even though they are only 8 of the
42 routed layers.

| Layer | Hit rate | Misses/24 tokens | Mean layer time |
| ---: | ---: | ---: | ---: |
| 3 | 27.6% | 139 | 14.39 ms |
| 4 | 21.9% | 150 | 9.69 ms |
| 5 | 27.6% | 139 | 8.97 ms |
| 6 | 35.4% | 124 | 8.45 ms |
| 8 | 39.1% | 117 | 8.29 ms |
| 9 | 28.1% | 138 | 9.54 ms |
| 10 | 30.2% | 134 | 9.16 ms |
| 18 | 82.3% | 34 | 4.63 ms |
| 19 | 78.1% | 42 | 4.49 ms |

Layer 3 is special: its 7.80 ms route barrier is more than twice the typical
routed-layer barrier, in addition to its low cache hit rate.

## Conclusion and next phase

The trace rejects the idea that SSD bandwidth itself is the remaining sole
bottleneck. It is already near 12 GB/s. The larger problem is scheduling:
GPU work and SSD reads overlap for only 1.4% of the observed wall interval.
The runtime is effectively paying the route/resident GPU phase and the SSD
phase serially.

The next implementation should proceed in this order:

1. Split every missed expert refill into gate/up and down stages. Read gate/up,
   enqueue the MXFP4 gate/up plus SwiGLU activation, and read the down projection
   concurrently before the down QMV consumes it. This requires the
   `mlx-io-glm` fork because the current native call fills the complete expert
   record in one operation.
2. Verify with the same Perfetto events that real GPU/disk overlap rises while
   token IDs remain exact. A useful first target is 40–50 ms of hidden SSD time
   per token.
3. Capture a longer route tape and simulate non-uniform per-layer cache
   capacities under the same hard 30 GiB ceiling. The early routed layers are
   the obvious candidates for extra slots; later high-hit layers can fund them.
4. Only after I/O staging and cache redistribution, revisit the stable
   approximately 120 ms route/resident GPU component and its 16,947 dispatches
   per token.

Using the untraced 264 ms/token baseline, hiding half of the approximately
104 ms GPU phase under SSD work gives about 212 ms/token, or 4.7 token/s.
Perfect overlap would put the arithmetic floor near
`max(104 ms, 143 ms) + 17 ms = 160 ms`, about 6.2 token/s. This is an upper
bound, not a forecast, but it brackets the value of the next phase.

## Capture hygiene

Raw Instruments `.trace` bundles embed the attached process environment.
They must stay local and must never be committed or uploaded without review.
The analysis uses sanitized exported GPU, disk-I/O, and time-profile tables;
Perfetto JSON contains model/runtime metadata but no copied process environment.
