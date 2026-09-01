# Native ExpertSSD status

Date: 2026-09-01

## Outcome

GLM-5.3-Flash now uses the same native ExpertSSD data-plane design as LivSeek:
fixed per-layer MXFP4 slot arrays, direct SSD reads into those arrays, parallel
I/O, native MXFP4 matmul, and a native Markov-LHD reuse policy. The official
Z.ai remainder remains resident. ScaleX is not used.

The implementation preserves the first-token correctness oracle. The 20-token
qualification prompt still begins with the same text (`Hi! I'm Claude...`),
and a real layer-3 routed expert produces bit-exact output against stock MLX
`gather_qmm`.

## Frozen native dependency

The build bootstraps `kg36/mlx-io-glm` from the pinned revision recorded in
`reference-lock.json`:

```text
7588c17195819353035bfc28c1226e8404687343
```

That revision is the already-qualified LivSeek native implementation, mirrored
from `kg36/mlx-io-ds`. Its direct-to-slot APIs are model-independent, so no
GLM-specific C++ changes were required. `build.sh` creates an isolated wheel
overlay under `native/testsite`; it does not silently replace the normal MLX
installation in `.venv`.

## Hard memory budget

`--memory` means the total MLX allocation ceiling. Capacity is derived only
after accounting for the resident graph and a fixed 3 GiB execution reserve:

```text
capacity = floor(
  (memory ceiling - resident bytes - 3 GiB)
  / (42 routed layers * 13,369,344 bytes per expert slot)
)
```

For `--memory 30`, the resolved plan is:

| Component | GiB |
| --- | ---: |
| Resident graph | 16.617216 |
| 19 slots/layer across 42 layers | 9.936035 |
| Runtime reserve | 3.000000 |
| Planned total | 29.553251 |
| Arithmetic headroom | 0.446749 |

MLX is configured with the 30 GiB memory limit. A full measured run peaked at
29.18 GiB; the allocation-only load used 26.55 GiB after its fixed slots were
created and wired.

## Native path

For every routed layer, the runtime:

1. opens one native shard-range source for the layer's 288 experts;
2. allocates a fixed bank of evaluated, wired arrays for all six group-32
   MXFP4 tensors in every cache slot;
3. asks the native Markov-LHD policy to map the current top-8 route to retained
   or replacement slots;
4. reads misses in parallel directly from safetensors into their final slots;
5. overlaps those reads with the resident shared-expert calculation; and
6. executes routed gate/up/down projections with the native MXFP4 kernels.

The Python `pread` reader remains useful as a test oracle, but the production
runtime refuses to fall back to it.

## Qualification

The unit suite contains 29 tests. In addition to graph and contract coverage,
the native tests prove:

- direct-to-slot bytes are exactly equal to the Python range reader on a
  synthetic record;
- a real layer-3/expert-0 record is copied byte-exactly into its native slot;
- the native real-geometry layer-3 top-8 MXFP4 calculation is bit-exact with
  stock MLX `gather_qmm`; and
- `--memory 30` resolves to exactly 19 slots/layer and the totals above.

## Measured 30 GiB baseline

Command:

```bash
./run.sh --memory 30 --max-tokens 20 "who are you"
```

The native Markov-LHD run measured:

| Metric | Result |
| --- | ---: |
| Prefill | 1.010 token/s |
| Decode | 1.250 token/s |
| Total expert I/O | 76.475 GiB |
| Overall cache hit rate | 47.8% |
| Decode cache hit rate | 51.6% |
| Decode SSD wait | 3.198 s |
| Decode visible SSD bandwidth | 12.92 GB/s |
| Decode route/graph synchronization | 9.852 s |
| MLX peak | 29.18 GiB |

Against the native LRU baseline, Markov-LHD avoided 336 expert reads, reduced
traffic by 4.18 GiB, and raised decode from 1.214 to 1.250 token/s.

## What the measurement says

The model's “A18B” active-parameter figure is not an SSD-byte count. Attention,
shared experts, dense layers, embeddings, and other non-routed weights are
already resident. The exact routed payload at a 100% miss rate is:

```text
42 layers * 8 experts * 13,369,344 bytes = 4.183594 GiB/token
```

At the measured 51.6% decode hit rate, the SSD supplies approximately 2.03
GiB/token. At 12.92 GB/s that accounts for about 0.168 seconds of visible wait
per decode forward, rather than the assumed 12 GiB and one second.

The SSD premise is validated: misses are delivered at roughly 12–13 GB/s.
However, this prompt reached a 51.6% decode hit rate rather than the assumed
60%, and SSD wait is no longer the dominant latency. Nineteen decode forwards
took about 15.2 seconds in total, of which 3.20 seconds were visible SSD wait
and 9.85 seconds were route/graph synchronization.

Therefore 2–3 token/s is not yet achievable by cache locality alone. The
measured baseline is 1.25 token/s, and the next performance phase should target
the resident compute graph—especially mHC/Sinkhorn, KDA, routing, and resident
FP8 execution—while keeping this ExpertSSD path as the fixed I/O baseline.
