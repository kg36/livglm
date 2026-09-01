# V1 implementation status

> Historical milestone: this documents the initial stock-MLX correctness
> path. The native direct-to-slot implementation that supersedes it is recorded
> in [`native-expertssd-status.md`](native-expertssd-status.md).

This checkpoint records the correctness-first implementation and its first
successful full-model token run.

## Implemented

- strict validation of the composite format, pinned source revisions, tensor
  inventory, safetensors headers, and supported 45-layer text profile;
- streamed official-remainder loading, including E4M3FN block-FP8 to BF16;
- exact one-range native MXFP4 expert records and a short-read-safe `pread`
  reader;
- mandatory eight-slot-per-layer ExpertSSD LRU and stock MLX `gather_qmm`;
- mHC, recurrent tokenwise KDA, real DSA/MLA projections with a guarded
  short-context dense-equivalent path, dense/shared MLPs, routing, and the
  45-layer text-only graph;
- official tokenizer/chat template, greedy generation, EOS handling,
  one-shot/chat CLI, memory checks, finite-value guards, and ExpertSSD stats;
- build and test scripts that do not launch the full model.

There is no routed resident-bank implementation, no ScaleX path, and no
dependency on `mlx-io-glm` in v1.

## Verified on the development Mac

- 25 unit tests pass under Python 3.13 and MLX 0.32.x, including the official
  Jinja chat-template path.
- Real metadata preflight resolves 1,425 resident source tensors to 1,246
  runtime arrays and 72,576 routed tensors to 12,096 exact expert records.
- Every resident destination name exists in the deferred 45-layer graph.
- A real layer-3 native expert read completed in one 13,369,344-byte system
  read and produced a finite `(1, 1, 4096)` BF16 output through stock MLX
  MXFP4 matmul.

## Full first-token result

After the machine became available, the following command was executed twice
from fresh processes:

```bash
./run.sh --memory 30 --max-tokens 1 \
  "Reply with exactly one word: hello"
```

The official template always opens a `<think>` section. The runtime now closes
that section with the official `</think>` token when `--thinking` is absent;
otherwise a nominally non-thinking one-token run merely exposed the first
reasoning token. Jinja2 is also an explicit runtime dependency rather than an
accidental transitive assumption.

With those fixes, two cold runs loaded the 17,842,600,184-byte resident graph,
traversed all 45 layers for 20 prompt tokens, and emitted the same requested
decoded first token: `Hello`. Both produced identical ExpertSSD trace totals:

- 6,720 routed lookups (`20 × 42 × 8`);
- 5,132 native expert loads/reads and 1,588 cache hits;
- 4,796 evictions; and
- 63.899 GiB of expert payload I/O.

The first-token command is therefore operational and cold-run deterministic at
the output and aggregate ExpertSSD-accounting level. The particular word is not
a semantic-quality qualification for this custom mixed-precision checkpoint.

## Still intentionally unqualified

- same-process warm-route determinism;
- repeated five-token direct-answer determinism and EOS behavior;
- peak active-memory/RSS measurements and saved per-layer trace artifacts; and
- contexts above 128, vision, MTP, ScaleX, or any optimized/native I/O path.

Normal first-token commands:

```bash
./build.sh
./run.sh --preflight
./run.sh --memory 24 --max-tokens 1 "Reply with exactly one word: hello"
```

Do not increase the 128-token context limit or eight-slot ExpertSSD capacity
until the first-token result and trace are understood.
