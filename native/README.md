# Native mlx-io-glm dependency

LivGLM pins [`kg36/mlx-io-glm`](https://github.com/kg36/mlx-io-glm) at
`7588c17195819353035bfc28c1226e8404687343`. That revision descends from the
native MLX data plane qualified by LivSeek's DeepSeek runtime and provides the
generic APIs used here:

- cached expert tensor offsets and one-range `preadv`;
- direct writes into evaluated, row-contiguous MLX arrays;
- parallel calls that release the Python GIL; and
- best-effort slot wiring.

LivGLM uses only native MXFP4 direct-to-slot loading in this phase. ScaleX,
speculative decoding, and DeepSeek-specific kernels exposed by the inherited
fork are not used.

`scripts/bootstrap_native_mlx.sh` reconstructs the ignored checkout at
`native/mlx-io-glm/`. `scripts/build_native_mlx.sh` builds an isolated wheel
overlay at `native/testsite/`; it does not replace stock MLX in `.venv`.
