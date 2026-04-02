# ML Kernels Workspace

This directory is a staging area for ML-oriented kernel work that does not fit cleanly into the homework subprojects.

Suggested layout:

- `attention/`: attention, softmax, RoPE, and KV-cache kernels.
- `conv/`: convolution, im2col, and fused activation kernels.
- `include/ml_kernels/`: shared headers and small utilities.
- `src/`: standalone experiments and smoke tests.

Build the starter target from the repository root:

```bash
mkdir -p build && cd build
cmake ..
make ml_kernel_smoke -j"$(nproc)"
./ml_kernels/ml_kernel_smoke
```

Keep benchmark drivers close to the kernel they measure, and prefer one kernel family per file.
