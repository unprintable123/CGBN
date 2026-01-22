# ECDLP Pollard Kangaroo Sample
===============================

This sample shows how to solve ECDLP on short Weierstrass curves using a Pollard kangaroo algorithm.

## Dependencies
- GMP development libraries (`libgmp`, `gmp.h`). Set `GMP_HOME` if not installed system-wide.
- CUDA toolkit (nvcc) and a supported NVIDIA GPU. CUDA 12+ is recommended.
- Optional: SageMath if you want to use the GPU solver via `ecdlp.py`.

## Building

Set proper `NBITS` in `ecc/ecc.h` for your curve size. `NBITS` **must** be a multiple of 128.

Then, run `make` based on your GPU architecture:

```
# Ampere GPU (sm_86)
make ampere

# Ada GPU (sm_89)
make ada

# Specify custom GMP path if needed
GMP_HOME=/path/to/gmp make ampere
```

## Usage

You can use the function provided in `ecdlp.py` to do curve conversion, run Pohlig-Hellman, and call the CPU or GPU solver automatically.
