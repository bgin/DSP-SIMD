# DSP-SIMD

C/C++ library containing explicitly vectorized Digital Signal Processing (DSP), complex arithmetic, linear algebra, and transcendental math primitives targeting x86/x86_64 SIMD instruction sets (SSE, AVX, AVX2, and AVX-512) [RF-EMT].

## Repository Structure

* **`src/common/`** — Platform configuration, aligned memory allocation, SIMD helper macros, and low-level hardware PMC instrumentation (`GMS_config.h`, `GMS_malloc.h`, `GMS_fast_pmc_access.h`).
* **`src/basic_functions/`** — Mixed-precision (FP32/FP64) AVX and AVX-512 (`zmm16r4`) element-wise vector operations.
* **`src/complex_math/`** — Packed complex register wrappers (`xmm`, `ymm`, `zmm`), overflow-safe complex division using Smith’s algorithm (`GMS_cdiv_smith_vec_zmm16r4`), and complex hyperbolic sine routines (standard intrinsics and Intel SVML).
* **`src/math_functions/`** — Vectorized special/transcendental functions, including modified Bessel functions of the first kind (`GMS_bessel_i0_sse.h`) and reference sine wave LUTs.
* **`src/linalg/`** — Small-matrix linear algebra routines such as Cholesky matrix inversion (`GMS_matrix_inv_cholesky`).
* **`unit-tests/` & `perf-tests/`** — Standalone drivers verifying SIMD outputs against reference implementations (covering both aligned and unaligned `_u` paths) alongside cycle-level benchmarks.

## Requirements & Building

* **Requirements:** x86/x86_64 CPU with SSE/AVX/AVX-512 support; GCC, Clang, or Intel oneAPI (`icx`/`icpx`) supporting C11/C++17.
* **Quick Build Examples:**

```bash
git clone https://github.com
cd DSP-SIMD

# Compile and run the SSE Bessel I0 unit test
g++ -O3 -msse4.2 -std=c++17 -I./src unit-tests/unit_test_bessel_i0_sse.cpp -o unit_test_bessel_i0_sse
./unit_test_bessel_i0_sse

# Compile and run an AVX-512 vector addition test
g++ -O3 -mavx512f -mavx512dq -std=c++17 -I./src src/basic_functions/GMS_add_vec_zmm16r4.cpp unit-tests/unit_test_add_vec_zmm16r4.cpp -o unit_test_add_vec_zmm16r4
./unit_test_add_vec_zmm16r4
```
```bash
git clone https://github.com
# CMake based compilation
cd DSP-SIMD
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Origin and License

Extracted from the numerical backend of [RF-EMT](https://github.com) and distributed under the GNU General Public License v3.0.

