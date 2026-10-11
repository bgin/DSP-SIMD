[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

**DSP-SIMD** is a high-performance, open-source C++ toolkit engineered for computationally intensive **Digital Signal Processing (DSP)** workloads. By leveraging hardware-level **SIMD (Single Instruction, Multiple Data)** vector instructions, this library accelerates core mathematical algorithms, transforming execution timelines from serial latency into parallel throughput. 

The library is architected for developers building real-time audio processors, software-defined radio (SDR) pipelines, telecommunication stacks, and high-frequency sensor arrays where raw computational efficiency is a hard requirement.

---

## 🚀 Key Architectural Features

*   **Hardware-Accelerated Vectorization:** Explicitly utilizes processor vector registers to compute multiple data points simultaneously, unlocking near-linear speedups on modern CPU architectures.
*   **Modular DSP Primitives:** Highly decoupled, reusable components designed for seamless drop-in integration into complex DSP pipelines.
*   **High-Performance Profiling:** Includes a robust integration test suite and benchmarking subsystem located under `perf-tests/` and `unit-tests/` to guarantee algorithmic precision and measure strict execution cycle optimizations.
*   **Modern C++ Design:** Adheres to low-overhead design patterns, zero-cost abstractions, and strict cache-aligned memory management protocols.

---

## 🛠 Supported SIMD Topologies

The toolkit is designed to abstract platform-specific compiler intrinsics, providing cross-platform scalability across a variety of hardware vectors:
*   **x86 / x86_64:** Intel & AMD hardware acceleration platforms including SSE4.2, AVX, AVX2, and AVX-512 extensions.


---

## 📦 Core DSP Library Modules

The codebase is systematically isolated into the following key execution modules:

```text
├── src/            # Core SIMD algorithm primitives and vectorized pipelines
├── perf-tests/     # Low-level performance evaluation benchmarks and cycle tracking
└── unit-tests/     # Strict logical correctness verification suites
```

### 1. Vectorized Arithmetic & Operations
Optimized block memory arithmetic routines processing contiguous data arrays in single-cycle instruction bounds:
*   Vector-Vector and Vector-Scalar Addition, Subtraction, Multiplication, and Division.
*   Complex number multi-lane multiplication and fused-multiply-add (FMA) routines.

---

## ⚙️ Getting Started

### Prerequisites
*   A modern C++ compiler supporting C++17 or higher (GCC, Clang, or MSVC).
*   Hardware support for SIMD extensions (e.g., AVX2 or ARM NEON enabled via compiler flags like `-march=native`).

### Cloning the Repository
Retrieve the source codebase directly from GitHub:

```bash
git clone https://github.com/bgin/DSP-SIMD.git
cd DSP-SIMD
```

### Compilation & Optimization Flags
To harness the true performance of this toolkit, ensure target-specific optimization flags are provided to the compiler during compilation:

```bash
# Example compilation command enabling native vectorization instructions
g++ -O3 -march=native -std=c++17 src/main.cpp -I ./src -o dsp_simd_app
```

---

## 🧪 Verification & Benchmarking

The toolkit features dedicated sub-systems to verify numerical accuracy against standardized floating-point benchmarks, while simultaneously logging instruction performance.

### Running Unit Tests
To confirm cross-platform precision and edge-case correctness:
```bash
# Navigate to unit test directory and compile suite
cd unit-tests
# Execute unit testing pipeline
```

### Executing Performance Tests
To run raw cycle benchmarks and compute throughput speedup analysis:
```bash
cd perf-tests
# Execute performance binaries to monitor SIMD processing improvements
```

---

## 📄 Licensing & Contribution

This toolkit is open-source and distributed under the terms of the **GNU General Public License v3 (GPLv3)**. 

*   Feel free to file issues, feature requests, or submit optimization pull requests via the official [GitHub Repository](https://github.com/bgin/DSP-SIMD).
*   *Created and actively maintained by bgin.*

