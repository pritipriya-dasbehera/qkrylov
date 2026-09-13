# Performance & Optimization Guide

This guide details the architectural design principles, memory scaling characteristics, and practical optimization tips for achieving peak performance with `qkrylov`.

---

## 1. Matrix-Free Memory Scaling

In quantum many-body physics, Hilbert space dimension $D$ scales exponentially with system size $N$:

- Spin-1/2 ($S_z = 0$ sector): $D = \binom{N}{N/2}$
- Fermi-Hubbard (half-filling): $D = \binom{N}{N/2}^2$

| System Size ($N$) | Hilbert Space $\dim (D)$ | Dense Matrix Memory ($16 D^2$) | Sparse CSR Matrix ($\approx 10 D$) | `qkrylov` Matrix-Free ($3 \times 16 D$) |
| :---: | :---: | :---: | :---: | :---: |
| **12 sites** | 924 | 13.6 MB | 148 KB | **44 KB** |
| **16 sites** | 12,870 | 2.65 GB | 2.0 MB | **618 KB** |
| **20 sites** | 184,756 | 546 GB | 29.5 MB | **8.8 MB** |
| **24 sites** | 2,704,156 | 116 TB | 432 MB | **129 MB** |
| **28 sites** | 40,116,600 | 25.7 PB | 6.4 GB | **1.9 GB** |
| **32 sites** | 601,080,390 | 5.7 EB | 96 GB | **28.8 GB** |

Traditional sparse-matrix methods must allocate gigabytes to store non-zero column indices and matrix elements before performing any calculations. `qkrylov` **never constructs or stores the matrix**, applying operator sums on-the-fly directly to the vector memory.

---

## 2. Solver Memory Profiles: One-Pass vs. Two-Pass

When computing ground states and Ritz wavefunctions, solver memory depends heavily on how Krylov basis vectors are handled:

### One-Pass Lanczos
- Generates and stores all $M$ Krylov vectors $v_1, \dots, v_M$ in memory:
  $$\text{Memory} \approx M \times D \times 16 \text{ bytes}$$
- For $D = 4 \times 10^7$ and $M = 100$, this requires **~64 GB RAM**.

### Two-Pass Lanczos (`TwoPass` / `two_pass=True`)
- **Pass 1**: Iterates using only 3 working vectors ($v_{j-1}, v_j, v_{j+1}$) to generate the tridiagonal coefficients $\alpha_n, \beta_n$. Eigenvalues and small subspace eigenvectors $s$ are computed.
- **Pass 2**: Re-runs the Lanczos recurrence from the same initial state, accumulating the ground state wavefunction on-the-fly: $|\psi_0\rangle = \sum_{j=1}^M s_j |v_j\rangle$.
- **Memory**: Strictly $3 \times D \times 16 \text{ bytes}$ (**~1.9 GB RAM** for $D = 4 \times 10^7$).
- **Trade-off**: Requires running $2M$ matrix-vector products instead of $M$, but enables simulations of systems that would otherwise crash from out-of-memory errors.

---

## 3. Decoupled FTLM Sweeps: 100x Efficiency Boost

Traditional Finite Temperature Lanczos implementations re-run the entire Krylov iteration for every single temperature $\beta$ and for every observable $\hat{O}$, scaling as $\mathcal{O}(N_T \times N_{\text{obs}} \times R \times M)$.

In `qkrylov`, FTLM is split into two independent stages:

1. **Stage 1 (Krylov Sampling)**:
   For each random starting vector, execute $M$ Lanczos steps and project each user observable onto the Krylov subspace:
   $$\mathcal{O}_{jk} = \langle v_j | \hat{O} | v_k \rangle$$
2. **Stage 2 (Thermodynamic Evaluation)**:
   Once the Krylov eigenpairs $(\epsilon_j, y_j)$ and observable projections $\mathcal{O}_{jk}$ are stored in tiny $M \times M$ matrices, evaluating partition functions $Z(\beta)$, free energies $F(\beta)$, specific heat $C_v(\beta)$, and observables $\langle \hat{O} \rangle_\beta$ across a grid of 1,000 temperature points takes **milliseconds** with zero matrix-vector products.

---

## 4. Hardware Acceleration & Threading

### CPU Multi-Core Scaling (OpenMP)
`qkrylov` parallelizes all matrix-free vector operations across CPU cores using OpenMP:

```bash
# Recommended OpenMP affinity settings for high-performance servers:
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```

### GPU Acceleration & Device-Resident Vectors
For GPU-accelerated builds (CUDA / ROCm), avoiding host-to-device memory transfer overhead is essential:
- Using `MatrixFreeHamiltonian` on `"cuda:0"` keeps operator actions purely inside GPU VRAM.
- `qkrylov_device_vector_h` (C ABI) and Kokkos device execution spaces execute repeated SpMV products and BLAS-1 vector reductions without host synchronization bottlenecks.

---

## 5. Dual Precision (FP64 vs. FP32)

| Precision | C++ Scalar | Python / NumPy | Memory Footprint | Convergence Tolerance | Ideal Use Case |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **FP64** | `double` | `complex128` | $16 \text{ bytes/element}$ | Down to $10^{-14}$ | Ground state precision, close-lying excited state gaps |
| **FP32** | `float` | `complex64` | $8 \text{ bytes/element}$ | Down to $10^{-6}$ | Consumer GPUs (RTX 30xx/40xx), high-temperature FTLM, broad dynamical spectral functions $S(\omega)$ |

Using FP32 doubles effective GPU VRAM capacity, allowing you to simulate systems with twice the Hilbert space dimension.
