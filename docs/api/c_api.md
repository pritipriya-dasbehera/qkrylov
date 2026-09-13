# `qkrylov` C API Reference (`extern "C"`)

The **`qkrylov` C API** provides a binary-stable, unmangled `extern "C"` interface for performing matrix-free exact diagonalization and Krylov calculations in quantum many-body physics. 

It enables seamless zero-copy interop with languages such as **C**, **Julia (`ccall`)**, **Rust (FFI)**, **Python (`ctypes`/`cffi`)**, **Fortran**, and **Go**.

---

## Table of Contents
1. [Overview & Core Features](#overview-core-features)
2. [Include & Linking](#include-linking)
3. [Opaque Handles & Data Types](#opaque-handles-data-types)
4. [Error Codes](#error-codes)
5. [API Function Reference](#api-function-reference)
   - [Sector API](#1-sector-api)
   - [Basis API](#2-basis-api)
   - [Site API](#3-site-api)
   - [OpSum API](#4-opsum-api)
   - [Device & Hardware Query API](#5-device-hardware-query-api)
   - [Matrix-Free Hamiltonian API](#6-matrix-free-hamiltonian-api)
   - [Solvers API](#7-solvers-api)
   - [Vector Operations (Kokkos Parallel BLAS-1 Kernels)](#vector-operations-kokkos-parallel-blas-1-kernels)
   - [Device-Resident Vectors & Zero-Copy GPU SpMV](#9-device-resident-vectors-zero-copy-gpu-spmv)
6. [Complete C Example](#complete-c-example)

---

## Overview & Core Features

- **Flat C ABI Linkage**: All functions are declared inside `extern "C"` blocks without C++ name mangling or class vtable dependencies.
- **Dual-Precision Architecture (FP32 & FP64)**: Full 64-bit IEEE 754 precision (`_fp64`, `double`) for CPU calculations requiring strict residual convergence ($< 10^{-12}$), alongside fast 32-bit precision (`_fp32`, `float`) for consumer GPU throughput and reduced VRAM footprint. Un-suffixed default functions alias to FP64 for maximum precision.
- **Opaque Handle Architecture**: C++ class instances (`Basis`, `Site`, `OpSum`, `MatrixFreeHamiltonian`, `DeviceVector`) are encapsulated behind opaque struct pointers (handles). Sector, Basis, Site, and OpSum handles are lightweight descriptors that work interchangeably with both FP32 and FP64 Hamiltonian instances.
- **Exception Safety**: Every C function catches internal C++ exceptions and returns standard integer error codes to prevent process crashes across language boundaries.
- **RAII Leak Protection**: Handle allocations use RAII mechanisms internally to ensure memory is never leaked even if construction fails.
- **Zero-Copy Performance**: Supports direct matrix-vector evaluation (`y = H * x`) using contiguous memory pointers (`[re, im, re, im...]`) or opaque GPU device handles (`qkrylov_device_vector_h`) without vector re-allocations or buffer copies.

---

## Include & Linking

To use the C API in your program, include the header file:

```c
#include <qkrylov/c_api.h>
```

Link against the built `qkrylov` library:

```bash
# Compile and link a C program
gcc -O3 main.c -I/path/to/qkrylov/include -L/path/to/qkrylov/build -lqkrylov -lm -o main_c
```

---

## Opaque Handles & Data Types

The C API uses incomplete struct pointers to represent C++ object instances:

| Handle Type | Encapsulated C++ Object | Description |
| :--- | :--- | :--- |
| `qkrylov_sector_h` | `qkrylov::Sector` | Quantum number conservation law / symmetry sector |
| `qkrylov_basis_h` | `qkrylov_basis_t` descriptor | Hilbert space basis (SpinHalf, SpinS, Fermion, Hubbard, t-J) |
| `qkrylov_site_h` | `qkrylov_site_t` descriptor | Site operator definition (SpinHalf, SpinS, Fermion, Hubbard, t-J) |
| `qkrylov_opsum_h` | `qkrylov_opsum_t` descriptor | Linear combination of local operator product terms |
| `qkrylov_hamiltonian_h` | `qkrylov_hamiltonian_t` | Matrix-free Hamiltonian evaluator ($y = Hx$, FP32 or FP64) |
| `qkrylov_device_vector_h` | `qkrylov_device_vector_t` | Device-resident vector handle for zero-copy GPU SpMV and BLAS-1 |

---

## Error Codes

Functions returning an `int` status code return one of the following macros defined in `qkrylov/c_api.h`:

| Constant | Value | Description |
| :--- | :---: | :--- |
| `QKRYLOV_SUCCESS` | `0` | Operation completed successfully. |
| `QKRYLOV_ERROR_INVALID_ARG` | `-1` | Null pointer passed or invalid argument range. |
| `QKRYLOV_ERROR_EXCEPTION` | `-2` | An internal C++ exception was caught safely. |

---

## API Function Reference

### 1. Sector API
Symmetries and quantum number sectors (e.g. total $S^z$, particle numbers).

#### `qkrylov_sector_create`
```c
qkrylov_sector_h qkrylov_sector_create(void);
```
Creates a new default symmetry sector handle. Returns `NULL` on allocation failure.

#### `qkrylov_sector_destroy`
```c
void qkrylov_sector_destroy(qkrylov_sector_h sector);
```
Frees the memory associated with a sector handle.

#### `qkrylov_sector_set_sz`
```c
int qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2);
```
Enforces total $S^z$ conservation sector where `sz2` is $2 \times S^z$ (e.g., `sz2 = 0` for $S^z = 0$). Returns `QKRYLOV_SUCCESS` on success.

#### `qkrylov_sector_set_n`
```c
int qkrylov_sector_set_n(qkrylov_sector_h sector, int n);
```
Enforces total particle number conservation for spinless fermion systems.

#### `qkrylov_sector_set_nb`
```c
int qkrylov_sector_set_nb(qkrylov_sector_h sector, int nb);
```
Enforces total boson particle number conservation.

---

### 2. Basis API
Constructs many-body Hilbert space bases.

#### `qkrylov_spinhalf_basis_create`
```c
qkrylov_basis_h qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector);
```
Creates a Spin-1/2 basis for `num_sites` sites, optionally restricted by `sector` (pass `NULL` for full basis).

#### `qkrylov_basis_create_spin_s`
```c
qkrylov_basis_h qkrylov_basis_create_spin_s(int N, double S, const qkrylov_sector_t* sector);
```
Creates a general Spin-$S$ basis (e.g. $S = 0.5, 1.0, 1.5, \dots$) for $N$ sites, optionally restricted to an $S^z$ sector (e.g. $S=1, N=4, S^z=0 \implies \dim = 19$).

#### `qkrylov_fermion_basis_create`
```c
qkrylov_basis_h qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector);
```
Creates a spinless fermion basis with Jordan-Wigner signs.

#### `qkrylov_hubbard_basis_create`
```c
qkrylov_basis_h qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector);
```
Creates an interacting Fermi-Hubbard basis (spin-$\uparrow$ and spin-$\downarrow$).

#### `qkrylov_tj_basis_create`
```c
qkrylov_basis_h qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector);
```
Creates a t-J model basis enforcing the no-double-occupancy constraint.

#### `qkrylov_basis_destroy`
```c
void qkrylov_basis_destroy(qkrylov_basis_h basis);
```
Destroys a basis handle and frees associated resources.

#### `qkrylov_basis_dimension`
```c
uint64_t qkrylov_basis_dimension(qkrylov_basis_h basis);
```
Returns the total dimension of the Hilbert space basis.

#### `qkrylov_basis_nsites`
```c
int qkrylov_basis_nsites(qkrylov_basis_h basis);
```
Returns the number of lattice sites in the basis.

#### `qkrylov_basis_state`
```c
uint64_t qkrylov_basis_state(qkrylov_basis_h basis, uint64_t index);
```
Returns the integer bitstring representation of the basis state at zero-based `index`.

#### `qkrylov_basis_index`
```c
int64_t qkrylov_basis_index(qkrylov_basis_h basis, uint64_t state_bitstring);
```
Returns the zero-based basis index for `state_bitstring`, or `-1` if the state does not belong to the sector basis.

#### `qkrylov_basis_contains`
```c
int qkrylov_basis_contains(qkrylov_basis_h basis, uint64_t state_bitstring);
```
Returns `1` if `state_bitstring` belongs to the sector basis, or `0` otherwise.

---

### 3. Site API
Defines site operator representations.

#### Constructor Functions
```c
qkrylov_site_h qkrylov_spinhalf_site_create(void);
qkrylov_site_h qkrylov_site_create_spin_s(double S);
qkrylov_site_h qkrylov_fermion_site_create(void);
qkrylov_site_h qkrylov_hubbard_site_create(void);
qkrylov_site_h qkrylov_tj_site_create(void);
```
Creates a site handle matching the corresponding model system. `qkrylov_site_create_spin_s` constructs general spin-$S$ matrix representations (local dimension $2S + 1$).

#### `qkrylov_site_destroy`
```c
void qkrylov_site_destroy(qkrylov_site_h site);
```
Frees a site handle.

---

### 4. OpSum API
Constructs linear combinations of operator strings. OpSum descriptors are precision-agnostic and convert automatically to FP32 or FP64 when bound to a Hamiltonian.

#### `qkrylov_opsum_create`
```c
qkrylov_opsum_h qkrylov_opsum_create(void);
```
Creates a new empty `OpSum` operator sum container.

#### `qkrylov_opsum_destroy`
```c
void qkrylov_opsum_destroy(qkrylov_opsum_h opsum);
```
Frees an `OpSum` container.

#### `qkrylov_opsum_clear`
```c
int qkrylov_opsum_clear(qkrylov_opsum_h opsum);
```
Removes all operator terms from the container.

#### `qkrylov_opsum_add_term_1body` & `_fp64` / `_fp32`
```c
int qkrylov_opsum_add_term_1body_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag, const char* op1, int site1);
int qkrylov_opsum_add_term_1body_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag, const char* op1, int site1);
int qkrylov_opsum_add_term_1body(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag, const char* op1, int site1); // Alias -> _fp64
```
Adds a single-body operator term (e.g. $h \cdot S_i^z$).

#### `qkrylov_opsum_add_term_2body` & `_fp64` / `_fp32`
```c
int qkrylov_opsum_add_term_2body_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag, const char* op1, int site1, const char* op2, int site2);
int qkrylov_opsum_add_term_2body_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag, const char* op1, int site1, const char* op2, int site2);
int qkrylov_opsum_add_term_2body(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag, const char* op1, int site1, const char* op2, int site2); // Alias -> _fp64
```
Adds a two-body operator interaction term (e.g. $J \cdot S_i^z S_j^z$ or $\frac{J}{2} S_i^+ S_j^-$).

#### `qkrylov_opsum_add_term_nbody` & `_fp64` / `_fp32`
```c
int qkrylov_opsum_add_term_nbody_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag, int n_factors, const char** ops, const int* sites);
int qkrylov_opsum_add_term_nbody_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag, int n_factors, const char** ops, const int* sites);
int qkrylov_opsum_add_term_nbody(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag, int n_factors, const char** ops, const int* sites); // Alias -> _fp64
```
Adds an arbitrary $N$-body operator interaction term (e.g. 3-body chiral term $S_i^x S_j^y S_k^z$ or 4-body ring exchange).

---

### 5. Device & Hardware Query API
Inspects compiled hardware acceleration backends and configures execution targets.

#### `qkrylov_is_gpu_build`
```c
int qkrylov_is_gpu_build(void);
```
Returns `1` if the shared library was compiled with GPU acceleration (CUDA, HIP, or SYCL), or `0` for a CPU-only build.

#### `qkrylov_find_gpu`
```c
const char* qkrylov_find_gpu(void);
```
Returns a string identifying the active GPU backend (`"cuda"`, `"hip"`, `"sycl"`), or `NULL` if the binary was built for CPU only.

#### `qkrylov_gpu_count`
```c
int qkrylov_gpu_count(void);
```
Returns the number of available physical GPUs detected on the host system.

#### `qkrylov_initialize_device`
```c
int qkrylov_initialize_device(const char* device_str);
```
Explicitly initializes the Kokkos execution spaces for a targeted device (e.g. `"cpu"`, `"cuda:0"`, `"hip:1"`). Returns `QKRYLOV_SUCCESS` on success.

---

### 6. Matrix-Free Hamiltonian API
Evaluates matrix-vector multiplication $y = Hx$ without storing the matrix. Supports both single-precision (`_fp32`, `float`) and double-precision (`_fp64`, `double`). Default un-suffixed functions alias to `_fp64`.

#### Constructors
```c
/* Double-Precision (FP64) Constructors (Default) */
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_fp64(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum);
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device_fp64(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum, const char* device_str);
qkrylov_hamiltonian_h qkrylov_hamiltonian_create(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum); // Alias -> _fp64
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum, const char* device_str); // Alias -> _fp64

/* Single-Precision (FP32) Constructors */
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_fp32(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum);
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device_fp32(qkrylov_basis_h basis, qkrylov_site_h site, qkrylov_opsum_h opsum, const char* device_str);
```

#### `qkrylov_hamiltonian_destroy` & `qkrylov_hamiltonian_dimension` & `qkrylov_hamiltonian_precision`
```c
void     qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h);
uint64_t qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h);
int      qkrylov_hamiltonian_precision(qkrylov_hamiltonian_h h); // Returns 0 for FP32, 1 for FP64
```

#### `qkrylov_hamiltonian_apply`
```c
int qkrylov_hamiltonian_apply_fp64(qkrylov_hamiltonian_h h, const double* x_real, const double* x_imag, double* y_real, double* y_imag);
int qkrylov_hamiltonian_apply_fp32(qkrylov_hamiltonian_h h, const float* x_real, const float* x_imag, float* y_real, float* y_imag);
int qkrylov_hamiltonian_apply(qkrylov_hamiltonian_h h, const double* x_real, const double* x_imag, double* y_real, double* y_imag); // Alias -> _fp64
```
Applies matrix action using separate real and imaginary arrays of size `dimension()`. Passing vectors of the wrong precision returns `QKRYLOV_ERROR_INVALID_ARG`.

#### `qkrylov_hamiltonian_apply_complex` (Zero-Copy)
```c
int qkrylov_hamiltonian_apply_complex_fp64(qkrylov_hamiltonian_h h, const double* x_complex, double* y_complex);
int qkrylov_hamiltonian_apply_complex_fp32(qkrylov_hamiltonian_h h, const float* x_complex, float* y_complex);
int qkrylov_hamiltonian_apply_complex(qkrylov_hamiltonian_h h, const double* x_complex, double* y_complex); // Alias -> _fp64
```
Performs **direct zero-copy** matrix-vector multiplication where `x_complex` and `y_complex` are contiguous arrays of $2 \times \text{dimension()}$ scalars `[re, im, re, im...]` (matching `std::complex<double>` or `std::complex<float>`).

#### `qkrylov_hamiltonian_diagonal`
```c
int qkrylov_hamiltonian_diagonal_fp64(qkrylov_hamiltonian_h h, double* diag_out);
int qkrylov_hamiltonian_diagonal_fp32(qkrylov_hamiltonian_h h, float* diag_out);
int qkrylov_hamiltonian_diagonal(qkrylov_hamiltonian_h h, double* diag_out); // Alias -> _fp64
```
Extracts matrix-free diagonal elements into caller-allocated buffer of length `dimension()`.

#### `qkrylov_hamiltonian_apply_device` (Pure Device SpMV)
```c
int qkrylov_hamiltonian_apply_device_fp64(qkrylov_hamiltonian_h h, const qkrylov_device_vector_h x_dev, qkrylov_device_vector_h y_dev);
int qkrylov_hamiltonian_apply_device_fp32(qkrylov_hamiltonian_h h, const qkrylov_device_vector_h x_dev, qkrylov_device_vector_h y_dev);
int qkrylov_hamiltonian_apply_device(qkrylov_hamiltonian_h h, const qkrylov_device_vector_h x_dev, qkrylov_device_vector_h y_dev); // Alias -> _fp64
```
Applies Hamiltonian matrix action directly on device memory (`y_dev = H * x_dev`). **Zero host↔device staging copies** occur during this operation. Both `x_dev` and `y_dev` must already reside on device memory.

#### `qkrylov_hamiltonian_diagonal_device`
```c
int qkrylov_hamiltonian_diagonal_device_fp64(qkrylov_hamiltonian_h h, qkrylov_device_vector_h diag_out);
int qkrylov_hamiltonian_diagonal_device_fp32(qkrylov_hamiltonian_h h, qkrylov_device_vector_h diag_out);
int qkrylov_hamiltonian_diagonal_device(qkrylov_hamiltonian_h h, qkrylov_device_vector_h diag_out); // Alias -> _fp64
```
Extracts the Hamiltonian diagonal directly into a device vector handle `diag_out` with zero host copies.

---

### 7. Solvers API

#### `qkrylov_lanczos_ground_state` & `_complex`
```c
typedef struct {
    double energy;
    int iterations;
    int converged;
} qkrylov_lanczos_result_fp64_t;
typedef qkrylov_lanczos_result_fp64_t qkrylov_lanczos_result_c_t;

typedef struct {
    float energy;
    int iterations;
    int converged;
} qkrylov_lanczos_result_fp32_t;

/* FP64 Endpoints (Default) */
int qkrylov_lanczos_ground_state_fp64(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_fp64_t* result);
int qkrylov_lanczos_ground_state_complex_fp64(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_fp64_t* result, double* eigenvector_complex);
int qkrylov_lanczos_ground_state(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_c_t* result);
int qkrylov_lanczos_ground_state_complex(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_c_t* result, double* eigenvector_complex);

/* FP32 Endpoints */
int qkrylov_lanczos_ground_state_fp32(qkrylov_hamiltonian_h h, int maxiter, float tol, qkrylov_lanczos_result_fp32_t* result);
int qkrylov_lanczos_ground_state_complex_fp32(qkrylov_hamiltonian_h h, int maxiter, float tol, qkrylov_lanczos_result_fp32_t* result, float* eigenvector_complex);
```

#### `qkrylov_lanczos_two_pass_ground_state` & `_complex`
Computes the ground state eigenvalue and eigenvector using a memory-frugal two-pass approach. In the first pass, tridiagonal coefficients are generated using only 3 working vectors. In the second pass, the Ritz eigenvector is reconstructed on-the-fly, eliminating the requirement to store all $M$ Krylov vectors in memory.
```c
/* FP64 Endpoints */
int qkrylov_lanczos_two_pass_ground_state_fp64(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_fp64_t* result);
int qkrylov_lanczos_two_pass_ground_state_complex_fp64(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_fp64_t* result, double* eigenvector_complex);
int qkrylov_lanczos_two_pass_ground_state(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_c_t* result);
int qkrylov_lanczos_two_pass_ground_state_complex(qkrylov_hamiltonian_h h, int maxiter, double tol, qkrylov_lanczos_result_c_t* result, double* eigenvector_complex);

/* FP32 Endpoints */
int qkrylov_lanczos_two_pass_ground_state_fp32(qkrylov_hamiltonian_h h, int maxiter, float tol, qkrylov_lanczos_result_fp32_t* result);
int qkrylov_lanczos_two_pass_ground_state_complex_fp32(qkrylov_hamiltonian_h h, int maxiter, float tol, qkrylov_lanczos_result_fp32_t* result, float* eigenvector_complex);
```

#### `qkrylov_lanczos_lowest_complex`
Computes the lowest $k$ eigenvalues and eigenvectors using thick-restart or full-subspace Lanczos iteration.
```c
typedef struct {
    int iterations;
    int converged;
} qkrylov_lanczos_lowest_result_c_t;

/* FP64 Endpoints */
int qkrylov_lanczos_lowest_complex_fp64(qkrylov_hamiltonian_h h, int n_eig, int maxiter, double tol, double* eigenvalues_out, double* eigenvectors_complex_out, qkrylov_lanczos_lowest_result_c_t* result_info, const double* initial_vector_complex);
int qkrylov_lanczos_lowest_complex(qkrylov_hamiltonian_h h, int n_eig, int maxiter, double tol, double* eigenvalues_out, double* eigenvectors_complex_out, qkrylov_lanczos_lowest_result_c_t* result_info, const double* initial_vector_complex);

/* FP32 Endpoints */
int qkrylov_lanczos_lowest_complex_fp32(qkrylov_hamiltonian_h h, int n_eig, int maxiter, float tol, float* eigenvalues_out, float* eigenvectors_complex_out, qkrylov_lanczos_lowest_result_c_t* result_info, const float* initial_vector_complex);
```

#### `qkrylov_davidson_lowest_complex`
```c
typedef struct {
    int iterations;
    int converged;
} qkrylov_davidson_result_c_t;

/* Subspace Davidson with Diagonal Preconditioning */
int qkrylov_davidson_lowest_complex_fp64(qkrylov_hamiltonian_h h, int n_eig, int max_subspace, double tol, double* eigenvalues_out, double* eigenvectors_complex_out, qkrylov_davidson_result_c_t* result_info);
int qkrylov_davidson_lowest_complex_fp32(qkrylov_hamiltonian_h h, int n_eig, int max_subspace, float tol, float* eigenvalues_out, float* eigenvectors_complex_out, qkrylov_davidson_result_c_t* result_info);
int qkrylov_davidson_lowest_complex(qkrylov_hamiltonian_h h, int n_eig, int max_subspace, double tol, double* eigenvalues_out, double* eigenvectors_complex_out, qkrylov_davidson_result_c_t* result_info);
```

#### `qkrylov_continued_fraction_coeffs_complex` & `qkrylov_evaluate_spectral_function`
```c
int    qkrylov_continued_fraction_coeffs_complex_fp64(qkrylov_hamiltonian_h h, const double* phi0_complex, int n_iter, double* alphas_out, double* betas_out, double* norm_phi0_out, int* num_coeffs_out);
int    qkrylov_continued_fraction_coeffs_complex_fp32(qkrylov_hamiltonian_h h, const float* phi0_complex, int n_iter, float* alphas_out, float* betas_out, float* norm_phi0_out, int* num_coeffs_out);
int    qkrylov_continued_fraction_coeffs_complex(qkrylov_hamiltonian_h h, const double* phi0_complex, int n_iter, double* alphas_out, double* betas_out, double* norm_phi0_out, int* num_coeffs_out);

double qkrylov_evaluate_spectral_function_fp64(const double* alphas, const double* betas, size_t n, double norm_phi0, double omega, double E0, double eta);
float  qkrylov_evaluate_spectral_function_fp32(const float* alphas, const float* betas, size_t n, float norm_phi0, float omega, float E0, float eta);
double qkrylov_evaluate_spectral_function(const double* alphas, const double* betas, size_t n, double norm_phi0, double omega, double E0, double eta);
```

#### `qkrylov_solver_correction_vector`
```c
typedef struct {
    double spectral_function;
    int iterations;
    int converged;
} qkrylov_correction_vector_result_fp64_t;
typedef qkrylov_correction_vector_result_fp64_t qkrylov_correction_vector_result_c_t;

typedef struct {
    float spectral_function;
    int iterations;
    int converged;
} qkrylov_correction_vector_result_fp32_t;

int qkrylov_solver_correction_vector_fp64(qkrylov_hamiltonian_h h, const double* op_psi0_complex, double e0, double omega, double eta, int max_iter, double tol, qkrylov_correction_vector_result_fp64_t* result, double* correction_vector_out_complex);
int qkrylov_solver_correction_vector_fp32(qkrylov_hamiltonian_h h, const float* op_psi0_complex, float e0, float omega, float eta, int max_iter, float tol, qkrylov_correction_vector_result_fp32_t* result, float* correction_vector_out_complex);
int qkrylov_solver_correction_vector(qkrylov_hamiltonian_h h, const double* op_psi0_complex, double e0, double omega, double eta, int max_iter, double tol, qkrylov_correction_vector_result_c_t* result, double* correction_vector_out_complex);
```

#### `qkrylov_ftlm` (Single Temperature)
```c
typedef struct {
    double beta;
    double partition_function;
    double internal_energy;
    double specific_heat;
} qkrylov_ftlm_result_fp64_t;
typedef qkrylov_ftlm_result_fp64_t qkrylov_ftlm_result_c_t;

typedef struct {
    float beta;
    float partition_function;
    float internal_energy;
    float specific_heat;
} qkrylov_ftlm_result_fp32_t;

int qkrylov_ftlm_fp64(qkrylov_hamiltonian_h h, double beta, int n_random, int n_steps, qkrylov_ftlm_result_fp64_t* result);
int qkrylov_ftlm_fp32(qkrylov_hamiltonian_h h, float beta, int n_random, int n_steps, qkrylov_ftlm_result_fp32_t* result);
int qkrylov_ftlm(qkrylov_hamiltonian_h h, double beta, int n_random, int n_steps, qkrylov_ftlm_result_c_t* result);
```

#### `qkrylov_ftlm_sweep` (Multi-Temperature & Observables Sweep)
Evaluates thermodynamic equations of state ($Z, F, E, C_v, S$) and arbitrary physical observables $\langle \hat{O}_m \rangle$ across a user-specified $\beta$ grid in a single pass without re-running Krylov iterations.
```c
typedef struct {
    int num_betas;
    int num_observables;
    const double* beta_grid;
    const double* partition_functions;
    const double* free_energies;
    const double* internal_energies;
    const double* specific_heats;
    const double* entropies;
    const double* observable_expectations; /* Row-major: num_observables x num_betas */
    const double* observable_errors;       /* Row-major: num_observables x num_betas */
} qkrylov_ftlm_sweep_result_fp64_t;
typedef qkrylov_ftlm_sweep_result_fp64_t qkrylov_ftlm_sweep_result_c_t;

typedef struct {
    int num_betas;
    int num_observables;
    const float* beta_grid;
    const float* partition_functions;
    const float* free_energies;
    const float* internal_energies;
    const float* specific_heats;
    const float* entropies;
    const float* observable_expectations; /* Row-major: num_observables x num_betas */
    const float* observable_errors;       /* Row-major: num_observables x num_betas */
} qkrylov_ftlm_sweep_result_fp32_t;

/* FP64 Endpoints */
int  qkrylov_ftlm_sweep_fp64(qkrylov_hamiltonian_h h, const double* beta_grid, int num_betas, const qkrylov_hamiltonian_h* observables, int num_observables, int n_random, int n_steps, uint64_t seed, qkrylov_ftlm_sweep_result_fp64_t* result);
void qkrylov_ftlm_sweep_result_free_fp64(qkrylov_ftlm_sweep_result_fp64_t* result);
int  qkrylov_ftlm_sweep(qkrylov_hamiltonian_h h, const double* beta_grid, int num_betas, const qkrylov_hamiltonian_h* observables, int num_observables, int n_random, int n_steps, uint64_t seed, qkrylov_ftlm_sweep_result_c_t* result);
void qkrylov_ftlm_sweep_result_free(qkrylov_ftlm_sweep_result_c_t* result);

/* Decoupled Workflow (Stage 1 Sampling + Stage 2 Evaluation) */
int  qkrylov_ftlm_sample_fp64(qkrylov_hamiltonian_h h, const qkrylov_hamiltonian_h* observables, int num_observables, int n_random, int n_steps, uint64_t seed, qkrylov_ftlm_samples_h* out_samples);
int  qkrylov_ftlm_evaluate_sweep_fp64(qkrylov_ftlm_samples_h samples, const double* beta_grid, int num_betas, qkrylov_ftlm_sweep_result_fp64_t* result);
int  qkrylov_ftlm_sample(qkrylov_hamiltonian_h h, const qkrylov_hamiltonian_h* observables, int num_observables, int n_random, int n_steps, uint64_t seed, qkrylov_ftlm_samples_h* out_samples);
int  qkrylov_ftlm_evaluate_sweep(qkrylov_ftlm_samples_h samples, const double* beta_grid, int num_betas, qkrylov_ftlm_sweep_result_c_t* result);
void qkrylov_ftlm_samples_destroy(qkrylov_ftlm_samples_h samples);
int  qkrylov_ftlm_samples_precision(qkrylov_ftlm_samples_h samples);

/* FP32 Endpoints */
int  qkrylov_ftlm_sweep_fp32(qkrylov_hamiltonian_h h, const float* beta_grid, int num_betas, const qkrylov_hamiltonian_h* observables, int num_observables, int n_random, int n_steps, uint64_t seed, qkrylov_ftlm_sweep_result_fp32_t* result);
void qkrylov_ftlm_sweep_result_free_fp32(qkrylov_ftlm_sweep_result_fp32_t* result);
int  qkrylov_ftlm_sample_fp32(qkrylov_hamiltonian_h h, const qkrylov_hamiltonian_h* observables, int num_observables, int n_random, int n_steps, uint64_t seed, qkrylov_ftlm_samples_h* out_samples);
int  qkrylov_ftlm_evaluate_sweep_fp32(qkrylov_ftlm_samples_h samples, const float* beta_grid, int num_betas, qkrylov_ftlm_sweep_result_fp32_t* result);
```

### Vector Operations (Kokkos Parallel BLAS-1 Kernels)

Hardware-accelerated Kokkos parallel reductions and dispatch kernels exported directly in the flat C ABI:

```c
/* Dot Product: <x|y> = sum_i conj(x_i) * y_i */
int qkrylov_vector_dot_fp64(uint64_t dim, const double* x_complex, const double* y_complex, double* dot_re, double* dot_im);
int qkrylov_vector_dot_fp32(uint64_t dim, const float* x_complex, const float* y_complex, float* dot_re, float* dot_im);
int qkrylov_vector_dot(uint64_t dim, const double* x_complex, const double* y_complex, double* dot_re, double* dot_im);

/* 2-Norm: ||x|| = sqrt(sum_i |x_i|^2) */
int qkrylov_vector_norm_fp64(uint64_t dim, const double* x_complex, double* norm_out);
int qkrylov_vector_norm_fp32(uint64_t dim, const float* x_complex, float* norm_out);
int qkrylov_vector_norm(uint64_t dim, const double* x_complex, double* norm_out);

/* AXPY: y = a * x + y */
int qkrylov_vector_axpy_fp64(uint64_t dim, double a_re, double a_im, const double* x_complex, double* y_complex);
int qkrylov_vector_axpy_fp32(uint64_t dim, float a_re, float a_im, const float* x_complex, float* y_complex);
int qkrylov_vector_axpy(uint64_t dim, double a_re, double a_im, const double* x_complex, double* y_complex);

/* Scale: x = a * x */
int qkrylov_vector_scal_fp64(uint64_t dim, double a_re, double a_im, double* x_complex);
int qkrylov_vector_scal_fp32(uint64_t dim, float a_re, float a_im, float* x_complex);
int qkrylov_vector_scal(uint64_t dim, double a_re, double a_im, double* x_complex);

/* Normalize: x = x / ||x|| */
int qkrylov_vector_normalize_fp64(uint64_t dim, double* x_complex);
int qkrylov_vector_normalize_fp32(uint64_t dim, float* x_complex);
int qkrylov_vector_normalize(uint64_t dim, double* x_complex);

/* Zero Fill: x = 0 */
int qkrylov_vector_zero_fill_fp64(uint64_t dim, double* x_complex);
int qkrylov_vector_zero_fill_fp32(uint64_t dim, float* x_complex);
int qkrylov_vector_zero_fill(uint64_t dim, double* x_complex);

/* Deep Copy: dst = src */
int qkrylov_vector_copy_fp64(uint64_t dim, const double* src_complex, double* dst_complex);
int qkrylov_vector_copy_fp32(uint64_t dim, const float* src_complex, float* dst_complex);
int qkrylov_vector_copy(uint64_t dim, const double* src_complex, double* dst_complex);
```

---

### 9. Device-Resident Vectors & Zero-Copy GPU SpMV
Enables managing GPU VRAM or accelerated device memory directly from C and higher-level bindings. Eliminates host $\leftrightarrow$ device staging bottlenecks when running hundreds of iterative SpMV multiplications or Krylov projection steps.

#### Allocation & Lifecycle
```c
/* Double-Precision (FP64) */
qkrylov_device_vector_h qkrylov_device_vector_create_fp64(uint64_t dim);
qkrylov_device_vector_h qkrylov_device_vector_create(uint64_t dim); // Alias -> _fp64

/* Single-Precision (FP32) */
qkrylov_device_vector_h qkrylov_device_vector_create_fp32(uint64_t dim);

/* Deallocation */
void qkrylov_device_vector_destroy(qkrylov_device_vector_h vec);
```

#### Metadata & Pointer Access
```c
uint64_t qkrylov_device_vector_dimension(qkrylov_device_vector_h vec);
int      qkrylov_device_vector_precision(qkrylov_device_vector_h vec); // Returns 0 for FP32, 1 for FP64
void*    qkrylov_device_vector_data(qkrylov_device_vector_h vec);      // Returns raw pointer to device memory
```

#### Staging Memory Transfers (Host <-> Device)
```c
/* Copy Host Array -> Device Vector */
int qkrylov_device_vector_copy_from_host_fp64(qkrylov_device_vector_h dst, const double* host_src_complex);
int qkrylov_device_vector_copy_from_host_fp32(qkrylov_device_vector_h dst, const float* host_src_complex);
int qkrylov_device_vector_copy_from_host(qkrylov_device_vector_h dst, const double* host_src_complex);

/* Copy Device Vector -> Host Array */
int qkrylov_device_vector_copy_to_host_fp64(const qkrylov_device_vector_h src, double* host_dst_complex);
int qkrylov_device_vector_copy_to_host_fp32(const qkrylov_device_vector_h src, float* host_dst_complex);
int qkrylov_device_vector_copy_to_host(const qkrylov_device_vector_h src, double* host_dst_complex);
```

#### Device Vector Operations (BLAS-1 Kernels)
Hardware-accelerated reductions and operations running purely within device memory:
```c
/* Dot Product: <x|y> */
int qkrylov_device_vector_dot_fp64(const qkrylov_device_vector_h x, const qkrylov_device_vector_h y, double* dot_re, double* dot_im);
int qkrylov_device_vector_dot_fp32(const qkrylov_device_vector_h x, const qkrylov_device_vector_h y, float* dot_re, float* dot_im);
int qkrylov_device_vector_dot(const qkrylov_device_vector_h x, const qkrylov_device_vector_h y, double* dot_re, double* dot_im);

/* 2-Norm: ||x|| */
int qkrylov_device_vector_norm_fp64(const qkrylov_device_vector_h x, double* norm_out);
int qkrylov_device_vector_norm_fp32(const qkrylov_device_vector_h x, float* norm_out);
int qkrylov_device_vector_norm(const qkrylov_device_vector_h x, double* norm_out);

/* AXPY: y = a*x + y */
int qkrylov_device_vector_axpy_fp64(double a_re, double a_im, const qkrylov_device_vector_h x, qkrylov_device_vector_h y);
int qkrylov_device_vector_axpy_fp32(float a_re, float a_im, const qkrylov_device_vector_h x, qkrylov_device_vector_h y);
int qkrylov_device_vector_axpy(double a_re, double a_im, const qkrylov_device_vector_h x, qkrylov_device_vector_h y);

/* SCAL: x = a*x */
int qkrylov_device_vector_scal_fp64(double a_re, double a_im, qkrylov_device_vector_h x);
int qkrylov_device_vector_scal_fp32(float a_re, float a_im, qkrylov_device_vector_h x);
int qkrylov_device_vector_scal(double a_re, double a_im, qkrylov_device_vector_h x);

/* Normalize: x = x / ||x|| */
int qkrylov_device_vector_normalize_fp64(qkrylov_device_vector_h x);
int qkrylov_device_vector_normalize_fp32(qkrylov_device_vector_h x);
int qkrylov_device_vector_normalize(qkrylov_device_vector_h x);

/* Zero Fill: x = 0 */
int qkrylov_device_vector_zero_fill_fp64(qkrylov_device_vector_h x);
int qkrylov_device_vector_zero_fill_fp32(qkrylov_device_vector_h x);
int qkrylov_device_vector_zero_fill(qkrylov_device_vector_h x);

/* Device-to-Device Copy: dst = src */
int qkrylov_device_vector_copy_fp64(const qkrylov_device_vector_h src, qkrylov_device_vector_h dst);
int qkrylov_device_vector_copy_fp32(const qkrylov_device_vector_h src, qkrylov_device_vector_h dst);
int qkrylov_device_vector_copy(const qkrylov_device_vector_h src, qkrylov_device_vector_h dst);
```

---

## Complete C Example

Below is a complete C program demonstrating double-precision exact diagonalization of a 4-site 1D Heisenberg antiferromagnet in the $S^z = 0$ symmetry sector:

```c
#include <qkrylov/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main(void) {
    int N = 4;
    printf("--- qkrylov C API Demo: 4-site Heisenberg Chain (Double Precision) ---\n");

    // 1. Create Sz=0 sector
    qkrylov_sector_h sector = qkrylov_sector_create();
    qkrylov_sector_set_sz(sector, 0);

    // 2. Create Spin-1/2 basis in Sz=0 sector
    qkrylov_basis_h basis = qkrylov_spinhalf_basis_create(N, sector);
    uint64_t dim = qkrylov_basis_dimension(basis);
    printf("Basis dimension: %llu\n", (unsigned long long)dim);

    // 3. Create SpinHalf Site operator rules
    qkrylov_site_h site = qkrylov_spinhalf_site_create();

    // 4. Construct Heisenberg OpSum (FP64)
    qkrylov_opsum_h opsum = qkrylov_opsum_create();
    for (int i = 0; i < N - 1; ++i) {
        // Sz_i Sz_{i+1}
        qkrylov_opsum_add_term_2body(opsum, 1.0, 0.0, "Sz", i, "Sz", i + 1);
        // 0.5 * Sp_i Sm_{i+1}
        qkrylov_opsum_add_term_2body(opsum, 0.5, 0.0, "Sp", i, "Sm", i + 1);
        // 0.5 * Sm_i Sp_{i+1}
        qkrylov_opsum_add_term_2body(opsum, 0.5, 0.0, "Sm", i, "Sp", i + 1);
    }

    // 5. Create Matrix-Free Hamiltonian (Default FP64)
    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create(basis, site, opsum);
    assert(qkrylov_hamiltonian_precision(H) == 1); // 1 = FP64

    // 6. Run Lanczos Ground State Solver (FP64)
    qkrylov_lanczos_result_c_t result;
    int status = qkrylov_lanczos_ground_state(H, 200, 1e-12, &result);

    if (status == QKRYLOV_SUCCESS) {
        printf("FP64 Ground State Energy: %.14f (Exact: -1.61602540378444)\n", result.energy);
        printf("Converged: %s in %d iterations\n", result.converged ? "YES" : "NO", result.iterations);
    } else {
        printf("Lanczos solver failed with error code: %d\n", status);
    }

    // 7. Cleanup Handles safely
    qkrylov_hamiltonian_destroy(H);
    qkrylov_opsum_destroy(opsum);
    qkrylov_site_destroy(site);
    qkrylov_basis_destroy(basis);
    qkrylov_sector_destroy(sector);

    return 0;
}
```
