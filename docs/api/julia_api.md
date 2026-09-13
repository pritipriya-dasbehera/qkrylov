# Julia API Documentation (`QuantumKrylov.jl`)

`QuantumKrylov.jl` provides idiomatic, high-performance Julia bindings for `qkrylov` via zero-copy C ABI calls (`ccall`) and full integration with the **SciML CommonSolve** dispatch interface (`solve(prob, alg)`).

---

## 1. Package Installation & Module Setup

In the Julia REPL (press `]` to enter Pkg mode):

```julia
pkg> add https://github.com/sjp95/qkrylov.git#julia-latest:bindings/julia
```

Or in Julia code:

```julia
using QuantumKrylov
```

### Shared Library Loading
`QuantumKrylov.jl` automatically locates prebuilt `qkrylov_jll` binary artifacts or local build artifacts (`libqkrylov.so`, `libqkrylov.dylib`, `qkrylov.dll`). To specify a custom shared library path:
```bash
export QKRYLOV_LIB_PATH=/path/to/libqkrylov.so
```

---

## 2. Symmetry Sectors (`Sector`)

Symmetry sectors restrict the many-body Hilbert space to target quantum numbers.

```julia
sec = Sector()
set_sz!(sec, 0) # Enforces Sz = 0 (pass 2 * Sz)
```

### Methods
- `Sector()`: Allocates a symmetry sector handle with automatic GC finalizer.
- `set_sz!(sec::Sector, sz2::Integer)`: Restricts total $S_z$ projection ($2 \times S_z$). For $S_z=0$, pass `0`; for $S_z=1/2$, pass `1`.
- `set_hubbard_particles!(sec::Sector, nup::Integer, ndn::Integer)`: Restricts spin-up ($N_\uparrow$) and spin-down ($N_\downarrow$) electron counts.
- `set_n!(sec::Sector, n::Integer)`: Restricts total particle number for spinless fermions.
- `set_nb!(sec::Sector, nb::Integer)`: Restricts total boson particle number.

---

## 3. Site Definitions (`AbstractSite`)

Site models describe local degrees of freedom and local operator algebras:

```julia
s1 = SpinHalfSite()      # Spin-1/2 (dim 2: |up>, |down>)
s2 = SpinSSite(1.0)       # Spin-S (e.g. S=1, dim 3: |+1>, |0>, |-1>)
s3 = FermionSite()       # Spinless fermion (dim 2: |0>, |1>)
s4 = HubbardSite()       # Spinful Fermi-Hubbard (dim 4: |0>, |up>, |down>, |up down>)
s5 = TJSite()            # t-J model (dim 3: |0>, |up>, |down>, no double occupancy)
```

---

## 4. Hilbert Space Bases (`AbstractBasis`)

Constructs many-body basis representations across $N$ lattice sites:

```julia
b1 = SpinHalfBasis(N; sz=0)                 # Spin-1/2 basis (Sz = 0)
b2 = SpinSBasis(N, 1.0; sector=sec)         # Spin-1 basis
b3 = FermionBasis(N; n=2)                   # Spinless fermion basis with 2 particles
b4 = HubbardBasis(N; nup=1, ndn=1)          # Fermi-Hubbard basis (1 up, 1 down)
b5 = TJBasis(N; nup=1, ndn=1)               # t-J basis
```

### Inspection Methods
- `dimension(b::AbstractBasis)::UInt64`: Total Hilbert space dimension.
- `nsites(b::AbstractBasis)::Int`: Number of physical sites.
- `state(b::AbstractBasis, index::Integer)::UInt64`: Integer bitstring representation at 0-based index.
- `basis_index(b::AbstractBasis, bitstring::Unsigned)::Int64`: 0-based basis index for a given bitstring (or `-1`).
- `bitstring in basis`: Returns `true` if `bitstring` belongs to the basis.
- `b[i]`: 1-based indexing returning state bitstring at index `i`.

---

## 5. Operator Expressions (`OpSum`)

`OpSum` constructs Hamiltonian and observable expressions using natural operator algebra:

```julia
op = OpSum()

# Spin-1/2 Heisenberg chain
for i in 0:(N-2)
    global op += 1.0 * Sz(i) * Sz(i+1) + 0.5 * (Sp(i) * Sm(i+1) + Sm(i) * Sp(i+1))
end

# Hubbard interaction: t-V hopping + onsite U
for i in 0:(N-2)
    global op += -1.0 * (CdagUp(i) * CUp(i+1) + CdagDn(i) * CDn(i+1))
end
for i in 0:(N-1)
    global op += 4.0 * Nupdn(i)
end
```

### Supported Operator Symbols
- **Spin-1/2**: `Sz(i)`, `Sp(i)`, `Sm(i)`, `Sx(i)`, `Sy(i)`
- **Fermions**: `c(i)`, `cdag(i)`, `n(i)`
- **Hubbard**: `CdagUp(i)`, `CUp(i)`, `CdagDn(i)`, `CDn(i)`, `Nup(i)`, `Ndn(i)`, `Nupdn(i)`
- **Bosons**: `Bdag(i)`, `B(i)`, `N(i)`

---

## 6. Matrix-Free Hamiltonian (`MatrixFreeHamiltonian`)

Evaluates $y = H x$ on-the-fly without materializing the matrix in memory:

```julia
# Constructs MatrixFreeHamiltonian (site model inferred from basis)
# Targets GPU if available, otherwise CPU:
device = is_gpu_build() ? "cuda:0" : "cpu"
H = MatrixFreeHamiltonian(basis, op; device=device)

# Zero-copy matrix-vector multiplication
x = rand(ComplexF64, dimension(H))
y = H * x

# Extract matrix diagonal
diag_H = diagonal(H)
```

---

## 7. SciML CommonSolve Interface (`solve(prob, alg)`)

`QuantumKrylov.jl` implements standard SciML problem/algorithm dispatches:

### A. Ground State (`GroundStateProblem`)
```julia
prob = GroundStateProblem(H)

# One-pass Lanczos
sol = solve(prob, Lanczos(maxiter=100, tol=1e-12, compute_eigenvector=true))
println("Ground state energy: ", sol.energy)
println("Wavefunction:        ", sol.state)

# Two-pass Lanczos (memory frugal: saves only 3 working vectors during Lanczos pass)
sol_tp = solve(prob, Lanczos(maxiter=100, tol=1e-12, compute_eigenvector=true, two_pass=true))
```

### B. Low-Lying Excited States (`ExcitedStatesProblem`)
```julia
prob = ExcitedStatesProblem(H, 3)

# Davidson subspace solver
sol_dav = solve(prob, Davidson(n_eig=3, max_subspace=20, tol=1e-8, compute_eigenvectors=true))
println("Lowest 3 energies: ", sol_dav.eigenvalues)
println("Eigenvectors:      ", sol_dav.eigenvectors)

# Lanczos lowest-k solver
sol_lanc = solve(prob, Lanczos(n_eig=3, maxiter=200, tol=1e-8))
```

### C. Thermodynamics & Multi-Temperature Sweeps (`ThermalProblem`)
```julia
# Single temperature
prob_single = ThermalProblem(H, 1.0)
sol_single = solve(prob_single, FTLM(n_random=10, n_steps=50))
println("Z(beta=1.0) = ", sol_single.partition_function)
println("E(beta=1.0) = ", sol_single.internal_energy)

# Multi-temperature grid with arbitrary observables
betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
prob_sweep = ThermalProblem(H, betas; observables=[H, Sz0_H])
sol_sweep = solve(prob_sweep, FTLM(n_random=20, n_steps=60, seed=42))

println("Free energies:     ", sol_sweep.free_energies)
println("Specific heats:    ", sol_sweep.specific_heats)
println("Entropies:         ", sol_sweep.entropies)
println("Observable <O>:    ", sol_sweep.observable_expectations)
println("Observable error:  ", sol_sweep.observable_errors)
```

### D. Dynamical Response (`DynamicalProblem` & `CorrectionVectorProblem`)
```julia
# Continued Fraction
prob_dyn = DynamicalProblem(H, phi0; e0=E0)
sol_dyn = solve(prob_dyn, ContinuedFraction(n_iter=100))
A_w = evaluate_spectral_function(sol_dyn, 1.0, E0, 0.05)

# Correction Vector (Shifted Linear Solve)
prob_cv = CorrectionVectorProblem(H, phi0; e0=E0, omega=1.0, eta=0.05)
sol_cv = solve(prob_cv, CorrectionVector(maxiter=500, tol=1e-8, return_vector=true))
println("Spectral weight: ", sol_cv.spectral_function)
println("Correction vec:  ", sol_cv.vector)
```

---

## 8. Functional Solvers Reference

| Function | Description | Key Arguments | Return Type |
| :--- | :--- | :--- | :--- |
| `lanczos_ground_state(H; ...)` | Ground state energy & wavefunction | `maxiter=100`, `tol=1e-12`, `return_state=false`, `two_pass=false` | `LanczosResult` |
| `davidson_lowest(H; ...)` | Subspace Davidson eigensolver | `n_eig=1`, `max_subspace=20`, `tol=1e-8`, `compute_eigenvectors=true` | `DavidsonResult` |
| `lanczos_lowest(H; ...)` | Thick-restart / full Lanczos lowest $k$ | `n_eig=1`, `maxiter=100`, `tol=1e-8`, `compute_eigenvectors=true` | `LanczosLowestResult` |
| `ftlm_sweep(H, beta_grid; ...)` | Multi-temperature FTLM sweep with observables | `observables=[]`, `n_random=10`, `n_steps=50`, `seed=0` | `FTLMSweepResult` |
| `ftlm(H; beta, ...)` | Single-temperature FTLM | `beta=1.0`, `n_random=10`, `n_steps=50` | `FTLMResult` |
| `continued_fraction_coeffs(H, phi0; ...)` | Continued fraction Lanczos $\alpha, \beta$ coefficients | `n_iter=100` | `ContinuedFractionResult` |
| `evaluate_spectral_function(cfr, w, E0, eta)` | Continued fraction Green's function evaluation | `w`, `E0`, `eta` | `Float64` |
| `solver_correction_vector(H, phi0; ...)` | Correction vector linear system solve | `e0`, `omega`, `eta=0.1`, `maxiter=500`, `tol=1e-8`, `return_vector=false` | `CorrectionVectorResult` |

---

## 9. Hardware & Accelerator Queries

```julia
using QuantumKrylov

# Check if compiled with GPU acceleration (CUDA, HIP, SYCL)
is_gpu = is_gpu_build() # Bool

# Active GPU backend name ("cuda", "hip", "sycl", or nothing)
gpu = find_gpu()

# Available physical GPU count
count = gpu_count()

# Explicitly initialize device runtime target
initialize_device!("cuda:0")
```

---

## 10. Complete End-to-End Example

```julia
using QuantumKrylov

# 1. Setup 6-site Spin-1/2 Heisenberg chain with Sz=0
N = 6
basis = SpinHalfBasis(N; sz=0)
op = OpSum()
for i in 0:(N-2)
    global op += 1.0 * Sz(i) * Sz(i+1) + 0.5 * (Sp(i) * Sm(i+1) + Sm(i) * Sp(i+1))
end

target = is_gpu_build() ? "cuda:0" : "cpu"
H = MatrixFreeHamiltonian(basis, op; device=target)

# 2. Ground state via SciML Lanczos
sol_gs = solve(GroundStateProblem(H), Lanczos(compute_eigenvector=true))
println("Ground state energy: ", sol_gs.energy)

# 3. Excited states via Davidson
sol_dav = solve(ExcitedStatesProblem(H, 3), Davidson(n_eig=3))
println("Lowest 3 energies:   ", sol_dav.eigenvalues)

# 4. Multi-temperature thermodynamic sweep
betas = [0.1, 0.5, 1.0, 2.0, 5.0]
sol_th = solve(ThermalProblem(H, betas; observables=[H]), FTLM(n_random=20, n_steps=50))
println("Specific heat Cv:    ", sol_th.specific_heats)
```
