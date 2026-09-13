# Quickstart: Heisenberg Model in 5 Minutes

Welcome to **qkrylov**! In this guide, we'll walk you through solving the quantum Heisenberg model step-by-step across **Python**, **Julia**, and **C++**.

By the end of this tutorial, you'll know how to define a Hilbert space, build operators, construct a matrix-free Hamiltonian, and find its ground state.

---

## Installation

=== "🐍 Python (Pip)"
    ```bash
    # Standard CPU (OpenMP accelerated)
    pip install qkrylov

    # NVIDIA CUDA 12
    pip install qkrylov --extra-index-url https://peithonking.github.io/qkrylov-wheels/cuda

    # AMD ROCm 6
    pip install qkrylov --extra-index-url https://peithonking.github.io/qkrylov-wheels/rocm
    ```

=== "🔴 Julia (Pkg)"
    ```julia
    using Pkg
    # Automatically downloads prebuilt native binaries (CPU or CUDA 12 on Linux)
    Pkg.add(url="https://github.com/sjp95/qkrylov.git", rev="julia-latest", subdir="bindings/julia")
    ```

=== "⚙️ C++ (Source)"
    ```bash
    git clone https://github.com/sjp95/qkrylov.git
    cd qkrylov
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4
    ```

---

## Step 1: Define the Hilbert Space (Basis)

In `qkrylov`, the Hilbert space is defined using a `Basis` object. For a spin-1/2 system, we use `SpinHalfBasis`.

Restricting the calculation to a conserved symmetry sector (like total $S_z = 0$) drastically reduces the dimension of the Hilbert space:

=== "Python"
    ```python
    import qkrylov as qk

    # A system of N=4 sites, restricting to the total Sz = 0 sector.
    N = 4
    sector = qk.Sector()
    sector.set_sz(0)
    basis = qk.SpinHalfBasis(N, sector)
    print(f"Sector dimension: {basis.dimension}")  # 6 states
    ```

=== "Julia"
    ```julia
    using QuantumKrylov

    N = 4
    sec = Sector()
    set_sz!(sec, 0)
    basis = SpinHalfBasis(N, sec)
    println("Sector dimension: ", dimension(basis))  # 6 states
    ```

=== "C++"
    ```cpp
    #include <qkrylov/qkrylov.hpp>
    #include <iostream>

    int N = 4;
    auto sector = std::make_shared<qkrylov::Sector>();
    sector->set_sz(0);
    auto basis = std::make_shared<qkrylov::SpinHalfBasis>(N, sector);
    std::cout << "Sector dimension: " << basis->dimension() << std::endl;
    ```

---

## Step 2: Define the Local Physics (Site)

A `Site` object defines what local operators are available at each lattice site. For spin-1/2 systems, `SpinHalfSite` provides the spin operators: `Sz`, `Sp` ($S^+$), and `Sm` ($S^-$).

=== "Python"
    ```python
    site = qk.SpinHalfSite()
    ```

=== "Julia"
    ```julia
    site = SpinHalfSite()
    # Note: When constructing MatrixFreeHamiltonian, site is automatically inferred if omitted
    ```

=== "C++"
    ```cpp
    auto site = std::make_shared<qkrylov::SpinHalfSite>();
    ```

---

## Step 3: Build the Hamiltonian (OpSum)

We represent our Hamiltonian as a sum of local operator terms using `OpSum`.
The 1D Heisenberg antiferromagnet Hamiltonian is:

$$H = J \sum_{i=0}^{N-2} \left[ S^z_i S^z_{i+1} + \frac{1}{2} \left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \right) \right]$$

=== "Python"
    ```python
    ops = qk.OpSum()
    J = 1.0

    for i in range(N - 1):
        ops += J, 'Sz', i, 'Sz', i + 1
        ops += J * 0.5, 'Sp', i, 'Sm', i + 1
        ops += J * 0.5, 'Sm', i, 'Sp', i + 1
    ```

=== "Julia"
    ```julia
    op = OpSum()
    J = 1.0

    for i in 0:(N - 2)
        global op += J * Sz(i) * Sz(i + 1) + (J * 0.5) * (Sp(i) * Sm(i + 1) + Sm(i) * Sp(i + 1))
    end
    ```

=== "C++"
    ```cpp
    qkrylov::OpSum ops;
    double J = 1.0;

    for (int i = 0; i < N - 1; ++i) {
        ops += {J, {{"Sz", i}, {"Sz", i + 1}}};
        ops += {J * 0.5, {{"Sp", i}, {"Sm", i + 1}}};
        ops += {J * 0.5, {{"Sm", i}, {"Sp", i + 1}}};
    }
    ```

---

## Step 4: Construct the MatrixFreeHamiltonian

The full Hamiltonian matrix is **never materialized in memory**. Instead, matrix-vector products $y = H x$ are computed directly on-the-fly, scaling memory strictly as $\mathcal{O}(\text{dim})$.

=== "Python"
    ```python
    H = qk.MatrixFreeHamiltonian(basis, site, ops)
    print(f"Hamiltonian dimension: {H.dimension}")

    # Matrix-vector multiplication (zero-copy NumPy interop):
    import numpy as np
    x = np.random.rand(H.dimension).astype(np.complex128)
    y = H.apply(x)

    # Extract diagonal elements:
    diag = H.diagonal()
    ```

=== "Julia"
    ```julia
    H = MatrixFreeHamiltonian(basis, site, op)
    println("Hamiltonian dimension: ", dimension(H))

    # Matrix-vector multiplication (zero-copy):
    x = rand(ComplexF64, dimension(H))
    y = H * x

    # Extract diagonal elements:
    diag_H = diagonal(H)
    ```

=== "C++"
    ```cpp
    qkrylov::MatrixFreeHamiltonian H(basis, site, ops);
    std::cout << "Hamiltonian dimension: " << H.dimension() << std::endl;
    ```

---

## Step 5: Solve for Ground State (Lanczos)

Use the Lanczos algorithm to compute the ground-state energy and Ritz eigenvector:

=== "Python"
    ```python
    res = qk.lanczos_ground_state(H, compute_eigenvectors=True)
    print(f"Ground State Energy: {res.energy:.10f}")  # -1.6160254038
    print(f"Iterations: {res.iterations}, Converged: {res.converged}")
    psi0 = res.eigenvector  # 1D numpy array
    ```

=== "Julia"
    ```julia
    # Functional interface:
    res = lanczos_ground_state(H; return_state=true)
    println("Ground State Energy: ", res.energy)  # -1.6160254038
    println("Iterations: ", res.iterations, ", Converged: ", res.converged)
    psi0 = res.state

    # SciML CommonSolve interface:
    sol = solve(GroundStateProblem(H), Lanczos(compute_eigenvector=true))
    println("SciML Energy: ", sol.energy)
    ```

=== "C++"
    ```cpp
    auto [energy, state] = qkrylov::lanczos_ground_state(H);
    std::cout << "Ground State Energy: " << energy << std::endl;
    ```

---

## Step 6: SciPy & SciML Ecosystem Integration

### Python: SciPy `LinearOperator`
`MatrixFreeHamiltonian` can be wrapped directly as a SciPy `LinearOperator`:

```python
import scipy.sparse.linalg as sla

lin_op = H.as_linear_operator()
evals, evecs = sla.eigsh(lin_op, k=2, which='SA')
print(f"SciPy Ground State: {evals[0]}")
```

### Julia: SciML `solve(prob, alg)`
```julia
using QuantumKrylov

prob = GroundStateProblem(H)
sol = solve(prob, Lanczos(maxiter=100, tol=1e-12))
```

---

## Next Steps

Explore the specialized solvers in `qkrylov`:

- **[Lanczos Algorithm](solvers/lanczos.md)**: Two-pass memory-frugal ground states and lowest-$k$ eigenpairs.
- **[Davidson Algorithm](solvers/davidson.md)**: Subspace iteration with diagonal preconditioning for multiple excited states.
- **[Spectral Functions & Dynamics](solvers/dynamics.md)**: Continued-fraction expansions and correction vector methods for $S(\omega)$.
- **[Finite Temperature Lanczos (FTLM)](solvers/ftlm.md)**: Multi-temperature sweeps of thermodynamic equations of state ($Z, F, E, C_v, S$) and physical observables.
