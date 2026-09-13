# Lanczos Eigensolver

The Lanczos algorithm is an iterative Krylov subspace method designed to find extremal eigenvalues and eigenvectors of large, sparse Hermitian quantum systems without building the dense $D \times D$ Hamiltonian in memory.

In `qkrylov`, the Lanczos solver is unified into a modern C++20 engine `solvers::lanczos<Policy, ExecSpace>` governed by compile-time execution policies that select optimal memory bounds and orthogonalization strategies.

---

## 1. Compile-Time Policies

`qkrylov` provides 4 specialized policies defined in `qkrylov::solvers::policy`:

| Policy | Memory Bound | State Type | Description |
| :--- | :---: | :---: | :--- |
| **`OnePass`** | $\mathcal{O}(D)$ | Ground state only (energy) | Keeps only 3 Krylov vectors in flight. Ideal when only the ground-state energy is required. |
| **`OnePass_DKGS`** | $\mathcal{O}(m \cdot D)$ | $k$ low-lying states | Implements Daniel-Gragg-Kaufman-Stewart (DKGS) full reorthogonalization to compute $k$ lowest eigenvalues and eigenvectors without spurious ghost eigenvalues. |
| **`OnePass_full`** | $\mathcal{O}(m \cdot D)$ | Ground state (with wavefunction) | Retains the Krylov basis in memory to reconstruct the ground-state eigenvector in a single pass. |
| **`TwoPass`** | $\mathcal{O}(D)$ | Ground state (with wavefunction) | Minimal memory bound. Pass 1 finds tridiagonal eigenvalues. Pass 2 reconstructs the eigenvector without storing the Krylov basis. |

---

## 2. Configuration & Return Types

### C++ `LanczosConfig`
```cpp
struct LanczosConfig {
    int n_eig = 1;                               // Number of lowest eigenpairs to compute (OnePass_DKGS)
    int maxiter = 200;                           // Maximum Krylov iterations
    int min_iterations = 1;                      // Minimum iterations before checking convergence
    int check_interval = 1;                      // Convergence check stride
    Real tol = Real(1.0e-12);                    // Convergence tolerance
    Real breakdown_tol = Real(0.0);              // Subspace breakdown tolerance (0 => 4 * eps)
    std::optional<uint64_t> seed = std::nullopt; // Deterministic PRNG seed
    HostVector initial_vector = {};              // Warm-starting trial vector
    bool compute_eigenvectors = true;            // Compute state vectors when policy supports it
};
```

### Structured Bindings
`LanczosResult` supports C++17 structured bindings:
```cpp
auto [energy, psi0, iterations, converged] = solvers::lanczos<solvers::policy::OnePass_full>(H, cfg);
```

---

## 3. Multi-Language Usage Examples

=== "Julia"
    ```julia
    using QuantumKrylov

    # 1. Define 8-site Heisenberg model
    sec = Sector()
    set_sz!(sec, 0)
    basis = SpinHalfBasis(8, sec)
    site  = SpinHalfSite()
    op    = OpSum()

    for i in 0:7
        next_i = mod(i + 1, 8)
        add_term!(op, 1.0, "Sz", i, "Sz", next_i)
        add_term!(op, 0.5, "Sp", i, "Sm", next_i)
        add_term!(op, 0.5, "Sm", i, "Sp", next_i)
    end
    H = MatrixFreeHamiltonian(basis, site, op)

    # 2. SciML Problem-Algorithm Interface
    # Ground State with One-Pass
    prob = GroundStateProblem(H)
    sol = solve(prob, Lanczos(variation=OnePass(), maxiter=100, tol=1e-12, return_state=true))
    println("Ground state energy: ", sol.energy)
    println("Wavefunction length: ", length(sol.eigenvector))

    # Low-memory Two-Pass for large Hilbert spaces
    sol_tp = solve(prob, Lanczos(variation=TwoPass(), maxiter=100))
    println("Two-pass energy: ", sol_tp.energy)

    # Multiple low-lying excited states via DKGS
    exc_prob = ExcitedStatesProblem(H, 3)
    sol_lowest = solve(exc_prob, Lanczos(maxiter=100, return_state=true))
    println("Lowest 3 energies: ", sol_lowest.eigenvalues)
    ```

=== "Python"
    ```python
    import numpy as np
    import qkrylov as qk

    # 1. Define 8-site Heisenberg model
    basis = qk.SpinHalfBasis(8, sz=0)
    site  = qk.SpinHalfSite()
    op    = qk.OpSum()

    for i in range(8):
        j = (i + 1) % 8
        op += 1.0 * qk.Sz(i) * qk.Sz(j) + 0.5 * (qk.Sp(i) * qk.Sm(j) + qk.Sm(i) * qk.Sp(j))
    H = qk.MatrixFreeHamiltonian(basis, site, op)

    # 2. Single-Pass Ground State (OOP interface)
    solver = qk.solvers.Lanczos(maxiter=100, tol=1e-12)
    res = solver.solve(H)
    print(f"Ground State Energy: {res.energy:.8f}")
    print(f"Iterations: {res.iterations}, Converged: {res.converged}")

    # Tuple unpacking
    e0, psi0 = solver.solve(H)

    # 3. Two-Pass Ground State (minimal memory bound)
    solver_tp = qk.solvers.LanczosTwoPass(maxiter=100)
    e0_tp, psi0_tp = solver_tp.solve(H)
    assert np.isclose(e0, e0_tp)
    ```

=== "C++"
    ```cpp
    #include <qkrylov/qkrylov.hpp>
    #include <iostream>

    using namespace qkrylov;
    using namespace qkrylov::fp64;

    int main() {
        Sector sec;
        sec.use_sz = true;
        sec.sz2 = 0;

        SpinHalfBasis basis(8, sec);
        SpinHalfSite site;
        OpSum op;

        for (int i = 0; i < 8; ++i) {
            int j = (i + 1) % 8;
            op.add_term({1.0, {{"Sz", i}, {"Sz", j}}});
            op.add_term({0.5, {{"Sp", i}, {"Sm", j}}});
            op.add_term({0.5, {{"Sm", i}, {"Sp", j}}});
        }

        MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, op);

        // Ground-state with OnePass_full policy
        LanczosConfig cfg;
        cfg.maxiter = 100;
        cfg.tol = 1e-12;

        auto [e0, psi0, iters, conv] = solvers::lanczos<solvers::policy::OnePass_full>(H, cfg);
        std::cout << "Energy: " << e0 << " after " << iters << " iterations (converged: " << conv << ")\n";

        // Multiple low-lying eigenvalues with OnePass_DKGS
        LanczosConfig dkgs_cfg;
        dkgs_cfg.n_eig = 3;
        dkgs_cfg.maxiter = 100;
        auto res_dkgs = solvers::lanczos<solvers::policy::OnePass_DKGS>(H, dkgs_cfg);
        for (size_t k = 0; k < res_dkgs.eigenvalues.size(); ++k) {
            std::cout << "Level " << k << ": " << res_dkgs.eigenvalues[k] << "\n";
        }

        return 0;
    }
    ```
