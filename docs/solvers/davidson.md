# Davidson Algorithm

The **Davidson algorithm** is an iterative subspace eigensolver designed to compute multiple low-lying eigenvalues and eigenvectors ($\lambda_0, \lambda_1, \dots, \lambda_{k-1}$) simultaneously.

While standard Lanczos can suffer from spurious ghost eigenvalues or require expensive full reorthogonalization across large subspaces, Davidson expands the search subspace using **diagonal preconditioning** $(D - \theta I)^{-1} r$, accelerating convergence to target states.

---

## Key Features

- **Multiple Eigenvalues**: Solves for the lowest $k$ eigenpairs simultaneously.
- **Diagonal Preconditioning**: Leverages the matrix diagonal $D_{ii} = H_{ii}$ directly from the matrix-free representation without full matrix storage.
- **Subspace Collapsing / Restart**: Controls subspace growth with `max_subspace` to bound memory usage.
- **Polyglot API**: Fully supported across C++20, Python (`nanobind`), Julia (`QuantumKrylov.jl` via SciML `solve`), and the flat C ABI.
- **Dual Precision**: Available in FP64 (`double`) and FP32 (`float`).

---

## Mathematical Formulation

1. **Subspace Projection**: Given an orthonormal search basis $V_m = [v_1, \dots, v_m]$, construct the small subspace Hamiltonian:
   $$H_m = V_m^\dagger H V_m$$
2. **Ritz Pairs**: Diagonalize $H_m y_i = \theta_i y_i$ to obtain Ritz approximations $\tilde{u}_i = V_m y_i$.
3. **Residual Vector**: For each unconverged state $i < k$, compute:
   $$r_i = H \tilde{u}_i - \theta_i \tilde{u}_i$$
4. **Preconditioned Correction Vector**:
   $$q_i = (D - \theta_i I)^{-1} r_i$$
   where $D = \text{diag}(H)$.
5. **Orthogonalization & Expansion**: Orthonormalize $q_i$ against $V_m$ via Gram-Schmidt and expand the subspace $V_{m+1} = [V_m, q_i]$. When $m \ge \text{max\_subspace}$, collapse the subspace back to the current $k$ best Ritz vectors.

---

## Polyglot Usage Examples

=== "Python"
    ```python
    import qkrylov as qk
    from qkrylov.solvers import Davidson

    # Setup Hamiltonian (e.g. 8-site Heisenberg chain)
    basis = qk.SpinHalfBasis(8, qk.Sector())
    op = qk.OpSum()
    for i in range(7):
        op += 1.0, "Sz", i, "Sz", i + 1
        op += 0.5, "Sp", i, "Sm", i + 1
        op += 0.5, "Sm", i, "Sp", i + 1

    H = qkrylov.MatrixFreeHamiltonian(basis, qkrylov.SpinHalfSite(), op)

    # 1. Functional Interface
    res = qk.davidson_lowest(H, n_eig=3, max_subspace=20, tol=1e-8)
    print("Lowest 3 Eigenvalues:", res.eigenvalues)
    print("Iterations:", res.iterations)
    print("Converged:", res.converged)

    # Access eigenvectors (list of numpy 1D arrays)
    psi0 = res.eigenvectors[0]
    psi1 = res.eigenvectors[1]

    # 2. Object-Oriented Solver Interface
    solver = Davidson(n_eig=3, max_subspace=20, tol=1e-8)
    res_oop = solver.solve(H)
    ```

=== "Julia"
    ```julia
    using QuantumKrylov

    # Setup Hamiltonian
    sec = Sector()
    set_sz!(sec, 0)
    basis = SpinHalfBasis(8, sec)
    op = OpSum()
    for i in 0:6
        global op += 1.0 * Sz(i) * Sz(i + 1) + 0.5 * (Sp(i) * Sm(i + 1) + Sm(i) * Sp(i + 1))
    end
    H = MatrixFreeHamiltonian(basis, op)

    # 1. SciML CommonSolve Interface
    prob = ExcitedStatesProblem(H, 3)
    sol = solve(prob, Davidson(n_eig=3, max_subspace=20, tol=1e-8, compute_eigenvectors=true))
    println("Eigenvalues: ", sol.eigenvalues)
    println("Iterations:  ", sol.iterations)
    println("Converged:   ", sol.converged)

    # 2. Functional Interface
    res = davidson_lowest(H; n_eig=3, max_subspace=20, tol=1e-8)
    for (i, E) in enumerate(res.eigenvalues)
        println("E[$i] = $E")
    end
    ```

=== "C++"
    ```cpp
    #include <qkrylov/qkrylov.hpp>
    #include <iostream>

    using namespace qkrylov;

    int main() {
        auto sector = std::make_shared<Sector>();
        sector->set_sz(0);
        auto basis = std::make_shared<SpinHalfBasis>(8, sector);
        auto site = std::make_shared<SpinHalfSite>();

        OpSum op;
        for (int i = 0; i < 7; ++i) {
            op += {1.0, {{"Sz", i}, {"Sz", i + 1}}};
            op += {0.5, {{"Sp", i}, {"Sm", i + 1}}};
            op += {0.5, {{"Sm", i}, {"Sp", i + 1}}};
        }

        MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, op);

        // Compute 3 lowest eigenvalues and eigenvectors
        DavidsonResult res = davidson_lowest(H, /*n_eig=*/3, /*max_subspace=*/20, /*tol=*/1e-8);

        std::cout << "Converged: " << std::boolalpha << res.converged << " in " << res.iterations << " iterations\n";
        for (size_t i = 0; i < res.eigenvalues.size(); ++i) {
            std::cout << "  lambda[" << i << "] = " << res.eigenvalues[i] << "\n";
        }
        return 0;
    }
    ```

=== "C ABI"
    ```c
    #include <qkrylov/c_api.h>
    #include <stdio.h>
    #include <stdlib.h>

    int n_eig = 3;
    int max_subspace = 20;
    double tol = 1e-8;

    double eigenvalues[3];
    uint64_t dim = qkrylov_hamiltonian_dimension(H);
    double* eigenvectors = (double*)malloc(2 * dim * n_eig * sizeof(double));

    qkrylov_davidson_result_c_t info;
    int status = qkrylov_davidson_lowest_complex_fp64(
        H, n_eig, max_subspace, tol,
        eigenvalues, eigenvectors, &info
    );

    if (status == QKRYLOV_SUCCESS) {
        printf("Converged in %d iterations\n", info.iterations);
        for (int i = 0; i < n_eig; ++i) {
            printf("  E[%d] = %f\n", i, eigenvalues[i]);
        }
    }
    free(eigenvectors);
    ```

---

## Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_eig` | `int` | `1` | Number of lowest eigenvalues to compute. |
| `max_subspace` | `int` | `20` | Maximum dimension of search subspace before restart / collapse. Recommended: $\max(2 \times n_{\text{eig}}, 20)$. |
| `tol` | `Real` | `1e-8` | Residual 2-norm convergence tolerance $\|r_i\|_2 < \text{tol}$. |
| `compute_eigenvectors` | `bool` | `true` | (Julia/Python) Whether to compute and return full eigenvectors. |

---

## When to Use Davidson vs. Lanczos

| Scenario | Recommended Solver | Rationale |
| :--- | :--- | :--- |
| **Ground state only ($k=1$)** | [`lanczos_ground_state`](lanczos.md) | Lanczos has minimal memory footprint (3 vectors for $E_0$) and no subspace orthogonalization overhead. |
| **Lowest $k$ eigenvalues ($k \ge 2$)** | `davidson_lowest` | Davidson computes all $k$ states simultaneously with diagonal preconditioning and built-in subspace restart. |
| **Dense spectral distribution** | `davidson_lowest` | Preconditioning prevents loss of orthogonality and ghost states. |
| **Dynamical Response $S(\omega)$** | [`continued_fraction_coeffs`](dynamics.md) | Requires the tridiagonal Lanczos representation directly. |
| **Thermodynamics ($T > 0$)** | [`ftlm_sweep`](ftlm.md) | Finite Temperature Lanczos handles trace sampling over random states. |
