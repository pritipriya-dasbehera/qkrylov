# Spectral Functions & Dynamics

Spectral functions reveal the dynamical and excited-state properties of quantum systems, such as the single-particle density of states (DOS), dynamical spin structure factor $S(q, \omega)$, and optical conductivity.

`qkrylov` provides two complementary matrix-free solvers for computing dynamical response functions:

1. **Continued Fraction Method**: Projects the Hamiltonian onto a single Krylov subspace to construct a continued-fraction expansion. Highly efficient for continuous broad-spectrum scans across wide frequency windows.
2. **Correction Vector Method**: Solves the complex-shifted linear system $(H - E_0 - \omega - i\eta) |x\rangle = \hat{O} |\psi_0\rangle$ directly. Delivers asymptotically exact spectral resolution at targeted frequencies $\omega$ without Krylov truncation error.

---

## 1. Continued Fraction Method

Given a ground state $|\psi_0\rangle$ and a local perturbation operator $\hat{O}$, define the perturbed state $|\phi_0\rangle = \hat{O} |\psi_0\rangle$. The zero-temperature dynamical correlation function is:

$$I(\omega) = -\frac{1}{\pi} \Im \left\langle \phi_0 \left| \frac{1}{\omega + E_0 - H + i\eta} \right| \phi_0 \right\rangle$$

Applying the Lanczos iteration starting from $|\phi_0\rangle$ yields the tridiagonal matrix representation with diagonal $\alpha_n$ and off-diagonal $\beta_n$. The Green's function is then evaluated via the continued fraction:

$$G(z) = \frac{\langle \phi_0 | \phi_0 \rangle}{z - \alpha_0 - \frac{\beta_1^2}{z - \alpha_1 - \frac{\beta_2^2}{z - \alpha_2 - \dots}}}$$

where $z = \omega + E_0 + i\eta$ and $\eta > 0$ is a Lorentzian broadening factor.

---

## 2. Correction Vector Method

The **correction vector** $|x(\omega, \eta)\rangle$ is defined as the solution to:

$$(H - E_0 - \omega - i\eta) |x\rangle = |\phi_0\rangle$$

The spectral function is obtained directly from the imaginary part of the inner product:

$$I(\omega) = -\frac{1}{\pi} \Im \langle \phi_0 | x(\omega, \eta) \rangle = \frac{\eta}{\pi} \langle x(\omega, \eta) | x(\omega, \eta) \rangle$$

In `qkrylov`, this linear system is solved iteratively using matrix-free conjugate gradient / Krylov subspace solvers without ever constructing the matrix.

---

## Polyglot Usage Examples

### Continued Fraction Expansion

=== "Python"
    ```python
    import numpy as np
    import qkrylov as qk
    from qkrylov.solvers import ContinuedFraction

    # 1. Compute ground state
    E0, psi0 = qk.lanczos_ground_state(H)

    # 2. Prepare perturbation phi0 = O |psi0>
    # (e.g. apply Sz on site 0 using operator or matrix-free apply)
    phi0 = apply_op(psi0)

    # 3. Generate continued fraction coefficients (alphas, betas)
    cfr = qk.continued_fraction_coeffs(H, phi0, n_iter=100)
    print("Computed", len(cfr.alphas), "Krylov coefficients")

    # 4. Evaluate spectral function across frequency grid
    omegas = np.linspace(0.0, 5.0, 500)
    eta = 0.05
    A_w = [
        qk.evaluate_spectral_function(cfr.alphas, cfr.betas, cfr.norm_phi0, w, E0, eta)
        for w in omegas
    ]
    ```

=== "Julia"
    ```julia
    using QuantumKrylov

    # 1. Ground state calculation
    res_gs = lanczos_ground_state(H; return_state=true)
    E0, psi0 = res_gs.energy, res_gs.state

    # 2. Perturbed state phi0 = O * psi0
    phi0 = apply_op(psi0)

    # 3. SciML CommonSolve Interface
    prob = DynamicalProblem(H, phi0; e0=E0)
    cfr = solve(prob, ContinuedFraction(n_iter=100))

    # 4. Evaluate spectral function across frequencies
    omegas = range(0.0, 5.0; length=500)
    eta = 0.05
    A_w = [evaluate_spectral_function(cfr, w, E0, eta) for w in omegas]
    ```

=== "C++"
    ```cpp
    #include <qkrylov/qkrylov.hpp>
    #include <qkrylov/solvers/dynamics.hpp>
    #include <vector>
    #include <iostream>

    using namespace qkrylov;

    // Given MatrixFreeHamiltonian H, ground energy E0, and perturbed vector phi0:
    DynamicsResult cfr = continued_fraction_coeffs(H, phi0, /*n_iter=*/100);

    double eta = 0.05;
    for (double omega = 0.0; omega <= 5.0; omega += 0.05) {
        double A = evaluate_spectral_function(
            cfr.alphas.data(), cfr.betas.data(),
            cfr.alphas.size(), cfr.norm_phi0,
            omega, E0, eta
        );
        std::cout << omega << " " << A << "\n";
    }
    ```

=== "C ABI"
    ```c
    #include <qkrylov/c_api.h>
    #include <stdlib.h>
    #include <stdio.h>

    int n_iter = 100;
    double alphas[100];
    double betas[100];
    double norm_phi0 = 0.0;
    int num_coeffs = 0;

    int status = qkrylov_continued_fraction_coeffs_complex_fp64(
        H, phi0_complex, n_iter,
        alphas, betas, &norm_phi0, &num_coeffs
    );

    if (status == QKRYLOV_SUCCESS) {
        double A = qkrylov_evaluate_spectral_function_fp64(
            alphas, betas, num_coeffs,
            norm_phi0, /*omega=*/1.0, /*E0=*/-2.0, /*eta=*/0.05
        );
        printf("A(omega=1.0) = %f\n", A);
    }
    ```

---

### Correction Vector Method

=== "Python"
    ```python
    import qkrylov as qk
    from qkrylov.solvers import CorrectionVector

    # 1. Functional Interface
    res = qk.correction_vector(
        H, phi0, E0=E0, omega=1.5, eta=0.05, max_iter=500, tol=1e-8
    )
    print("Spectral function at omega=1.5:", res.spectral_function)
    print("Iterations:", res.iterations)
    print("Converged:", res.converged)

    # Access full correction vector state:
    x_vec = res.correction_vector  # 1D numpy array

    # 2. OOP Solver Interface
    solver = CorrectionVector(max_iter=500, tol=1e-8)
    res_oop = solver.solve(H, phi0, E0=E0, omega=1.5, eta=0.05)
    ```

=== "Julia"
    ```julia
    using QuantumKrylov

    # 1. SciML CommonSolve Interface
    prob = CorrectionVectorProblem(H, phi0; e0=E0, omega=1.5, eta=0.05)
    sol = solve(prob, CorrectionVector(maxiter=500, tol=1e-8, return_vector=true))
    println("Spectral function: ", sol.spectral_function)
    println("Iterations:        ", sol.iterations)
    println("Correction vector: ", sol.vector)

    # 2. Functional Interface
    res = solver_correction_vector(
        H, phi0; e0=E0, omega=1.5, eta=0.05, maxiter=500, tol=1e-8, return_vector=true
    )
    ```

=== "C++"
    ```cpp
    #include <qkrylov/solvers/correction_vector.hpp>
    #include <iostream>

    using namespace qkrylov;

    CorrectionVectorResult res = correction_vector_spectral(
        H, phi0, /*E0=*/E0, /*omega=*/1.5, /*eta=*/0.05,
        /*max_iter=*/500, /*tol=*/1e-8
    );

    std::cout << "A(omega=1.5) = " << res.spectral_function << "\n";
    std::cout << "Converged: " << std::boolalpha << res.converged << " in " << res.iterations << " iters\n";
    ```

=== "C ABI"
    ```c
    #include <qkrylov/c_api.h>
    #include <stdio.h>
    #include <stdlib.h>

    uint64_t dim = qkrylov_hamiltonian_dimension(H);
    double* corr_vec = (double*)malloc(2 * dim * sizeof(double));
    qkrylov_correction_vector_result_c_t res;

    int status = qkrylov_solver_correction_vector_fp64(
        H, phi0_complex, /*e0=*/E0, /*omega=*/1.5, /*eta=*/0.05,
        /*max_iter=*/500, /*tol=*/1e-8, &res, corr_vec
    );

    if (status == QKRYLOV_SUCCESS) {
        printf("Spectral weight: %f\n", res.spectral_function);
        printf("Converged: %d (iterations: %d)\n", res.converged, res.iterations);
    }
    free(corr_vec);
    ```

---

## Comparison: Continued Fraction vs. Correction Vector

| Feature | Continued Fraction | Correction Vector |
| :--- | :--- | :--- |
| **Primary Advantage** | Evaluates hundreds of $\omega$ points in milliseconds once $\alpha, \beta$ are computed | High precision at individual sharp peaks, bounds error rigorously |
| **Computational Cost** | $O(N_{\text{iter}})$ matrix-vector multiplications total for all $\omega$ | $O(N_{\text{iter}})$ matrix-vector multiplications **per frequency point $\omega$** |
| **Broadening Parameter ($\eta$)** | Post-processing parameter; can be varied without recomputing Krylov steps | Must be chosen a priori before solving the linear system |
| **Accuracy** | Good for broad continuous spectra; can suffer from ghost peaks at large $N_{\text{iter}}$ | Exact within solver convergence tolerance `tol` |
| **Recommended Workflow** | Use Continued Fraction for coarse wide-band spectrum scans $\rightarrow$ refine isolated features with Correction Vector |
