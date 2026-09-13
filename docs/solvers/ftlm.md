# Finite-Temperature Lanczos Method (FTLM)

The Finite-Temperature Lanczos Method (FTLM) computes thermodynamic equations of state and arbitrary quantum observables ($\hat{O} \neq \hat{H}$) without requiring full diagonalization of the Hamiltonian matrix. 

By combining stochastic trace sampling over random initial vectors with Krylov subspace expansions, `qkrylov` computes thermal averages efficiently for systems with Hilbert space dimensions far exceeding the limits of full exact diagonalization.

---

## 1. Mathematical Formulation

Thermal expectation values of an observable $\hat{O}$ at inverse temperature $\beta = 1 / (k_B T)$ are given by:
$$\langle \hat{O} \rangle_\beta = \frac{\text{Tr}\left( e^{-\beta \hat{H}} \hat{O} \right)}{\text{Tr}\left( e^{-\beta \hat{H}} \right)} = \frac{1}{Z(\beta)} \text{Tr}\left( e^{-\beta \hat{H}/2} \hat{O} e^{-\beta \hat{H}/2} \right)$$

### A. Stochastic Trace Estimation
The trace over the $D$-dimensional Hilbert space is approximated by an average over $R$ random normalized vectors $\{|r\rangle\}_{r=1}^R$:
$$\text{Tr}(\hat{A}) \approx \frac{D}{R} \sum_{r=1}^R \langle r | \hat{A} | r \rangle$$

### B. Krylov Subspace Projection
For each random sample $|r\rangle$, an $M$-step Lanczos iteration constructs an orthonormal basis $\{|v_k^{(r)}\rangle\}_{k=0}^{M-1}$. The tridiagonal projection $\mathbf{T}^{(r)}$ yields eigenvalues $\{\epsilon_m^{(r)}\}$ and eigenvectors $\{y_m^{(r)}\}$.

For arbitrary quantum operators $\hat{O}$, `qkrylov` computes the projected matrix elements:
$$\mathcal{O}_{jk}^{(r)} = \langle v_j^{(r)} | \hat{O} | v_k^{(r)} \rangle$$

The thermal expectation value of $\hat{O}$ for sample $r$ is:
$$\langle \hat{O} \rangle_r(\beta) = \frac{\mathbf{c}(\beta)^\dagger \mathcal{O}^{(r)} \mathbf{c}(\beta)}{Z_r(\beta)}, \quad \text{where } c_j(\beta) = \sum_{m=0}^{M-1} (y_m^{(r)})_0 \, y_{jm}^{(r)} \, e^{-\beta \epsilon_m^{(r)} / 2}$$

### C. Low-Temperature Stability ($E_{\min}$ Shifting)
To prevent exponential overflow at low temperatures ($\beta \to \infty$), the entire spectrum across all samples is shifted by the lowest observed Ritz value $E_{\min} = \min_{r,m} \epsilon_m^{(r)}$:
$$e^{-\beta (\epsilon_m - E_{\min})}$$
Because $e^{\beta E_{\min}}$ cancels identically between the numerator and denominator, this guarantees exact invariance with zero numerical overflow.

---

## 2. Decoupled Two-Stage Architecture

`qkrylov` implements a decoupled architecture that separates computationally heavy matrix-vector multiplications from thermal temperature evaluations:

1. **Stage 1 (`ftlm_sample`)**: Runs $R$ Lanczos expansions of length $M$, diagonalizes the tridiagonal matrices, and evaluates the projected observable matrices $\mathcal{O}_{jk}$.
2. **Stage 2 (`ftlm_evaluate_sweep`)**: Decoupled from sample generation. Takes an arbitrary grid of inverse temperatures $\vec{\beta} = (\beta_1, \beta_2, \dots, \beta_K)$ and evaluates the full thermodynamic equation of state and all observable expectations with **zero additional SpMV iterations**.

---

## 3. Computed Thermodynamic Quantities

For every temperature $\beta$ on the grid, `qkrylov` computes:

| Quantity | Symbol | Formula |
| :--- | :---: | :--- |
| **Partition Function** | $Z(\beta)$ | $\frac{D}{R} \sum_{r,m} |(y_m^{(r)})_0|^2 e^{-\beta \epsilon_m^{(r)}}$ |
| **Free Energy** | $F(\beta)$ | $-\frac{1}{\beta} \ln Z(\beta)$ |
| **Internal Energy** | $E(\beta)$ | $\langle \hat{H} \rangle_\beta = -\frac{\partial \ln Z}{\partial \beta}$ |
| **Specific Heat** | $C_v(\beta)$ | $\beta^2 \left( \langle \hat{H}^2 \rangle_\beta - \langle \hat{H} \rangle_\beta^2 \right)$ |
| **Entropy** | $S(\beta)$ | $\beta \left( E(\beta) - F(\beta) \right)$ |
| **Observable Expectation** | $\langle \hat{O} \rangle_\beta$ | $\frac{1}{Z(\beta)} \frac{D}{R} \sum_r \mathbf{c}_r(\beta)^\dagger \mathcal{O}^{(r)} \mathbf{c}_r(\beta)$ |
| **Statistical Error Bar** | $\sigma_{\langle O \rangle}(\beta)$ | Sample standard error $\approx \frac{\text{std}(\langle O \rangle_r)}{\sqrt{R}}$ |

---

## 4. Multi-Language Usage Examples

=== "Julia"
    ```julia
    using QuantumKrylov

    # 1. Construct Hamiltonian and observables
    basis = SpinHalfBasis(6, Sector())
    site  = SpinHalfSite()
    op_H  = OpSum()
    op_Sz = OpSum()

    # 1D Heisenberg model
    for i in 0:5
        next_i = mod(i + 1, 6)
        add_term!(op_H, 1.0, "Sz", i, "Sz", next_i)
        add_term!(op_H, 0.5, "Sp", i, "Sm", next_i)
        add_term!(op_H, 0.5, "Sm", i, "Sp", next_i)
    end
    # Spin-spin correlation observable: S^z_0 S^z_1
    add_term!(op_Sz, 1.0, "Sz", 0, "Sz", 1)

    H   = MatrixFreeHamiltonian(basis, site, op_H)
    Sz01 = MatrixFreeHamiltonian(basis, site, op_Sz)

    # 2. Solve using SciML ThermalProblem
    betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    prob = ThermalProblem(H, betas; observables=[Sz01])
    sweep = solve(prob, FTLM(n_random=20, n_steps=50, seed=42))

    # 3. Inspect thermodynamics & observables
    for (i, b) in enumerate(sweep.beta_grid)
        println("beta = $b | E = $(sweep.internal_energies[i]) | Cv = $(sweep.specific_heats[i]) | <SzSz> = $(sweep.observable_expectations[1][i]) +/- $(sweep.observable_errors[1][i])")
    end

    # 4. Decoupled Workflow (Zero SpMV Re-evaluation)
    samples = ftlm_sample(H; observables=[Sz01], n_random=20, n_steps=50, seed=42)
    sweep_dense = ftlm_evaluate_sweep(samples, 0.1:0.05:10.0) # Zero additional SpMVs!
    sweep_sciml = solve(ThermalProblem(H, 1.0:0.5:20.0), samples)
    ```

=== "Python"
    ```python
    import numpy as np
    import qkrylov as qk

    # 1. Construct Hamiltonian and observables
    basis = qk.SpinHalfBasis(6)
    site  = qk.SpinHalfSite()
    os_H  = qk.OpSum()
    os_Sz = qk.OpSum()

    for i in range(6):
        j = (i + 1) % 6
        os_H += 1.0 * qk.Sz(i) * qk.Sz(j) + 0.5 * (qk.Sp(i) * qk.Sm(j) + qk.Sm(i) * qk.Sp(j))
    os_Sz += 1.0 * qk.Sz(0) * qk.Sz(1)

    H   = qk.MatrixFreeHamiltonian(basis, site, os_H)
    Sz01 = qk.MatrixFreeHamiltonian(basis, site, os_Sz)

    # 2. Multi-temperature sweep with observables
    betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    solver = qk.solvers.FTLM(n_random=20, n_steps=50, seed=42)
    res = solver.solve(H, betas=betas, observables=[Sz01])

    # 3. Access vectorized NumPy properties
    print("Beta grid:", res.beta_grid)
    print("Internal energies:", res.internal_energies)
    print("Heat capacities:", res.specific_heats)
    print("<Sz0 Sz1> expectation:", res.observable_expectations[0])
    print("<Sz0 Sz1> error bars:", res.observable_errors[0])

    # 4. Decoupled Workflow (Zero SpMV Re-evaluation)
    samples = solver.sample(H, observables=[Sz01])
    # Re-evaluate anywhere instantly with zero SpMVs:
    res_dense = samples.evaluate_sweep(np.linspace(0.1, 10.0, 100))
    ```

=== "C++"
    ```cpp
    #include <qkrylov/qkrylov.hpp>
    #include <iostream>

    using namespace qkrylov;
    using namespace qkrylov::fp64;

    int main() {
        SpinHalfBasis basis(6);
        SpinHalfSite site;
        OpSum op_H, op_Sz;

        for (int i = 0; i < 6; ++i) {
            int j = (i + 1) % 6;
            op_H.add_term({1.0, {{"Sz", i}, {"Sz", j}}});
            op_H.add_term({0.5, {{"Sp", i}, {"Sm", j}}});
            op_H.add_term({0.5, {{"Sm", i}, {"Sp", j}}});
        }
        op_Sz.add_term({1.0, {{"Sz", 0}, {"Sz", 1}}});

        MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H(basis, site, op_H);
        MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> Sz01(basis, site, op_Sz);

        // Stage 1: Generate Krylov subspace samples & project observables
        auto samples = ftlm_sample(H, {Sz01}, /*n_random=*/20, /*n_steps=*/50, /*seed=*/42);

        // Stage 2: Evaluate thermodynamic sweep on temperature grid
        std::vector<Real> beta_grid = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
        auto sweep = ftlm_evaluate_sweep(samples, beta_grid);

        for (size_t i = 0; i < beta_grid.size(); ++i) {
            std::cout << "beta = " << beta_grid[i]
                      << " | E = " << sweep.internal_energies[i]
                      << " | Cv = " << sweep.specific_heats[i]
                      << " | <SzSz> = " << sweep.observable_expectations[0][i]
                      << " +/- " << sweep.observable_errors[0][i] << "\n";
        }

        return 0;
    }
    ```
