# Fermionic Models

`qkrylov` provides native matrix-free operators for spinless fermionic systems using `FermionBasis` and `FermionSite`. Jordan-Wigner anticommutation signs $\pm 1$ are computed dynamically on-the-fly during operator evaluation without full matrix storage.

---

## 1. Basis & Sectors

- **Constructor**: `FermionBasis(num_sites, sector)`
- **Local Dimension**: 2 ($|0\rangle, |1\rangle$)
- **Sector Conservation**: Total particle number $N = \sum_i c^\dagger_i c_i$. Enforced via `sector.set_n(N)`.

For spinful fermions, use the [`HubbardBasis`](index.md#full-hubbard-model) or [`TJBasis`](index.md#t-j-model).

---

## 2. Site Operators

`FermionSite` provides standard fermionic creation, annihilation, and number operators:
- `"Cdag"`: Creation operator $c^\dagger$
- `"C"`: Annihilation operator $c$
- `"n"`: Number operator $n = c^\dagger c$

Jordan-Wigner phase strings are handled automatically:
$$c^\dagger_j = \left( \prod_{k < j} (-1)^{n_k} \right) \sigma^+_j$$

---

## 3. Example: 1D Interacting Fermion Chain ($t$-$V$ Model)

Hamiltonian:
$$H = -t \sum_{i=0}^{N-2} (c^\dagger_i c_{i+1} + c^\dagger_{i+1} c_i) + V \sum_{i=0}^{N-2} n_i n_{i+1}$$

=== "Python"
    ```python
    import qkrylov as qk

    # 10 sites, half-filling (N = 5 particles)
    sec = qk.Sector()
    sec.set_n(5)
    basis = qk.FermionBasis(10, sec)
    site = qk.FermionSite()
    op = qk.OpSum()

    t = 1.0
    V = 0.5
    for i in range(9):
        # Hopping terms: -t (c^\dagger_i c_{i+1} + c^\dagger_{i+1} c_i)
        op += -t, "Cdag", i, "C", i + 1
        op += -t, "Cdag", i + 1, "C", i
        # Nearest-neighbor repulsion: V n_i n_{i+1}
        op += V, "n", i, "n", i + 1

    H = qk.MatrixFreeHamiltonian(basis, site, op)
    print(f"Hilbert space dimension: {H.dimension}")  # binom(10, 5) = 252

    res = qk.lanczos_ground_state(H)
    print(f"Ground State Energy: {res.energy:.8f}")
    ```

=== "Julia"
    ```julia
    using QuantumKrylov

    # 10 sites, 5 particles
    sec = Sector()
    set_n!(sec, 5)
    basis = FermionBasis(10, sec)
    site = FermionSite()
    op = OpSum()

    t = 1.0
    V = 0.5
    for i in 0:8
        global op += -t * cdag(i) * c(i + 1) - t * cdag(i + 1) * c(i) + V * n(i) * n(i + 1)
    end

    H = MatrixFreeHamiltonian(basis, site, op)
    res = lanczos_ground_state(H)
    println("Ground State Energy: ", res.energy)
    ```

=== "C++"
    ```cpp
    #include <qkrylov/qkrylov.hpp>
    #include <iostream>

    using namespace qkrylov;

    int main() {
        auto sec = std::make_shared<Sector>();
        sec->set_n(5);
        auto basis = std::make_shared<FermionBasis>(10, sec);
        auto site = std::make_shared<FermionSite>();

        OpSum op;
        double t = 1.0, V = 0.5;
        for (int i = 0; i < 9; ++i) {
            op += {-t, {{"Cdag", i}, {"C", i + 1}}};
            op += {-t, {{"Cdag", i + 1}, {"C", i}}};
            op += {V, {{"n", i}, {"n", i + 1}}};
        }

        MatrixFreeHamiltonian H(basis, site, op);
        auto res = lanczos_ground_state(H);
        std::cout << "Ground State Energy: " << res.energy << std::endl;
        return 0;
    }
    ```
