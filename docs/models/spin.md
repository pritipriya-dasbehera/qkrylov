# Spin Models

`qkrylov` provides native support for both spin-1/2 systems and arbitrary general spin-$S$ systems.

---

## 1. Spin-1/2 Models

The spin-1/2 framework is built around `SpinHalfBasis` and `SpinHalfSite`.

### Basis and Sectors
- **Constructor**: `SpinHalfBasis(N, sector)`
- **Local Dimension**: 2 ($|\uparrow\rangle, |\downarrow\rangle$)
- **Sector Conservation**: Total magnetization $S^z$. Controlled by `set_sz(sz2)` where `sz2 = 2 * Sz`.

| $N_{\uparrow}$ | $N_{\downarrow}$ | Total $N$ | `sz2` ($2S^z$) | $S^z$ |
| :---: | :---: | :---: | :---: | :---: |
| 4 | 0 | 4 | +4 | +2.0 |
| 3 | 1 | 4 | +2 | +1.0 |
| 2 | 2 | 4 | 0 | 0.0 |
| 1 | 3 | 4 | -2 | -1.0 |
| 0 | 4 | 4 | -4 | -2.0 |

### Site Operators
`SpinHalfSite` provides the standard spin-1/2 Pauli operator representations:
- `"Sz"`: $S^z$ operator ($+1/2, -1/2$)
- `"Sp"`: $S^+$ raising operator
- `"Sm"`: $S^-$ lowering operator
- `"Sx"`: $S^x = \frac{1}{2}(S^+ + S^-)$
- `"Sy"`: $S^y = \frac{1}{2i}(S^+ - S^-)$

---

## 2. General Spin-$S$ Models

For arbitrary spin $S = 0.5, 1.0, 1.5, 2.0, \dots$, `qkrylov` provides `SpinSBasis` and `SpinSSite`:
- **Local Dimension**: $2S + 1$
- **States**: $|S, m\rangle$ for $m = -S, -S+1, \dots, +S$
- **Example**: For $S=1$ and $N=4$, the full space has $3^4 = 81$ states; restricting to $S^z = 0$ yields $\dim = 19$.

---

## Examples

### 1D Antiferromagnetic Heisenberg Chain (Spin-1/2)
Hamiltonian: $H = J \sum_{i=0}^{N-2} \left[ S^z_i S^z_{i+1} + \frac{1}{2}(S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) \right]$

=== "Python"
    ```python
    import qkrylov as qk

    # 8 sites, Sz=0 sector
    sec = qk.Sector()
    sec.set_sz(0)
    basis = qk.SpinHalfBasis(8, sec)
    site = qk.SpinHalfSite()
    op = qk.OpSum()

    J = 1.0
    for i in range(7):
        op += J, "Sz", i, "Sz", i + 1
        op += 0.5 * J, "Sp", i, "Sm", i + 1
        op += 0.5 * J, "Sm", i, "Sp", i + 1

    H = qk.MatrixFreeHamiltonian(basis, site, op)
    res = qk.lanczos_ground_state(H)
    print(f"Ground State Energy: {res.energy:.8f}")
    ```

=== "Julia"
    ```julia
    using QuantumKrylov

    sec = Sector()
    set_sz!(sec, 0)
    basis = SpinHalfBasis(8, sec)
    op = OpSum()

    J = 1.0
    for i in 0:6
        global op += J * Sz(i) * Sz(i + 1) + 0.5 * J * (Sp(i) * Sm(i + 1) + Sm(i) * Sp(i + 1))
    end

    H = MatrixFreeHamiltonian(basis, op)
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
        sec->set_sz(0);
        auto basis = std::make_shared<SpinHalfBasis>(8, sec);
        auto site = std::make_shared<SpinHalfSite>();

        OpSum op;
        double J = 1.0;
        for (int i = 0; i < 7; ++i) {
            op += {J, {{"Sz", i}, {"Sz", i + 1}}};
            op += {0.5 * J, {{"Sp", i}, {"Sm", i + 1}}};
            op += {0.5 * J, {{"Sm", i}, {"Sp", i + 1}}};
        }

        MatrixFreeHamiltonian H(basis, site, op);
        auto res = lanczos_ground_state(H);
        std::cout << "Ground State Energy: " << res.energy << std::endl;
        return 0;
    }
    ```

---

### Spin-1 Haldane Chain (General Spin-$S$)
Hamiltonian: $H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + D \sum_i (S^z_i)^2$ with $S=1$.

=== "Python"
    ```python
    import qkrylov as qk

    # 4 sites, S=1, Sz=0 sector
    sec = qk.Sector()
    sec.set_sz(0)
    basis = qk.SpinSBasis(4, 1.0, sec)
    site = qk.SpinSSite(1.0)
    op = qk.OpSum()

    J = 1.0
    for i in range(3):
        op += J, "Sz", i, "Sz", i + 1
        op += 0.5 * J, "Sp", i, "Sm", i + 1
        op += 0.5 * J, "Sm", i, "Sp", i + 1

    H = qk.MatrixFreeHamiltonian(basis, site, op)
    print(f"Spin-1 Sz=0 Hilbert space dimension: {H.dimension}")  # 19
    res = qk.lanczos_ground_state(H)
    print(f"Ground state energy: {res.energy:.8f}")
    ```

=== "Julia"
    ```julia
    using QuantumKrylov

    sec = Sector()
    set_sz!(sec, 0)
    basis = SpinSBasis(4, 1.0; sector=sec)
    site = SpinSSite(1.0)
    op = OpSum()

    J = 1.0
    for i in 0:2
        global op += J * Sz(i) * Sz(i + 1) + 0.5 * J * (Sp(i) * Sm(i + 1) + Sm(i) * Sp(i + 1))
    end

    H = MatrixFreeHamiltonian(basis, site, op)
    res = lanczos_ground_state(H)
    println("Spin-1 Ground State Energy: ", res.energy)
    ```

=== "C++"
    ```cpp
    #include <qkrylov/qkrylov.hpp>
    #include <iostream>

    using namespace qkrylov;

    int main() {
        auto sec = std::make_shared<Sector>();
        sec->set_sz(0);
        auto basis = std::make_shared<SpinSBasis>(4, 1.0, sec);
        auto site = std::make_shared<SpinSSite>(1.0);

        OpSum op;
        for (int i = 0; i < 3; ++i) {
            op += {1.0, {{"Sz", i}, {"Sz", i + 1}}};
            op += {0.5, {{"Sp", i}, {"Sm", i + 1}}};
            op += {0.5, {{"Sm", i}, {"Sp", i + 1}}};
        }

        MatrixFreeHamiltonian H(basis, site, op);
        auto res = lanczos_ground_state(H);
        std::cout << "Spin-1 Ground State: " << res.energy << std::endl;
        return 0;
    }
    ```
