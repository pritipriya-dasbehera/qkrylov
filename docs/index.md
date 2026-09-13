# qkrylov

**A matrix-free quantum exact diagonalization and dynamics engine in modern C++ and Python.**

## What is qkrylov?

`qkrylov` is an open-source library designed for ultra-efficient exact diagonalization (ED) and time-evolution of quantum many-body systems. By operating entirely matrix-free, `qkrylov` enables the simulation of significantly larger quantum systems than traditional sparse-matrix approaches allow. 

The library provides a high-level, expressive interface in Python and C++ for defining quantum operators, bases, and symmetries, while delegating the heavy lifting to a highly optimized C++ core. It seamlessly handles Spin-1/2, Fermion, Hubbard, and t-J models with robust support for various symmetry sectors (such as fixed magnetization or particle number).

Whether you are computing ground states, low-lying spectra, spectral functions, or finite-temperature properties via Finite-Temperature Lanczos Methods (FTLM), `qkrylov` is built to maximize performance without compromising on user experience.

## Why qkrylov?

* **Matrix-Free Engine:** Operators are applied to states on-the-fly. The Hamiltonian matrix is *never* constructed or stored in memory.
* **Zero-Copy Memory Philosophy:** Python `numpy` arrays are passed directly to the C++ backend without any memory copying overhead.
* **SciPy Integration:** The Hamiltonian can be treated as a SciPy `LinearOperator`, unlocking access to the entire SciPy sparse linear algebra ecosystem.
* **OpenMP Parallelism:** The C++ core natively utilizes OpenMP for shared-memory parallelism, ensuring rapid operator application.
* **Comprehensive Model Support:** Built-in primitives for Spin-1/2, Fermion, Hubbard, and t-J models out of the box.

## Philosophy

`qkrylov` was born out of the need for a modern, fast, and memory-efficient quantum many-body solver that doesn't feel like a chore to use. We believe that researchers should spend their time formulating physical models rather than managing memory or writing boilerplate code. 

To this end, the library embraces a "zero-overhead" philosophy where possible, particularly across language boundaries. The Python interface is designed to be idiomatic and expressive, while the C++ backend handles the intense computational demands.

## Installation

=== "Python"
    ```bash
    pip install qkrylov
    ```

=== "C++"
    ```bash
    git clone https://github.com/sjp95/qkrylov.git
    cd qkrylov
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4
    sudo make install
    ```

=== "Julia"
    ```bash
    git clone https://github.com/sjp95/qkrylov.git
    cd qkrylov
    cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build
    julia --project=bindings/julia -e 'using Pkg; Pkg.test()'
    ```

## Quick Example: 2-Site Heisenberg Model

Here is a simple example computing the ground state energy of a 2-site anti-ferromagnetic Heisenberg model.

=== "Python"
    ```python
    import qkrylov as qk
    import numpy as np

    basis = qk.SpinHalfBasis(2, sz=0)  # 2 sites, Sz=0 sector
    site = qk.SpinHalfSite()
    ops = qk.OpSum()
    ops += (0.25, 'Sz', 0, 'Sz', 1)
    ops += (0.5, 'Sp', 0, 'Sm', 1)
    ops += (0.5, 'Sm', 0, 'Sp', 1)

    H = qk.MatrixFreeHamiltonian(basis, site, ops)
    energy, state = qk.lanczos_ground_state(H)
    print(f'Ground state energy: {energy:.6f}')  # -0.75
    ```

=== "C++"
    ```cpp
    #include <qkrylov/qkrylov.hpp>
    #include <iostream>

    int main() {
        // 2 sites, Sz=0 sector
        auto sector = std::make_shared<qkrylov::Sector>();
        sector->set_sz(0);
        auto basis = std::make_shared<qkrylov::SpinHalfBasis>(2, sector);
        auto site = std::make_shared<qkrylov::SpinHalfSite>();
        
        qkrylov::OpSum ops;
        ops += {0.25, {{"Sz", 0}, {"Sz", 1}}};
        ops += {0.5, {{"Sp", 0}, {"Sm", 1}}};
        ops += {0.5, {{"Sm", 0}, {"Sp", 1}}};

        qkrylov::MatrixFreeHamiltonian H(basis, site, ops);
        
        auto [energy, state] = qkrylov::lanczos_ground_state(H);
        std::cout << "Ground state energy: " << energy << std::endl;
        
        return 0;
    }
    ```

=== "Julia"
    ```julia
    using QuantumKrylov

    sec = Sector()
    set_sz!(sec, 0)

    basis = SpinHalfBasis(2, sec)
    site  = SpinHalfSite()

    op = OpSum()
    add_term!(op, 0.25, "Sz", 0, "Sz", 1)
    add_term!(op, 0.5, "Sp", 0, "Sm", 1)
    add_term!(op, 0.5, "Sm", 0, "Sp", 1)

    H = MatrixFreeHamiltonian(basis, site, op)
    res = lanczos_ground_state(H)
    println("Ground state energy: ", res.energy)  # -0.75
    ```

## Performance

The core advantage of `qkrylov` is its **matrix-free** architecture. In traditional sparse exact diagonalization, storing the Hamiltonian matrix requires memory that scales as $\mathcal{O}(\text{dim}^2)$ (or $\mathcal{O}(\text{dim} \times z)$ for sparse matrices, where $z$ is the number of non-zero elements per row). For large systems, this memory footprint becomes the primary bottleneck.

By applying operators directly to the state vectors on-the-fly, `qkrylov`'s memory requirement scales strictly as $\mathcal{O}(\text{dim})$—the size of the vectors themselves. This allows you to simulate systems with significantly larger Hilbert spaces on the same hardware, accelerated by OpenMP parallelism.

## Contributing

We welcome contributions! If you have a feature request, bug report, or want to contribute code, please check out our [GitHub repository](https://github.com/sjp95/qkrylov) and open an issue or pull request. 

## Citation

If you use `qkrylov` in your research, please consider citing it:

```bibtex
@software{qkrylov,
  author = {Pal, Subhajyoti and Mukhopadhyay, Aritra and Dasbehera, Pritipriya},
  title = {qkrylov: Matrix-free Krylov methods for quantum many-body physics},
  url = {https://github.com/sjp95/qkrylov}
}
```
