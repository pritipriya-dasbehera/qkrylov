# Supported Physical Models

`qkrylov` supports several quantum lattice models out of the box, categorized by their local degrees of freedom and basis states. The Hilbert space for these models scales exponentially with the number of sites, making matrix-free techniques essential.

## [Spin-1/2 Models](spin.md)
**Basis:** `SpinHalfBasis`, **Site:** `SpinHalfSite`  
Used for modeling spin systems like the Heisenberg or transverse-field Ising models. The local Hilbert space dimension is 2 ($|\uparrow\rangle, |\downarrow\rangle$). Sectors can be restricted using total magnetization $S^z$.

## [General Spin-$S$ Models](spin.md#2-general-spin-s-models)
**Basis:** `SpinSBasis`, **Site:** `SpinSSite`  
Supports arbitrary local spin $S = 0.5, 1.0, 1.5, 2.0, \dots$ with local dimension $2S + 1$. Ideal for Spin-1 Haldane chains, large-spin magnetic molecules, and mixed-spin ladders.

## [Fermionic Models](fermion.md)
**Basis:** `FermionBasis`, **Site:** `FermionSite`  
Models spinless fermions with Jordan-Wigner signs. The local Hilbert space dimension is 2 ($|0\rangle, |1\rangle$). Sectors can be restricted by total particle number $N$.

## Full Hubbard Model
**Basis:** `HubbardBasis`, **Site:** `HubbardSite`  
Models spinful fermions with on-site interaction (Hubbard $U$). The local Hilbert space dimension is 4 ($|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle$). Available quantum number sectors include total spin $S^z$, total particle number $N$, and spin-resolved numbers $N_{\uparrow}$ and $N_{\downarrow}$.

## t-J Model
**Basis:** `TJBasis`, **Site:** `TJSite`  
Models strongly correlated doped antiferromagnets where double occupancy is strictly prohibited. The local Hilbert space dimension is 3 ($|0\rangle, |\uparrow\rangle, |\downarrow\rangle$). Supported sectors include $S^z$, $N$, $N_{\uparrow}$, and $N_{\downarrow}$.
