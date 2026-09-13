# Solvers

`qkrylov` includes several highly optimized, matrix-free solvers tailored for quantum many-body systems across CPU and GPU hardware backends:

- **[Lanczos Algorithm](lanczos.md)**: Compile-time policy eigensolver (`OnePass`, `OnePass_DKGS`, `OnePass_full`, `TwoPass`) for ground-state energy, wavefunctions, and low-lying excited states.
- **[Davidson Algorithm](davidson.md)**: Subspace expansion with diagonal preconditioning for simultaneous computation of multiple clustered eigenvalues and eigenvectors.
- **[Spectral Functions & Dynamics](dynamics.md)**: Dynamical response functions $A(\omega)$ and Green's functions using continued-fraction expansions and the correction vector method.
- **[Finite Temperature Lanczos Method (FTLM)](ftlm.md)**: Decoupled two-stage stochastic Krylov sampling for full thermodynamic equations of state ($Z, F, E, C_v, S$) and arbitrary quantum observables ($\hat{O} \neq \hat{H}$) across multi-temperature grids with zero-cost re-evaluation.
