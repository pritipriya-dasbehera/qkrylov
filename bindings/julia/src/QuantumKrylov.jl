module QuantumKrylov

using Artifacts

const VERSION = v"0.1.0"

include("libqkrylov.jl")
include("sector.jl")
include("site.jl")
include("basis.jl")
include("opsum.jl")
include("hamiltonian.jl")
include("solvers.jl")

export Sector, set_sz!, set_hubbard_particles!, set_n!, set_nb!, get_sz, get_hubbard_particles, get_n, get_nb
export AbstractSite, SpinHalfSite, FermionSite, HubbardSite, TJSite
export AbstractBasis, SpinHalfBasis, FermionBasis, HubbardBasis, TJBasis, dimension, nsites, state, basis_index
export OpSum, add_term!, clear!, OpTerm, OpExpr, Sz, Sp, Sm, Sx, Sy, n, c, cdag, validate, validate!
export MatrixFreeHamiltonian, diagonal
export lanczos_ground_state, LanczosResult
export davidson_lowest, DavidsonResult
export continued_fraction_coeffs, ContinuedFractionResult, evaluate_spectral_function
export ftlm, FTLMResult

end
