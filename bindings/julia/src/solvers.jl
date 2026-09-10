# Lanczos solver wrappers

struct LanczosResultFP32C
    energy::Cfloat
    iterations::Cint
    converged::Cint
end

struct LanczosResultFP64C
    energy::Cdouble
    iterations::Cint
    converged::Cint
end

struct DavidsonResultC
    iterations::Cint
    converged::Cint
end

struct FTLMResultFP32C
    beta::Cfloat
    partition_function::Cfloat
    internal_energy::Cfloat
    specific_heat::Cfloat
end

struct FTLMResultFP64C
    beta::Cdouble
    partition_function::Cdouble
    internal_energy::Cdouble
    specific_heat::Cdouble
end

struct CorrectionVectorResultFP32C
    spectral_function::Cfloat
    iterations::Cint
    converged::Cint
end

struct CorrectionVectorResultFP64C
    spectral_function::Cdouble
    iterations::Cint
    converged::Cint
end

struct LanczosResult{T<:Real}
    energy::T
    iterations::Int
    converged::Bool
    _state::Union{Vector{Complex{T}}, Nothing}
    has_state::Bool

    function LanczosResult(energy::T, iterations::Integer, converged::Bool, state::Union{Vector{Complex{T}}, Nothing}=nothing) where {T<:Real}
        return new{T}(energy, Int(iterations), converged, state, state !== nothing)
    end
end

function Base.getproperty(res::LanczosResult, sym::Symbol)
    if sym === :state || sym === :eigenvector || sym === :vector
        if !getfield(res, :has_state) || getfield(res, :_state) === nothing
            error("Ground state wavefunction was not computed. Pass `return_state=true` to `lanczos_ground_state` to compute the state vector.")
        end
        return getfield(res, :_state)
    end
    return getfield(res, sym)
end

function Base.propertynames(res::LanczosResult, private::Bool=false)
    return private ? fieldnames(LanczosResult) : (:energy, :iterations, :converged, :state, :eigenvector)
end

function Base.iterate(res::LanczosResult, state=1)
    if state == 1
        return (res.energy, 2)
    elseif state == 2
        if !res.has_state
            error("Ground state wavefunction was not computed. Pass `return_state=true` to `lanczos_ground_state` to compute the state vector.")
        end
        return (res.state, 3)
    else
        return nothing
    end
end

function lanczos_ground_state(
    H::MatrixFreeHamiltonian{Float64};
    maxiter::Integer=100,
    tol::Real=1e-12,
    return_state::Bool=false,
    compute_eigenvector::Bool=return_state
)::LanczosResult{Float64}
    dim = Int(dimension(H))
    should_compute = return_state || compute_eigenvector
    res_c = Ref{LanczosResultFP64C}(LanczosResultFP64C(0.0, 0, 0))

    if should_compute
        psi = Vector{ComplexF64}(undef, dim)

        GC.@preserve psi begin
            status = ccall(
                (:qkrylov_lanczos_ground_state_complex_fp64, libqkrylov),
                Cint,
                (Ptr{Cvoid}, Cint, Cdouble, Ref{LanczosResultFP64C}, Ptr{Cdouble}),
                H.ptr, Cint(maxiter), Cdouble(tol), res_c, pointer(psi)
            )
        end
        _check_status(status, "Lanczos ground state solver failed")
        return LanczosResult(res_c[].energy, Int(res_c[].iterations), res_c[].converged != 0, psi)
    else
        status = ccall(
            (:qkrylov_lanczos_ground_state_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cdouble, Ref{LanczosResultFP64C}),
            H.ptr, Cint(maxiter), Cdouble(tol), res_c
        )
        _check_status(status, "Lanczos ground state solver failed")
        return LanczosResult(res_c[].energy, Int(res_c[].iterations), res_c[].converged != 0, nothing)
    end
end

function lanczos_ground_state(
    H::MatrixFreeHamiltonian{Float32};
    maxiter::Integer=100,
    tol::Real=1e-6,
    return_state::Bool=false,
    compute_eigenvector::Bool=return_state
)::LanczosResult{Float32}
    dim = Int(dimension(H))
    should_compute = return_state || compute_eigenvector
    res_c = Ref{LanczosResultFP32C}(LanczosResultFP32C(0.0f0, 0, 0))

    if should_compute
        psi = Vector{ComplexF32}(undef, dim)

        GC.@preserve psi begin
            status = ccall(
                (:qkrylov_lanczos_ground_state_complex_fp32, libqkrylov),
                Cint,
                (Ptr{Cvoid}, Cint, Cfloat, Ref{LanczosResultFP32C}, Ptr{Cfloat}),
                H.ptr, Cint(maxiter), Cfloat(tol), res_c, pointer(psi)
            )
        end
        _check_status(status, "Lanczos ground state solver failed")
        return LanczosResult(res_c[].energy, Int(res_c[].iterations), res_c[].converged != 0, psi)
    else
        status = ccall(
            (:qkrylov_lanczos_ground_state_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cfloat, Ref{LanczosResultFP32C}),
            H.ptr, Cint(maxiter), Cfloat(tol), res_c
        )
        _check_status(status, "Lanczos ground state solver failed")
        return LanczosResult(res_c[].energy, Int(res_c[].iterations), res_c[].converged != 0, nothing)
    end
end

# Davidson Preconditioned Eigensolver
struct DavidsonResult{T<:Real}
    eigenvalues::Vector{T}
    eigenvectors::Union{Vector{Vector{Complex{T}}}, Nothing}
    iterations::Int
    converged::Bool
end

function Base.getproperty(res::DavidsonResult{T}, sym::Symbol) where {T}
    if sym === :values || sym === :evals
        return getfield(res, :eigenvalues)
    elseif sym === :vectors || sym === :evecs
        vecs = getfield(res, :eigenvectors)
        if vecs === nothing
            error("Eigenvectors were not computed for this Davidson run. Pass `compute_eigenvectors=true` to `davidson_lowest`.")
        end
        return vecs
    end
    return getfield(res, sym)
end

function Base.propertynames(res::DavidsonResult, private::Bool=false)
    return private ? fieldnames(DavidsonResult) : (:eigenvalues, :eigenvectors, :iterations, :converged, :values, :evals, :vectors, :evecs)
end

function davidson_lowest(
    H::MatrixFreeHamiltonian{Float64};
    n_eig::Integer=1,
    max_subspace::Integer=20,
    tol::Real=1e-8,
    compute_eigenvectors::Bool=true
)::DavidsonResult{Float64}
    dim = Int(dimension(H))
    evals = Vector{Float64}(undef, n_eig)
    evecs_flat = compute_eigenvectors ? Vector{ComplexF64}(undef, n_eig * dim) : ComplexF64[]
    dav_c = Ref{DavidsonResultC}(DavidsonResultC(0, 0))

    GC.@preserve evals evecs_flat begin
        evecs_ptr = compute_eigenvectors ? pointer(evecs_flat) : Ptr{ComplexF64}(C_NULL)
        status = ccall(
            (:qkrylov_davidson_lowest_complex_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cint, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Ref{DavidsonResultC}),
            H.ptr, Cint(n_eig), Cint(max_subspace), Cdouble(tol), pointer(evals), Ptr{Cdouble}(evecs_ptr), dav_c
        )
        _check_status(status, "Davidson solver failed")
    end

    iters = Int(dav_c[].iterations)
    conv  = dav_c[].converged != 0

    if !compute_eigenvectors
        return DavidsonResult{Float64}(evals, nothing, iters, conv)
    end

    evecs = Vector{Vector{ComplexF64}}(undef, n_eig)
    for idx in 1:n_eig
        evecs[idx] = evecs_flat[(idx-1)*dim + 1 : idx*dim]
    end
    return DavidsonResult{Float64}(evals, evecs, iters, conv)
end

function davidson_lowest(
    H::MatrixFreeHamiltonian{Float32};
    n_eig::Integer=1,
    max_subspace::Integer=20,
    tol::Real=1e-5,
    compute_eigenvectors::Bool=true
)::DavidsonResult{Float32}
    dim = Int(dimension(H))
    evals = Vector{Float32}(undef, n_eig)
    evecs_flat = compute_eigenvectors ? Vector{ComplexF32}(undef, n_eig * dim) : ComplexF32[]
    dav_c = Ref{DavidsonResultC}(DavidsonResultC(0, 0))

    GC.@preserve evals evecs_flat begin
        evecs_ptr = compute_eigenvectors ? pointer(evecs_flat) : Ptr{ComplexF32}(C_NULL)
        status = ccall(
            (:qkrylov_davidson_lowest_complex_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Cint, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Ref{DavidsonResultC}),
            H.ptr, Cint(n_eig), Cint(max_subspace), Cfloat(tol), pointer(evals), Ptr{Cfloat}(evecs_ptr), dav_c
        )
        _check_status(status, "Davidson solver failed")
    end

    iters = Int(dav_c[].iterations)
    conv  = dav_c[].converged != 0

    if !compute_eigenvectors
        return DavidsonResult{Float32}(evals, nothing, iters, conv)
    end

    evecs = Vector{Vector{ComplexF32}}(undef, n_eig)
    for idx in 1:n_eig
        evecs[idx] = evecs_flat[(idx-1)*dim + 1 : idx*dim]
    end
    return DavidsonResult{Float32}(evals, evecs, iters, conv)
end

# Dynamics & Spectral Function
struct ContinuedFractionResult{T<:Real}
    alphas::Vector{T}
    betas::Vector{T}
    norm_phi0::T
end

function continued_fraction_coeffs(
    H::MatrixFreeHamiltonian{Float64},
    phi0::AbstractVector{<:Number};
    n_iter::Integer=100
)::ContinuedFractionResult{Float64}
    dim = Int(dimension(H))
    @assert length(phi0) == dim "Initial vector phi0 size $(length(phi0)) does not match Hamiltonian dimension $dim"

    phi0_c = (phi0 isa Vector{ComplexF64}) ? phi0 : Vector{ComplexF64}(phi0)
    alphas_buf = Vector{Float64}(undef, n_iter)
    betas_buf  = Vector{Float64}(undef, n_iter)
    norm_ref   = Ref{Cdouble}(0.0)
    num_coeffs = Ref{Cint}(0)

    GC.@preserve phi0_c alphas_buf betas_buf begin
        status = ccall(
            (:qkrylov_continued_fraction_coeffs_complex_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Cint}),
            H.ptr, pointer(phi0_c), Cint(n_iter), pointer(alphas_buf), pointer(betas_buf), norm_ref, num_coeffs
        )
        _check_status(status, "Continued fraction solver failed")
    end

    k = Int(num_coeffs[])
    alphas = alphas_buf[1:k]
    betas  = betas_buf[1:max(0, k - 1)]
    return ContinuedFractionResult{Float64}(alphas, betas, norm_ref[])
end

function continued_fraction_coeffs(
    H::MatrixFreeHamiltonian{Float32},
    phi0::AbstractVector{<:Number};
    n_iter::Integer=100
)::ContinuedFractionResult{Float32}
    dim = Int(dimension(H))
    @assert length(phi0) == dim "Initial vector phi0 size $(length(phi0)) does not match Hamiltonian dimension $dim"

    phi0_c = (phi0 isa Vector{ComplexF32}) ? phi0 : Vector{ComplexF32}(phi0)
    alphas_buf = Vector{Float32}(undef, n_iter)
    betas_buf  = Vector{Float32}(undef, n_iter)
    norm_ref   = Ref{Cfloat}(0.0f0)
    num_coeffs = Ref{Cint}(0)

    GC.@preserve phi0_c alphas_buf betas_buf begin
        status = ccall(
            (:qkrylov_continued_fraction_coeffs_complex_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ref{Cfloat}, Ref{Cint}),
            H.ptr, pointer(phi0_c), Cint(n_iter), pointer(alphas_buf), pointer(betas_buf), norm_ref, num_coeffs
        )
        _check_status(status, "Continued fraction solver failed")
    end

    k = Int(num_coeffs[])
    alphas = alphas_buf[1:k]
    betas  = betas_buf[1:max(0, k - 1)]
    return ContinuedFractionResult{Float32}(alphas, betas, norm_ref[])
end

function evaluate_spectral_function(
    cfr::ContinuedFractionResult{T},
    omega::Real,
    E0::Real,
    eta::Real
)::T where {T<:Real}
    return evaluate_spectral_function(cfr.alphas, cfr.betas, cfr.norm_phi0, omega, E0, eta)
end

function evaluate_spectral_function(
    alphas::AbstractVector{Float64},
    betas::AbstractVector{Float64},
    norm_phi0::Real,
    omega::Real,
    E0::Real,
    eta::Real
)::Float64
    alphas_buf = (alphas isa Vector{Float64}) ? alphas : Vector{Float64}(alphas)
    betas_buf  = (betas isa Vector{Float64}) ? betas : Vector{Float64}(betas)
    n = length(alphas_buf)

    GC.@preserve alphas_buf betas_buf begin
        val = ccall(
            (:qkrylov_evaluate_spectral_function_fp64, libqkrylov),
            Cdouble,
            (Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Cdouble, Cdouble, Cdouble, Cdouble),
            pointer(alphas_buf), pointer(betas_buf), Csize_t(n),
            Cdouble(norm_phi0), Cdouble(omega), Cdouble(E0), Cdouble(eta)
        )
    end
    return val
end

function evaluate_spectral_function(
    alphas::AbstractVector{Float32},
    betas::AbstractVector{Float32},
    norm_phi0::Real,
    omega::Real,
    E0::Real,
    eta::Real
)::Float32
    alphas_buf = (alphas isa Vector{Float32}) ? alphas : Vector{Float32}(alphas)
    betas_buf  = (betas isa Vector{Float32}) ? betas : Vector{Float32}(betas)
    n = length(alphas_buf)

    GC.@preserve alphas_buf betas_buf begin
        val = ccall(
            (:qkrylov_evaluate_spectral_function_fp32, libqkrylov),
            Cfloat,
            (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Cfloat, Cfloat, Cfloat, Cfloat),
            pointer(alphas_buf), pointer(betas_buf), Csize_t(n),
            Cfloat(norm_phi0), Cfloat(omega), Cfloat(E0), Cfloat(eta)
        )
    end
    return val
end

function evaluate_spectral_function(
    alphas::AbstractVector{<:Real},
    betas::AbstractVector{<:Real},
    norm_phi0::Real,
    omega::Real,
    E0::Real,
    eta::Real
)::Float64
    return evaluate_spectral_function(Vector{Float64}(alphas), Vector{Float64}(betas), Float64(norm_phi0), Float64(omega), Float64(E0), Float64(eta))
end

# FTLM (Finite Temperature Lanczos) Solver
struct FTLMResult{T<:Real}
    beta::T
    partition_function::T
    internal_energy::T
    specific_heat::T
end

function ftlm(
    H::MatrixFreeHamiltonian{Float64};
    beta::Real=1.0,
    n_random::Integer=10,
    n_steps::Integer=50
)::FTLMResult{Float64}
    res_c = Ref{FTLMResultFP64C}(FTLMResultFP64C(0.0, 0.0, 0.0, 0.0))
    status = ccall(
        (:qkrylov_ftlm_fp64, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cdouble, Cint, Cint, Ref{FTLMResultFP64C}),
        H.ptr, Cdouble(beta), Cint(n_random), Cint(n_steps), res_c
    )
    _check_status(status, "FTLM solver failed")
    return FTLMResult{Float64}(
        res_c[].beta,
        res_c[].partition_function,
        res_c[].internal_energy,
        res_c[].specific_heat
    )
end

function ftlm(
    H::MatrixFreeHamiltonian{Float32};
    beta::Real=1.0,
    n_random::Integer=10,
    n_steps::Integer=50
)::FTLMResult{Float32}
    res_c = Ref{FTLMResultFP32C}(FTLMResultFP32C(0.0f0, 0.0f0, 0.0f0, 0.0f0))
    status = ccall(
        (:qkrylov_ftlm_fp32, libqkrylov),
        Cint,
        (Ptr{Cvoid}, Cfloat, Cint, Cint, Ref{FTLMResultFP32C}),
        H.ptr, Cfloat(beta), Cint(n_random), Cint(n_steps), res_c
    )
    _check_status(status, "FTLM solver failed")
    return FTLMResult{Float32}(
        res_c[].beta,
        res_c[].partition_function,
        res_c[].internal_energy,
        res_c[].specific_heat
    )
end

# Correction Vector Spectroscopy Solver
struct CorrectionVectorResult{T<:Real}
    spectral_function::T
    iterations::Int
    converged::Bool
    _vector::Union{Vector{Complex{T}}, Nothing}
    has_vector::Bool

    function CorrectionVectorResult(spec::T, iterations::Integer, converged::Bool, vec::Union{Vector{Complex{T}}, Nothing}=nothing) where {T<:Real}
        return new{T}(spec, Int(iterations), converged, vec, vec !== nothing)
    end
end

function Base.getproperty(res::CorrectionVectorResult, sym::Symbol)
    if sym === :vector || sym === :correction_vector
        if !getfield(res, :has_vector) || getfield(res, :_vector) === nothing
            error("Correction vector was not computed. Pass `return_vector=true` to `solver_correction_vector`.")
        end
        return getfield(res, :_vector)
    end
    return getfield(res, sym)
end

function Base.propertynames(res::CorrectionVectorResult, private::Bool=false)
    return private ? fieldnames(CorrectionVectorResult) : (:spectral_function, :iterations, :converged, :vector, :correction_vector)
end

function solver_correction_vector(
    H::MatrixFreeHamiltonian{Float64},
    op_psi0::AbstractVector{<:Number};
    e0::Real,
    omega::Real,
    eta::Real,
    maxiter::Integer=100,
    tol::Real=1e-8,
    return_vector::Bool=false
)::CorrectionVectorResult{Float64}
    dim = Int(dimension(H))
    @assert length(op_psi0) == dim "Input vector size $(length(op_psi0)) does not match Hamiltonian dimension $dim"

    op_psi0_c = (op_psi0 isa Vector{ComplexF64}) ? op_psi0 : Vector{ComplexF64}(op_psi0)
    vec_out = return_vector ? Vector{ComplexF64}(undef, dim) : ComplexF64[]
    res_c = Ref{CorrectionVectorResultFP64C}(CorrectionVectorResultFP64C(0.0, 0, 0))

    GC.@preserve op_psi0_c vec_out begin
        vec_ptr = return_vector ? pointer(vec_out) : Ptr{ComplexF64}(C_NULL)
        status = ccall(
            (:qkrylov_solver_correction_vector_fp64, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Cdouble, Cdouble, Cdouble, Cint, Cdouble, Ref{CorrectionVectorResultFP64C}, Ptr{Cdouble}),
            H.ptr, pointer(op_psi0_c), Cdouble(e0), Cdouble(omega), Cdouble(eta), Cint(maxiter), Cdouble(tol), res_c, Ptr{Cdouble}(vec_ptr)
        )
        _check_status(status, "Correction vector solver failed")
    end
    return CorrectionVectorResult(res_c[].spectral_function, Int(res_c[].iterations), res_c[].converged != 0, return_vector ? vec_out : nothing)
end

function solver_correction_vector(
    H::MatrixFreeHamiltonian{Float32},
    op_psi0::AbstractVector{<:Number};
    e0::Real,
    omega::Real,
    eta::Real,
    maxiter::Integer=100,
    tol::Real=1e-5,
    return_vector::Bool=false
)::CorrectionVectorResult{Float32}
    dim = Int(dimension(H))
    @assert length(op_psi0) == dim "Input vector size $(length(op_psi0)) does not match Hamiltonian dimension $dim"

    op_psi0_c = (op_psi0 isa Vector{ComplexF32}) ? op_psi0 : Vector{ComplexF32}(op_psi0)
    vec_out = return_vector ? Vector{ComplexF32}(undef, dim) : ComplexF32[]
    res_c = Ref{CorrectionVectorResultFP32C}(CorrectionVectorResultFP32C(0.0f0, 0, 0))

    GC.@preserve op_psi0_c vec_out begin
        vec_ptr = return_vector ? pointer(vec_out) : Ptr{ComplexF32}(C_NULL)
        status = ccall(
            (:qkrylov_solver_correction_vector_fp32, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}, Cfloat, Cfloat, Cfloat, Cint, Cfloat, Ref{CorrectionVectorResultFP32C}, Ptr{Cfloat}),
            H.ptr, pointer(op_psi0_c), Cfloat(e0), Cfloat(omega), Cfloat(eta), Cint(maxiter), Cfloat(tol), res_c, Ptr{Cfloat}(vec_ptr)
        )
        _check_status(status, "Correction vector solver failed")
    end
    return CorrectionVectorResult(res_c[].spectral_function, Int(res_c[].iterations), res_c[].converged != 0, return_vector ? vec_out : nothing)
end

# -----------------------------------------------------------------------------
# Base.show Formatting for Solver Results
# -----------------------------------------------------------------------------

function Base.show(io::IO, res::LanczosResult{T}) where {T}
    status_str = res.converged ? "converged = true" : "WARNING: maxiter hit without converging!"
    state_str = res.has_state ? ", state = Vector{Complex{$T}}(dim=$(length(res._state)))" : ""
    print(io, "LanczosResult{$T}(energy = $(res.energy), iterations = $(res.iterations), $status_str$state_str)")
end

function Base.show(io::IO, res::DavidsonResult{T}) where {T}
    n = length(res.eigenvalues)
    has_v = res.eigenvectors !== nothing
    status_str = res.converged ? "converged = true" : "WARNING: maxiter hit without converging!"
    print(io, "DavidsonResult{$T}(n_eig = $n, energies = $(res.eigenvalues), iterations = $(res.iterations), $status_str, has_eigenvectors = $has_v)")
end

function Base.show(io::IO, res::ContinuedFractionResult{T}) where {T}
    n = length(res.alphas)
    print(io, "ContinuedFractionResult{$T}(n_coeffs = $n, norm_phi0 = $(res.norm_phi0))")
end

function Base.show(io::IO, res::FTLMResult{T}) where {T}
    print(io, "FTLMResult{$T}(beta = $(res.beta), Z = $(res.partition_function), E = $(res.internal_energy), Cv = $(res.specific_heat))")
end

function Base.show(io::IO, res::CorrectionVectorResult{T}) where {T}
    status_str = res.converged ? "converged = true" : "WARNING: maxiter hit without converging!"
    vec_str = res.has_vector ? ", vector = Vector{Complex{$T}}(dim=$(length(res._vector)))" : ""
    print(io, "CorrectionVectorResult{$T}(S = $(res.spectral_function), iterations = $(res.iterations), $status_str$vec_str)")
end
