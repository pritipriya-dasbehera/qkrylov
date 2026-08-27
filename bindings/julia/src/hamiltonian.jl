# MatrixFreeHamiltonian & operator overloading

mutable struct MatrixFreeHamiltonian
    ptr::Ptr{Cvoid}
    basis::AbstractBasis
    site::AbstractSite
    opsum::OpSum
    device::String

    function MatrixFreeHamiltonian(
        basis::AbstractBasis,
        site::AbstractSite,
        opsum::OpSum;
        device::AbstractString="cpu"
    )
        validate!(opsum, nsites(basis))

        d_lower = lowercase(device)
        if occursin("cuda", d_lower) || occursin("hip", d_lower) || occursin("sycl", d_lower) || d_lower == "gpu"
            if !is_gpu_build()
                throw(ArgumentError("QKrylov was not built with GPU support. Install/compile a GPU build of libqkrylov with Kokkos CUDA/HIP enabled."))
            end
        end

        ptr = if _has_symbol(:qkrylov_hamiltonian_create_device)
            ccall(
                (:qkrylov_hamiltonian_create_device, libqkrylov),
                Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
                basis.ptr, site.ptr, opsum.ptr, device
            )
        else
            ccall(
                (:qkrylov_hamiltonian_create, libqkrylov),
                Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                basis.ptr, site.ptr, opsum.ptr
            )
        end
        ptr == C_NULL && error("Failed to create MatrixFreeHamiltonian on device: $device")
        
        obj = new(ptr, basis, site, opsum, String(device))
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_hamiltonian_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

default_site(b::SpinHalfBasis) = SpinHalfSite()
default_site(b::FermionBasis)  = FermionSite()
default_site(b::HubbardBasis)  = HubbardSite()
default_site(b::TJBasis)       = TJSite()
function default_site(b::AbstractBasis)
    error("Cannot automatically infer Site model for basis type $(typeof(b)). Please provide the site argument explicitly: MatrixFreeHamiltonian(basis, site, opsum)")
end

function MatrixFreeHamiltonian(basis::AbstractBasis, opsum::OpSum; device::AbstractString="cpu")
    site = default_site(basis)
    return MatrixFreeHamiltonian(basis, site, opsum; device=device)
end

function dimension(H::MatrixFreeHamiltonian)::UInt64
    return ccall((:qkrylov_hamiltonian_dimension, libqkrylov), UInt64, (Ptr{Cvoid},), H.ptr)
end

Base.size(H::MatrixFreeHamiltonian) = (Int(dimension(H)), Int(dimension(H)))
Base.size(H::MatrixFreeHamiltonian, d::Integer) = (d == 1 || d == 2) ? Int(dimension(H)) : 1

function Base.:*(H::MatrixFreeHamiltonian, x::AbstractVector{<:Number})::Vector{ComplexF64}
    dim = Int(dimension(H))
    @assert length(x) == dim "Input vector size $(length(x)) does not match Hamiltonian dimension $dim"

    x_c = Vector{ComplexF64}(x)
    x_re = Vector{Float32}(real.(x_c))
    x_im = Vector{Float32}(imag.(x_c))
    y_re = Vector{Float32}(undef, dim)
    y_im = Vector{Float32}(undef, dim)

    GC.@preserve x_re x_im y_re y_im begin
        status = ccall(
            (:qkrylov_hamiltonian_apply, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
            H.ptr, pointer(x_re), pointer(x_im), pointer(y_re), pointer(y_im)
        )
        status != QKRYLOV_SUCCESS && error("MatrixFreeHamiltonian apply failed with status code $status")
    end

    y_c = Vector{ComplexF64}(undef, dim)
    @inbounds for i in 1:dim
        y_c[i] = ComplexF64(Float64(y_re[i]), Float64(y_im[i]))
    end
    return y_c
end

function diagonal(H::MatrixFreeHamiltonian)::Vector{Float64}
    dim = Int(dimension(H))
    diag_buf = Vector{Float32}(undef, dim)

    GC.@preserve diag_buf begin
        status = ccall(
            (:qkrylov_hamiltonian_diagonal, libqkrylov),
            Cint,
            (Ptr{Cvoid}, Ptr{Cfloat}),
            H.ptr, pointer(diag_buf)
        )
        status != QKRYLOV_SUCCESS && error("Failed to extract Hamiltonian diagonal (status code $status)")
    end
    return Vector{Float64}(diag_buf)
end

function Base.show(io::IO, H::MatrixFreeHamiltonian)
    d = dimension(H)
    print(io, "MatrixFreeHamiltonian(dim = $d, basis = $(H.basis))")
end
