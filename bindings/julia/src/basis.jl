# Hilbert space basis wrappers

abstract type AbstractBasis end

function dimension(b::AbstractBasis)::UInt64
    return ccall((:qkrylov_basis_dimension, libqkrylov), UInt64, (Ptr{Cvoid},), b.ptr)
end

function nsites(b::AbstractBasis)::Int
    return Int(ccall((:qkrylov_basis_nsites, libqkrylov), Cint, (Ptr{Cvoid},), b.ptr))
end

function state(b::AbstractBasis, index::Integer)::UInt64
    return ccall((:qkrylov_basis_state, libqkrylov), UInt64, (Ptr{Cvoid}, UInt64), b.ptr, UInt64(index))
end

function basis_index(b::AbstractBasis, state_bitstring::Unsigned)::Int64
    return ccall((:qkrylov_basis_index, libqkrylov), Int64, (Ptr{Cvoid}, UInt64), b.ptr, UInt64(state_bitstring))
end

function Base.in(state_bitstring::Unsigned, b::AbstractBasis)::Bool
    res = ccall((:qkrylov_basis_contains, libqkrylov), Cint, (Ptr{Cvoid}, UInt64), b.ptr, UInt64(state_bitstring))
    return res != 0
end

Base.size(b::AbstractBasis) = (Int(dimension(b)), Int(dimension(b)))
Base.length(b::AbstractBasis) = Int(dimension(b))
Base.getindex(b::AbstractBasis, i::Integer) = state(b, i - 1)

mutable struct SpinHalfBasis <: AbstractBasis
    ptr::Ptr{Cvoid}
    sector::Union{Sector, Nothing}

    function SpinHalfBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing; sz::Union{Real, Nothing}=nothing)
        effective_sec = sector
        if effective_sec === nothing && sz !== nothing
            effective_sec = Sector()
            set_sz!(effective_sec, round(Int, 2 * sz))
        end
        sec_ptr = effective_sec === nothing ? C_NULL : effective_sec.ptr
        ptr = ccall((:qkrylov_spinhalf_basis_create, libqkrylov), Ptr{Cvoid}, (Cint, Ptr{Cvoid}), Cint(num_sites), sec_ptr)
        ptr == C_NULL && error("Failed to create SpinHalfBasis")
        obj = new(ptr, effective_sec)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_basis_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct FermionBasis <: AbstractBasis
    ptr::Ptr{Cvoid}
    sector::Union{Sector, Nothing}

    function FermionBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing; n::Union{Integer, Nothing}=nothing)
        effective_sec = sector
        if effective_sec === nothing && n !== nothing
            effective_sec = Sector()
            set_n!(effective_sec, Int(n))
        end
        sec_ptr = effective_sec === nothing ? C_NULL : effective_sec.ptr
        ptr = ccall((:qkrylov_fermion_basis_create, libqkrylov), Ptr{Cvoid}, (Cint, Ptr{Cvoid}), Cint(num_sites), sec_ptr)
        ptr == C_NULL && error("Failed to create FermionBasis")
        obj = new(ptr, effective_sec)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_basis_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct HubbardBasis <: AbstractBasis
    ptr::Ptr{Cvoid}
    sector::Union{Sector, Nothing}

    function HubbardBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing; nup::Union{Integer, Nothing}=nothing, ndn::Union{Integer, Nothing}=nothing)
        effective_sec = sector
        if effective_sec === nothing && (nup !== nothing || ndn !== nothing)
            effective_sec = Sector()
            set_hubbard_particles!(effective_sec, something(nup, 0), something(ndn, 0))
        end
        sec_ptr = effective_sec === nothing ? C_NULL : effective_sec.ptr
        ptr = ccall((:qkrylov_hubbard_basis_create, libqkrylov), Ptr{Cvoid}, (Cint, Ptr{Cvoid}), Cint(num_sites), sec_ptr)
        ptr == C_NULL && error("Failed to create HubbardBasis")
        obj = new(ptr, effective_sec)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_basis_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct TJBasis <: AbstractBasis
    ptr::Ptr{Cvoid}
    sector::Union{Sector, Nothing}

    function TJBasis(num_sites::Integer, sector::Union{Sector, Nothing}=nothing; nup::Union{Integer, Nothing}=nothing, ndn::Union{Integer, Nothing}=nothing)
        effective_sec = sector
        if effective_sec === nothing && (nup !== nothing || ndn !== nothing)
            effective_sec = Sector()
            set_hubbard_particles!(effective_sec, something(nup, 0), something(ndn, 0))
        end
        sec_ptr = effective_sec === nothing ? C_NULL : effective_sec.ptr
        ptr = ccall((:qkrylov_tj_basis_create, libqkrylov), Ptr{Cvoid}, (Cint, Ptr{Cvoid}), Cint(num_sites), sec_ptr)
        ptr == C_NULL && error("Failed to create TJBasis")
        obj = new(ptr, effective_sec)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall((:qkrylov_basis_destroy, libqkrylov), Cvoid, (Ptr{Cvoid},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function Base.show(io::IO, b::AbstractBasis)
    name = string(typeof(b))
    n = nsites(b)
    d = dimension(b)
    if b.sector !== nothing
        print(io, "$name(sites = $n, dim = $d, sector = $(b.sector))")
    else
        print(io, "$name(sites = $n, dim = $d)")
    end
end
