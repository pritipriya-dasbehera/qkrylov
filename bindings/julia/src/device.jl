# Device management and hardware query wrappers
using Libdl

function _has_symbol(sym::Symbol)::Bool
    try
        h = Libdl.dlopen(libqkrylov, Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL)
        return Libdl.dlsym_e(h, sym) != C_NULL
    catch
        return false
    end
end

"""
    is_gpu_build() -> Bool

Return `true` if the underlying `libqkrylov` binary was compiled with GPU acceleration
(CUDA, HIP, or SYCL), or `false` for a CPU-only build.
"""
function is_gpu_build()::Bool
    if !_has_symbol(:qkrylov_is_gpu_build)
        return false
    end
    return ccall((:qkrylov_is_gpu_build, libqkrylov), Cint, ()) != 0
end

"""
    find_gpu() -> Union{String, Nothing}

Return the name of the compiled GPU backend ("cuda", "hip", "sycl") if available,
or `nothing` if built for CPU only. (Matches Python API `qkrylov.find_gpu()`).
"""
function find_gpu()::Union{String, Nothing}
    if !_has_symbol(:qkrylov_find_gpu)
        return nothing
    end
    ptr = ccall((:qkrylov_find_gpu, libqkrylov), Cstring, ())
    return ptr == C_NULL ? nothing : unsafe_string(ptr)
end

"""
    gpu_count() -> Int

Return the number of available physical GPUs detected on the system.
(Matches Python API `qkrylov.gpu_count()`).
"""
function gpu_count()::Int
    if !_has_symbol(:qkrylov_gpu_count)
        return 0
    end
    return Int(ccall((:qkrylov_gpu_count, libqkrylov), Cint, ()))
end

"""
    initialize_device!(device::AbstractString="cpu")

Explicitly initialize Kokkos execution spaces for a targeted device (e.g. "cpu", "cuda:0").
"""
function initialize_device!(device::AbstractString="cpu")
    if !_has_symbol(:qkrylov_initialize_device)
        return nothing
    end
    status = ccall((:qkrylov_initialize_device, libqkrylov), Cint, (Cstring,), device)
    status != QKRYLOV_SUCCESS && error("Failed to initialize target device: $device with status code $status")
    return nothing
end
