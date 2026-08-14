# Shared library loader and C API status codes for qkrylov

const QKRYLOV_SUCCESS           =  0
const QKRYLOV_ERROR_INVALID_ARG = -1
const QKRYLOV_ERROR_EXCEPTION   = -2

function find_libqkrylov()
    # 1. Custom environment variable override (for local development)
    if haskey(ENV, "QKRYLOV_LIB_PATH") && isfile(ENV["QKRYLOV_LIB_PATH"])
        return ENV["QKRYLOV_LIB_PATH"]
    end

    # 2. Local repository build path (for development)
    root_dir = normpath(joinpath(@__DIR__, "..", "..", ".."))
    candidates = [
        joinpath(root_dir, "build", "libqkrylov.so"),
        joinpath(root_dir, "build", "libqkrylov.dylib"),
        joinpath(root_dir, "build", "qkrylov.dll"),
        joinpath(root_dir, "build", "Release", "qkrylov.dll"),
        joinpath(root_dir, "build", "Debug", "qkrylov.dll")
    ]

    for path in candidates
        if isfile(path)
            return path
        end
    end

    # 3. Native Julia Artifacts resolution (auto-downloaded by Pkg via Artifacts.toml)
    artifacts_file = joinpath(@__DIR__, "..", "Artifacts.toml")
    if isfile(artifacts_file)
        try
            meta = Artifacts.artifact_meta("libqkrylov", artifacts_file)
            if meta !== nothing
                hash = Base.SHA1(meta["git-tree-sha1"])
                if !Artifacts.artifact_exists(hash)
                    try
                        Pkg.Artifacts.ensure_artifact_installed("libqkrylov", artifacts_file)
                    catch
                    end
                end
                if Artifacts.artifact_exists(hash)
                    artifact_dir = Artifacts.artifact_path(hash)
                    for candidate_rel in [
                        joinpath("lib", "libqkrylov.so"),
                        joinpath("lib", "libqkrylov.dylib"),
                        joinpath("lib", "qkrylov.dll"),
                        joinpath("bin", "qkrylov.dll"),
                        "libqkrylov.so",
                        "libqkrylov.dylib",
                        "qkrylov.dll"
                    ]
                        candidate_path = joinpath(artifact_dir, candidate_rel)
                        if isfile(candidate_path)
                            return candidate_path
                        end
                    end
                end
            end
        catch
        end
    end

    # 4. Fallback to qkrylov_jll if available in runtime environment
    try
        if isdefined(QuantumKrylov, :qkrylov_jll) && isdefined(qkrylov_jll, :libqkrylov)
            return qkrylov_jll.libqkrylov
        end
    catch
    end

    # 5. Fallback to system library resolution
    return "libqkrylov"
end

global libqkrylov::String = find_libqkrylov()

function __init__()
    global libqkrylov = find_libqkrylov()
end

