"""Python interface for qkrylov.

A modern C++20 framework for matrix-free Krylov methods in quantum many-body physics,
now with a Pythonic wrapper layer.
"""

from .operators import (
    Op, OpSum,
    Sz, Sp, Sm, Sx, Sy,
    CdagUp, CUp, CdagDn, CDn,
    Nup, Ndn, Nupdn,
    Bdag, B, N
)
from .site import Site, SpinHalfSite, SpinSSite, FermionSite, HubbardSite, TJSite
from .basis import Basis, SpinHalfBasis, SpinSBasis, FermionBasis, HubbardBasis, TJBasis
from .hamiltonian import MatrixFreeHamiltonian
from .solvers import (
    LanczosResult,
    lanczos_ground_state,
    DavidsonResult,
    davidson_lowest,
    DynamicsResult,
    continued_fraction_coeffs,
    evaluate_spectral_function,
    FTLMResult,
    ftlm,
    CorrectionVectorResult,
    correction_vector,
    correction_vector_spectral,
)

import os
import site
import glob
import ctypes

# Preload NVIDIA CUDA libraries from pip packages if they exist
try:
    for site_dir in site.getsitepackages():
        cuda_libs = glob.glob(os.path.join(site_dir, "nvidia", "*", "lib"))
        for lib_dir in cuda_libs:
            for so_file in glob.glob(os.path.join(lib_dir, "*.so*")):
                try:
                    ctypes.CDLL(so_file, mode=os.RTLD_GLOBAL)
                except OSError:
                    pass
except Exception:
    pass

from ._qkrylov_cpp import Device_FP32 as Device

def find_gpu():
    """Return the name of the GPU backend ('cuda', 'hip', 'sycl') if available, else None."""
    if Device.is_gpu_build():
        return Device.backend_name()
    return None

def gpu_count():
    """Return the number of available GPUs."""
    return Device.gpu_count()


try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("qkrylov")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    # Operators
    "Op",
    "OpSum",
    
    # Operator Generators
    "Sz", "Sp", "Sm", "Sx", "Sy",
    "CdagUp", "CUp", "CdagDn", "CDn",
    "Nup", "Ndn", "Nupdn",
    "Bdag", "B", "N",
    
    # Sites
    "Site",
    "SpinHalfSite",
    "SpinSSite",
    "FermionSite",
    "HubbardSite",
    "TJSite",
    
    # Bases
    "Basis",
    "SpinHalfBasis",
    "SpinSBasis",
    "FermionBasis",
    "HubbardBasis",
    "TJBasis",
    
    # Hamiltonian
    "MatrixFreeHamiltonian",
    
    # Solvers
    "LanczosResult",
    "lanczos_ground_state",
    "DavidsonResult",
    "davidson_lowest",
    "DynamicsResult",
    "continued_fraction_coeffs",
    "evaluate_spectral_function",
    "FTLMResult",
    "ftlm",
    "CorrectionVectorResult",
    "correction_vector",
    "correction_vector_spectral",
    
    # Utilities
    "find_gpu",
    "gpu_count",
]
