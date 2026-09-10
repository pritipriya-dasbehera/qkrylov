#pragma once

#include "qkrylov/core/types.hpp"
#include "qkrylov/linalg/vector_ops.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


struct LanczosResult
{
    Real energy = 0.0;
    int iterations = 0;
    bool converged = false;

    HostVector eigenvector;
};

struct LanczosLowestResult
{
    std::vector<Real> eigenvalues;
    std::vector<HostVector> eigenvectors;
    int iterations = 0;
    bool converged = false;
};

template <typename ExecSpace>
LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int maxiter = 200,
    Real tol = 1.0e-12,
    const HostVector& initial_vector = HostVector()
);

template <typename ExecSpace>
LanczosLowestResult lanczos_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig = 1,
    int maxiter = 200,
    Real tol = 1.0e-12,
    bool compute_eigenvectors = true,
    const HostVector& initial_vector = HostVector()
);

}

}
