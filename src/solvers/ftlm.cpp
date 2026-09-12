#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/linalg/tridiag_qr.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {

template <typename ExecSpace>
FTLMResult ftlm(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    Real beta,
    int n_random,
    int n_steps
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {beta};

    std::mt19937 rng(42);
    std::normal_distribution<Real> dist(0.0, 1.0);

    struct SampleData {
        Real nrm = 0.0;
        linalg::TridiagAllEigensystem eig;
    };
    std::vector<SampleData> samples(n_random);

    Real E_min = std::numeric_limits<Real>::infinity();

    for (int r = 0; r < n_random; ++r) {
        HostVector r_vec_host(dim);
        Real sum_sq = Real(0.0);
        for (Index i = 0; i < dim; ++i) {
            Complex c(dist(rng), dist(rng));
            r_vec_host[i] = c;
            sum_sq += std::norm(c);
        }
        Real nrm = std::sqrt(sum_sq);

        LanczosConfig cfg;
        cfg.maxiter = n_steps;
        cfg.min_iterations = n_steps;
        cfg.tol = Real(0.0);
        cfg.initial_vector = r_vec_host;
        cfg.compute_eigenvectors = false;

        auto l_res = solvers::lanczos<solvers::policy::OnePass>(H, cfg);
        auto eig = linalg::tridiag_eigensystem_full(l_res.alphas, l_res.betas, static_cast<int>(l_res.alphas.size()));

        if (!eig.eigenvalues.empty() && eig.eigenvalues[0] < E_min) {
            E_min = eig.eigenvalues[0];
        }

        samples[r].nrm = nrm;
        samples[r].eig = std::move(eig);
    }

    if (std::isinf(E_min)) return {beta};

    Real Z_shifted = 0.0;
    Real E_shifted = 0.0;
    Real E2_shifted = 0.0;

    for (int r = 0; r < n_random; ++r) {
        Real nrm = samples[r].nrm;
        const auto& eig = samples[r].eig;

        for (size_t i = 0; i < eig.eigenvalues.size(); ++i) {
            Real exponent = -beta * (eig.eigenvalues[i] - E_min);
            Real exp_val = (exponent < -Real(80.0)) ? Real(0.0) : std::exp(exponent);
            Real v0 = eig.eigenvectors[i][0];
            Real weight = nrm * nrm * v0 * v0 * exp_val;
            Z_shifted += weight;
            E_shifted += eig.eigenvalues[i] * weight;
            E2_shifted += eig.eigenvalues[i] * eig.eigenvalues[i] * weight;
        }
    }

    Z_shifted /= n_random;
    E_shifted /= n_random;
    E2_shifted /= n_random;

    FTLMResult res;
    res.beta = beta;

    if (Z_shifted > Real(0.0)) {
        Real log_Z = std::log(Z_shifted) - beta * E_min;
        const Real max_exp = (sizeof(Real) > 4) ? Real(700.0) : Real(85.0);
        res.partition_function = (log_Z < max_exp) ? std::exp(log_Z) : std::numeric_limits<Real>::infinity();
        res.internal_energy = E_shifted / Z_shifted;
        res.specific_heat = (beta * beta) * (E2_shifted / Z_shifted - (E_shifted / Z_shifted) * (E_shifted / Z_shifted));
    }

    return res;
}


// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template FTLMResult ftlm<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, Real, int, int);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template FTLMResult ftlm<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, Real, int, int);
#endif
#ifdef KOKKOS_ENABLE_THREADS
template FTLMResult ftlm<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, Real, int, int);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template FTLMResult ftlm<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, Real, int, int);
#endif
#ifdef KOKKOS_ENABLE_HIP
template FTLMResult ftlm<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, Real, int, int);
#endif
#ifdef KOKKOS_ENABLE_SYCL
template FTLMResult ftlm<Kokkos::Experimental::SYCL>(const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, Real, int, int);
#endif
}

}
