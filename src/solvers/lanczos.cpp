#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/lanczos.hpp"

#include <random>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


namespace
{

struct TridiagResult {
    Real energy;
    std::vector<Real> eigenvector;
};

// Diagonalize symmetric tridiagonal matrix and return ground state energy and eigenvector
TridiagResult tridiag_ground_state_full(const std::vector<Real>& alpha, const std::vector<Real>& beta, int n)
{
    if (n == 0) return {0.0, {}};
    if (n == 1) return {alpha[0], {1.0}};

    std::vector<Real> d = alpha;
    std::vector<Real> e = beta;
    std::vector<std::vector<Real>> z(n, std::vector<Real>(n, 0.0));
    for (int i = 0; i < n; ++i) z[i][i] = 1.0;

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(4.0);

    for (int iter = 0; iter < 1000; ++iter) {
        for (int i = 0; i < n - 1; ++i) {
            if (std::abs(e[i]) <= eps * (std::abs(d[i]) + std::abs(d[i+1]))) {
                e[i] = Real(0.0);
            }
        }

        int m = n - 1;
        while (m > 0 && e[m-1] == 0.0) m--;
        if (m == 0) break;

        int l = m - 1;
        while (l > 0 && e[l-1] != 0.0) l--;

        Real b = (d[m-1] - d[m]) / 2.0;
        Real c = e[m-1] * e[m-1];
        Real s = std::sqrt(b*b + c);
        Real shift = (b > 0) ? d[m] - c / (b + s) : d[m] - c / (b - s);

        Real p = d[l] - shift;
        Real g = e[l];

        for (int i = l; i < m; ++i) {
            Real r = std::hypot(p, g);
            Real cos_theta = p / r;
            Real sin_theta = g / r;

            if (i > l) e[i-1] = r;

            Real f = cos_theta * d[i] + sin_theta * e[i];
            Real g_next = cos_theta * e[i] + sin_theta * d[i+1];
            Real h = sin_theta * d[i] - cos_theta * e[i];
            Real k = sin_theta * e[i] - cos_theta * d[i+1];

            d[i] = cos_theta * f + sin_theta * g_next;
            e[i] = cos_theta * h + sin_theta * k;
            d[i+1] = sin_theta * h - cos_theta * k;

            // Update eigenvectors z
            for (int j = 0; j < n; ++j) {
                Real z1 = z[j][i];
                Real z2 = z[j][i+1];
                z[j][i] = cos_theta * z1 + sin_theta * z2;
                z[j][i+1] = sin_theta * z1 - cos_theta * z2;
            }

            if (i < m - 1) {
                p = e[i];
                g = sin_theta * e[i+1];
                e[i+1] = -cos_theta * e[i+1];
            }
        }
    }

    int min_idx = 0;
    for (int i = 1; i < n; ++i) {
        if (d[i] < d[min_idx]) min_idx = i;
    }

    std::vector<Real> res_v(n);
    for (int i = 0; i < n; ++i) res_v[i] = z[i][min_idx];

    return {d[min_idx], res_v};
}
} // namespace

} // namespace QKRYLOV_PRECISION_NAMESPACE

namespace solvers {

template <typename Policy, typename ExecSpace>
QKRYLOV_PRECISION_NAMESPACE::LanczosResult lanczos(
    const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<ExecSpace>& H,
    const LanczosConfig& config
)
{
    using namespace QKRYLOV_PRECISION_NAMESPACE;
    constexpr bool is_two_pass = std::is_same_v<Policy, policy::TwoPass>;
    static_assert(
        std::is_same_v<Policy, policy::Default> ||
        std::is_same_v<Policy, policy::SinglePass> ||
        std::is_same_v<Policy, policy::TwoPass>,
        "Unknown solver policy"
    );

    const Index dim = H.dimension();
    if (dim == 0) return {};

    int maxiter = config.maxiter;
    Real tol = config.tol;

    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> v_curr("v_curr", dim);
    VectorView<ExecSpace> w("w", dim);

    const uint32_t seed = 1234;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    
    auto v_curr_host = Kokkos::create_mirror_view(v_curr);
    for(Index i=0; i<dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
    Kokkos::deep_copy(v_curr, v_curr_host);
    normalize(v_curr);

    std::vector<VectorView<ExecSpace>> basis_vectors;
    if constexpr (!is_two_pass) {
        VectorView<ExecSpace> v_curr_copy("basis_curr", dim);
        Kokkos::deep_copy(v_curr_copy, v_curr);
        basis_vectors.push_back(v_curr_copy);
    }

    std::vector<Real> alphas;
    std::vector<Real> betas;

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    bool is_converged = false;
    Real energy_old = std::numeric_limits<Real>::infinity();
    int actual_iters = 0;

    for(int iter=0; iter < std::min<int>(maxiter, dim); ++iter)
    {
        actual_iters = iter + 1;
        H.apply(v_curr, w);

        Real alpha = dot(v_curr, w).real();
        alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (iter > 0) {
            axpy(-betas.back(), v_prev, w);
        }

        if constexpr (!is_two_pass) {
            // DGKS full reorthogonalization ("twice is enough") to maintain stability to machine precision
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& bv : basis_vectors) {
                    axpy(-dot(bv, w), bv, w);
                }
            }
        }

        Real beta = norm(w);

        if (beta < mach_eps) {
             is_converged = true;
             break;
        }
        if (iter + 1 == dim) {
             is_converged = true;
             break;
        }
        if (iter + 1 == maxiter) {
             break;
        }

        betas.push_back(beta);

        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(1.0/beta, v_curr);
        
        if constexpr (!is_two_pass) {
            VectorView<ExecSpace> v_new("basis", dim);
            Kokkos::deep_copy(v_new, v_curr);
            basis_vectors.push_back(v_new);
        }

        if (iter > 0) {
            // Check convergence only every few iterations or after some initial steps
            auto tridiag = tridiag_ground_state_full(alphas, betas, alphas.size());
            if (std::abs(tridiag.energy - energy_old) < tol) {
                energy_old = tridiag.energy;
                is_converged = true;
                break;
            }
            energy_old = tridiag.energy;
        }
    }

    auto final_tridiag = tridiag_ground_state_full(alphas, betas, alphas.size());

    LanczosResult res;
    res.energy = final_tridiag.energy;
    res.iterations = actual_iters;
    res.converged = is_converged;

    if constexpr (!is_two_pass) {
        // Compute Ritz vector using single pass
        VectorView<ExecSpace> ritz("ritz", dim);
        for (int i = 0; i < (int)alphas.size(); ++i) {
            axpy(KComplex(final_tridiag.eigenvector[i], 0.0), basis_vectors[i], ritz);
        }
        normalize(ritz);
        copy_device_to_host(ritz, res.eigenvector);
        return res;
    } else {
        // Two-pass reconstruction
        VectorView<ExecSpace> ritz("ritz", dim);
        Kokkos::deep_copy(ritz, KComplex(0.0, 0.0));

        rng.seed(seed);
        for(Index i=0; i<dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
        Kokkos::deep_copy(v_curr, v_curr_host);
        normalize(v_curr);
        
        Kokkos::deep_copy(v_prev, KComplex(0.0, 0.0));

        int m = static_cast<int>(alphas.size());
        for (int iter = 0; iter < m; ++iter) {
            axpy(KComplex(final_tridiag.eigenvector[iter], 0.0), v_curr, ritz);

            if (iter + 1 == m) break;

            H.apply(v_curr, w);
            axpy(-alphas[iter], v_curr, w);
            if (iter > 0) {
                axpy(-betas[iter-1], v_prev, w);
            }

            Kokkos::deep_copy(v_prev, v_curr);
            Kokkos::deep_copy(v_curr, w);
            scal(1.0 / betas[iter], v_curr);
        }

        normalize(ritz);
        copy_device_to_host(ritz, res.eigenvector);

        return res;
    }
}

} // namespace solvers

// Explicit instantiations
#define INSTANTIATE_LANCZOS(Space) \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::Default, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::SinglePass, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&); \
    template QKRYLOV_PRECISION_NAMESPACE::LanczosResult solvers::lanczos<solvers::policy::TwoPass, Space>(const QKRYLOV_PRECISION_NAMESPACE::MatrixFreeHamiltonian<Space>&, const QKRYLOV_PRECISION_NAMESPACE::LanczosConfig&);

#ifdef KOKKOS_ENABLE_SERIAL
INSTANTIATE_LANCZOS(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
INSTANTIATE_LANCZOS(Kokkos::OpenMP)
#endif
#ifdef KOKKOS_ENABLE_THREADS
INSTANTIATE_LANCZOS(Kokkos::Threads)
#endif
#ifdef KOKKOS_ENABLE_CUDA
INSTANTIATE_LANCZOS(Kokkos::Cuda)
#endif
#ifdef KOKKOS_ENABLE_HIP
INSTANTIATE_LANCZOS(Kokkos::HIP)
#endif
#ifdef KOKKOS_ENABLE_SYCL
INSTANTIATE_LANCZOS(Kokkos::Experimental::SYCL)
#endif

#undef INSTANTIATE_LANCZOS

} // namespace qkrylov
