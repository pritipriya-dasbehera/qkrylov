#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/ftlm.hpp"
#include "qkrylov/linalg/vector_ops.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


namespace
{
// Helper for tridiagonalization (simple version without Ritz vector storage)
struct Tridiag
{
    std::vector<Real> alphas;
    std::vector<Real> betas;
};

template <typename ExecSpace>
Tridiag compute_tridiag(const MatrixFreeHamiltonian<ExecSpace>& H, const VectorView<ExecSpace>& v_start, int n_steps)
{
    const Index dim = H.dimension();
    VectorView<ExecSpace> v_curr("v_curr", dim);
    Kokkos::deep_copy(v_curr, v_start);
    normalize(v_curr);

    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> w("w", dim);
    Tridiag res;

    const Real mach_eps = std::numeric_limits<Real>::epsilon() * Real(4.0);

    for (int i = 0; i < n_steps; ++i) {
        H.apply(v_curr, w);
        Real alpha = dot(v_curr, w).real();
        res.alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (i > 0) axpy(-res.betas.back(), v_prev, w);

        Real beta = norm(w);
        if (beta < mach_eps) break;
        res.betas.push_back(beta);

        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(1.0/beta, v_curr);
    }
    return res;
}

// Simple tridiagonal diagonalization to get all eigenvalues and first components of eigenvectors
struct FullTridiagResult {
    std::vector<Real> eigenvalues;
    std::vector<Real> first_components;
};

FullTridiagResult diagonalize_tridiag_components(const std::vector<Real>& alpha, const std::vector<Real>& beta)
{
    int n = alpha.size();
    if (n == 0) return {};
    if (n == 1) return { {alpha[0]}, {1.0} };

    std::vector<Real> d = alpha;
    std::vector<Real> e = beta;
    std::vector<std::vector<Real>> z(n, std::vector<Real>(n, 0.0));
    for (int i = 0; i < n; ++i) z[i][i] = 1.0;

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(4.0);

    for (int iter = 0; iter < 1000; ++iter) {
        for (int i = 0; i < n - 1; ++i) {
            if (std::abs(e[i]) <= eps * (std::abs(d[i]) + std::abs(d[i+1]))) e[i] = Real(0.0);
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

    FullTridiagResult res;
    for (int i = 0; i < n; ++i) {
        res.eigenvalues.push_back(d[i]);
        res.first_components.push_back(z[0][i]);
    }
    return res;
}
}

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

    Real Z = 0.0;
    Real E = 0.0;
    Real E2 = 0.0;

    for (int r = 0; r < n_random; ++r) {
        VectorView<ExecSpace> r_vec("r_vec", dim);
        auto r_vec_host = Kokkos::create_mirror_view(r_vec);
        for (Index i = 0; i < dim; ++i) r_vec_host(i) = KComplex(dist(rng), dist(rng));
        Kokkos::deep_copy(r_vec, r_vec_host);

        Real nrm = norm(r_vec);

        auto tridiag = compute_tridiag(H, r_vec, n_steps);
        auto eig = diagonalize_tridiag_components(tridiag.alphas, tridiag.betas);

        for (size_t i = 0; i < eig.eigenvalues.size(); ++i) {
            Real weight = nrm * nrm * eig.first_components[i] * eig.first_components[i] * std::exp(-beta * eig.eigenvalues[i]);
            Z += weight;
            E += eig.eigenvalues[i] * weight;
            E2 += eig.eigenvalues[i] * eig.eigenvalues[i] * weight;
        }
    }

    Z /= n_random;
    E /= n_random;
    E2 /= n_random;

    FTLMResult res;
    res.beta = beta;
    res.partition_function = Z;
    res.internal_energy = E / Z;
    res.specific_heat = (beta * beta) * (E2 / Z - (E / Z) * (E / Z));

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
