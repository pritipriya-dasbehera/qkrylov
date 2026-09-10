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

struct TridiagAllEigensystem {
    std::vector<Real> eigenvalues;
    std::vector<std::vector<Real>> eigenvectors;
};

// Diagonalize symmetric tridiagonal matrix and return all eigenvalues and eigenvectors sorted in ascending order
TridiagAllEigensystem tridiag_eigensystem_full(const std::vector<Real>& alpha, const std::vector<Real>& beta, int n)
{
    if (n <= 0) return {{}, {}};
    if (n == 1) return {{alpha[0]}, {{1.0}}};

    std::vector<Real> d(alpha.begin(), alpha.begin() + n);
    std::vector<Real> e(n, 0.0);
    for (int i = 0; i < n - 1 && i < static_cast<int>(beta.size()); ++i) {
        e[i] = beta[i];
    }

    std::vector<std::vector<Real>> z(n, std::vector<Real>(n, 0.0));
    for (int i = 0; i < n; ++i) z[i][i] = 1.0;

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(4.0);
    const int max_qr_iter = std::max(1000, 100 * n);

    for (int iter = 0; iter < max_qr_iter; ++iter) {
        for (int i = 0; i < n - 1; ++i) {
            if (std::abs(e[i]) <= eps * (std::abs(d[i]) + std::abs(d[i+1]))) {
                e[i] = 0.0;
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

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return d[a] < d[b];
    });

    TridiagAllEigensystem res;
    res.eigenvalues.resize(n);
    res.eigenvectors.resize(n, std::vector<Real>(n));
    for (int k = 0; k < n; ++k) {
        int col = idx[k];
        res.eigenvalues[k] = d[col];
        for (int j = 0; j < n; ++j) {
            res.eigenvectors[k][j] = z[j][col];
        }
    }

    return res;
}

struct TridiagResult {
    Real energy;
    std::vector<Real> eigenvector;
};

TridiagResult tridiag_ground_state_full(const std::vector<Real>& alpha, const std::vector<Real>& beta, int n)
{
    auto all = tridiag_eigensystem_full(alpha, beta, n);
    if (all.eigenvalues.empty()) return {0.0, {}};
    return {all.eigenvalues[0], all.eigenvectors[0]};
}

} // anonymous namespace

template <typename ExecSpace>
LanczosLowestResult lanczos_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig,
    int maxiter,
    Real tol,
    bool compute_eigenvectors,
    const HostVector& initial_vector
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {};

    n_eig = std::min<int>(n_eig, dim);
    if (n_eig <= 0) return {};

    maxiter = std::max(maxiter, n_eig);

    VectorView<ExecSpace> v_prev("v_prev", dim);
    VectorView<ExecSpace> v_curr("v_curr", dim);
    VectorView<ExecSpace> w("w", dim);

    if (!initial_vector.empty()) {
        if (initial_vector.size() != dim) {
            throw std::invalid_argument("initial_vector size (" + std::to_string(initial_vector.size()) +
                                       ") does not match Hamiltonian dimension (" + std::to_string(dim) + ")");
        }
        auto v_curr_host = Kokkos::create_mirror_view(v_curr);
        for (Index i = 0; i < dim; ++i) {
            v_curr_host(i) = KComplex(static_cast<Real>(initial_vector[i].real()),
                                      static_cast<Real>(initial_vector[i].imag()));
        }
        Kokkos::deep_copy(v_curr, v_curr_host);
        Real init_norm = norm(v_curr);
        if (init_norm < 1e-15) {
            throw std::invalid_argument("initial_vector has zero norm");
        }
        normalize(v_curr);
    } else {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<Real> dist(-1.0, 1.0);
        
        auto v_curr_host = Kokkos::create_mirror_view(v_curr);
        for (Index i = 0; i < dim; ++i) v_curr_host(i) = KComplex(dist(rng), dist(rng));
        Kokkos::deep_copy(v_curr, v_curr_host);
        normalize(v_curr);
    }

    std::vector<VectorView<ExecSpace>> basis_vectors;
    VectorView<ExecSpace> v_curr_copy("basis_curr", dim);
    Kokkos::deep_copy(v_curr_copy, v_curr);
    basis_vectors.push_back(v_curr_copy);

    std::vector<Real> alphas;
    std::vector<Real> betas;

    bool is_converged = false;
    std::vector<Real> prev_energies;
    int actual_iters = 0;

    for (int iter = 0; iter < std::min<int>(maxiter, dim); ++iter)
    {
        actual_iters = iter + 1;
        H.apply(v_curr, w);

        Real alpha = dot(v_curr, w).real();
        alphas.push_back(alpha);

        axpy(-alpha, v_curr, w);
        if (iter > 0) {
            axpy(-betas.back(), v_prev, w);
        }

        // Full reorthogonalization (twice-is-enough to maintain orthogonality)
        for (const auto& bv : basis_vectors) {
            axpy(-dot(bv, w), bv, w);
        }
        for (const auto& bv : basis_vectors) {
            axpy(-dot(bv, w), bv, w);
        }

        Real beta = norm(w);

        if (beta < 1e-15) {
            is_converged = true;
            break;
        }
        if (iter + 1 == std::min<int>(maxiter, dim)) {
            break;
        }

        betas.push_back(beta);

        Kokkos::deep_copy(v_prev, v_curr);
        Kokkos::deep_copy(v_curr, w);
        scal(1.0 / beta, v_curr);
        
        VectorView<ExecSpace> v_new("basis", dim);
        Kokkos::deep_copy(v_new, v_curr);
        basis_vectors.push_back(v_new);

        int m = static_cast<int>(alphas.size());
        if (m >= n_eig) {
            auto tridiag = tridiag_eigensystem_full(alphas, betas, m);
            if (!prev_energies.empty()) {
                bool all_converged = true;
                for (int k = 0; k < n_eig; ++k) {
                    Real diff = std::abs(tridiag.eigenvalues[k] - prev_energies[k]);
                    Real ritz_res = beta * std::abs(tridiag.eigenvectors[k][m - 1]);
                    if (diff > tol && ritz_res > tol) {
                        all_converged = false;
                        break;
                    }
                }
                if (all_converged) {
                    is_converged = true;
                    break;
                }
            }
            prev_energies.assign(tridiag.eigenvalues.begin(), tridiag.eigenvalues.begin() + n_eig);
        }
    }

    auto final_tridiag = tridiag_eigensystem_full(alphas, betas, static_cast<int>(alphas.size()));

    LanczosLowestResult res;
    res.iterations = actual_iters;
    res.converged = is_converged;

    int num_out = std::min<int>(n_eig, static_cast<int>(final_tridiag.eigenvalues.size()));
    res.eigenvalues.assign(final_tridiag.eigenvalues.begin(), final_tridiag.eigenvalues.begin() + num_out);

    if (compute_eigenvectors) {
        for (int k = 0; k < num_out; ++k) {
            VectorView<ExecSpace> ritz("ritz", dim);
            for (int i = 0; i < static_cast<int>(alphas.size()); ++i) {
                axpy(KComplex(final_tridiag.eigenvectors[k][i], 0.0), basis_vectors[i], ritz);
            }
            normalize(ritz);
            HostVector host_ritz;
            copy_device_to_host(ritz, host_ritz);
            res.eigenvectors.push_back(host_ritz);
        }
    }

    return res;
}

template <typename ExecSpace>
LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int maxiter,
    Real tol,
    const HostVector& initial_vector
)
{
    auto lowest = lanczos_lowest<ExecSpace>(H, 1, maxiter, tol, true, initial_vector);
    LanczosResult res;
    res.energy = lowest.eigenvalues.empty() ? 0.0 : lowest.eigenvalues[0];
    res.iterations = lowest.iterations;
    res.converged = lowest.converged;
    if (!lowest.eigenvectors.empty()) {
        res.eigenvector = std::move(lowest.eigenvectors[0]);
    }
    return res;
}

// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template LanczosResult lanczos_ground_state<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, int, Real, const HostVector&);
template LanczosLowestResult lanczos_lowest<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, int, int, Real, bool, const HostVector&);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template LanczosResult lanczos_ground_state<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, int, Real, const HostVector&);
template LanczosLowestResult lanczos_lowest<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, int, int, Real, bool, const HostVector&);
#endif
#ifdef KOKKOS_ENABLE_THREADS
template LanczosResult lanczos_ground_state<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, int, Real, const HostVector&);
template LanczosLowestResult lanczos_lowest<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, int, int, Real, bool, const HostVector&);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template LanczosResult lanczos_ground_state<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, int, Real, const HostVector&);
template LanczosLowestResult lanczos_lowest<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, int, int, Real, bool, const HostVector&);
#endif
#ifdef KOKKOS_ENABLE_HIP
template LanczosResult lanczos_ground_state<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, int, Real, const HostVector&);
template LanczosLowestResult lanczos_lowest<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, int, int, Real, bool, const HostVector&);
#endif
#ifdef KOKKOS_ENABLE_SYCL
template LanczosResult lanczos_ground_state<Kokkos::Experimental::SYCL>(const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, int, Real, const HostVector&);
template LanczosLowestResult lanczos_lowest<Kokkos::Experimental::SYCL>(const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, int, int, Real, bool, const HostVector&);
#endif
}

}
