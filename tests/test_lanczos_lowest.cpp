#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

int main() {
    std::cout << "Testing lanczos_lowest multi-state solver...\n";

    // 1. Test N = 4 Heisenberg chain (dim = 16, full space)
    int N = 4;
    auto basis4 = std::make_shared<SpinHalfBasis>(N);
    auto site4 = std::make_shared<SpinHalfSite>();
    OpSum os4;
    for (int i = 0; i < N; ++i) {
        int next_i = (i + 1) % N;
        os4 += {1.0, {{"Sz", i}, {"Sz", next_i}}};
        os4 += {0.5, {{"Sp", i}, {"Sm", next_i}}};
        os4 += {0.5, {{"Sm", i}, {"Sp", next_i}}};
    }
    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H4(basis4, site4, os4);

    // Compute lowest 4 states
    int k4 = 4;
    auto res4 = lanczos_lowest<Kokkos::DefaultExecutionSpace>(H4, k4, 100, 1e-10, true);

    std::cout << "N=4 Lanczos Lowest " << k4 << " states:\n";
    std::cout << "  Converged: " << res4.converged << ", Iterations: " << res4.iterations << "\n";
    for (size_t i = 0; i < res4.eigenvalues.size(); ++i) {
        std::cout << "  E[" << i << "] = " << res4.eigenvalues[i] << "\n";
    }
    assert(res4.converged);
    assert(res4.eigenvalues.size() == static_cast<size_t>(k4));
    assert(res4.eigenvectors.size() == static_cast<size_t>(k4));
    assert(std::abs(res4.eigenvalues[0] - (-2.0)) < 1e-5);
    assert(res4.eigenvalues[0] <= res4.eigenvalues[1]);
    assert(res4.eigenvalues[1] <= res4.eigenvalues[2]);
    assert(res4.eigenvalues[2] <= res4.eigenvalues[3]);

    // 2. Test N = 6 Trimerized Heisenberg Chain from davidson_behavior_report.md
    // Ground state: E0 = -1.9819, E1 = -1.4977, E2 = -1.0607
    int N6 = 6;
    auto basis6 = std::make_shared<SpinHalfBasis>(N6);
    auto site6 = std::make_shared<SpinHalfSite>();
    OpSum os6;
    double J1 = std::cos(M_PI / 4.0); // ~ 0.70710678
    double J2 = std::sin(M_PI / 4.0); // ~ 0.70710678
    for (int i = 0; i < N6; ++i) {
        int next_i = (i + 1) % N6;
        double J = (i % 3 == 2) ? J2 : J1;
        os6 += {J, {{"Sz", i}, {"Sz", next_i}}};
        os6 += {0.5 * J, {{"Sp", i}, {"Sm", next_i}}};
        os6 += {0.5 * J, {{"Sm", i}, {"Sp", next_i}}};
    }
    MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> H6(basis6, site6, os6);

    // Compute lowest 3 states (where Davidson failed with runaway divergence)
    int k6 = 3;
    auto res6 = lanczos_lowest<Kokkos::DefaultExecutionSpace>(H6, k6, 100, 1e-8, true);

    std::cout << "\nN=6 Trimerized Chain (where Davidson failed for k=3):\n";
    std::cout << "  Converged: " << res6.converged << ", Iterations: " << res6.iterations << "\n";
    for (size_t i = 0; i < res6.eigenvalues.size(); ++i) {
        std::cout << "  E[" << i << "] = " << res6.eigenvalues[i] << "\n";
    }
    assert(res6.converged);
    assert(res6.eigenvalues.size() == static_cast<size_t>(k6));
    assert(std::abs(res6.eigenvalues[0] - (-1.9819)) < 1e-3);
    assert(std::abs(res6.eigenvalues[1] - (-1.4977)) < 1e-3);
    assert(std::abs(res6.eigenvalues[2] - (-1.0607)) < 1e-3);

    // 3. Test lanczos_ground_state and lanczos_lowest with initial trial vector
    HostVector init_v(H4.dimension());
    for (size_t i = 0; i < init_v.size(); ++i) {
        init_v[i] = res4.eigenvectors[0][i];
    }
    auto res4_init = lanczos_lowest<Kokkos::DefaultExecutionSpace>(H4, 1, 100, 1e-10, true, init_v);
    assert(res4_init.converged);
    assert(std::abs(res4_init.eigenvalues[0] - res4.eigenvalues[0]) < 1e-8);
    assert(res4_init.iterations <= 2);

    auto res_gs_init = lanczos_ground_state<Kokkos::DefaultExecutionSpace>(H4, 100, 1e-10, init_v);
    assert(res_gs_init.converged);
    assert(std::abs(res_gs_init.energy - res4.eigenvalues[0]) < 1e-8);
    assert(res_gs_init.iterations <= 2);

    // Test dimension mismatch throwing invalid_argument
    HostVector bad_v(H4.dimension() + 1);
    bool threw_dim = false;
    try {
        lanczos_lowest<Kokkos::DefaultExecutionSpace>(H4, 1, 100, 1e-10, true, bad_v);
    } catch (const std::invalid_argument&) {
        threw_dim = true;
    }
    assert(threw_dim);

    // Test zero norm throwing invalid_argument
    HostVector zero_v(H4.dimension(), Complex(0.0, 0.0));
    bool threw_zero = false;
    try {
        lanczos_lowest<Kokkos::DefaultExecutionSpace>(H4, 1, 100, 1e-10, true, zero_v);
    } catch (const std::invalid_argument&) {
        threw_zero = true;
    }
    assert(threw_zero);

    std::cout << "\nAll lanczos_lowest tests passed successfully!\n";
    return 0;
}
