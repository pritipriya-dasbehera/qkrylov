#include "qkrylov/c_api.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <complex>

int main() {
    std::cout << "Testing C API..." << std::endl;

    // 1. Test Sector API
    qkrylov_sector_h sec = qkrylov_sector_create();
    assert(sec != NULL);
    int res_sz = qkrylov_sector_set_sz(sec, 0);
    assert(res_sz == QKRYLOV_SUCCESS);

    // 2. Test Basis API (4-site SpinHalfBasis with total Sz = 0)
    int N = 4;
    qkrylov_basis_h basis = qkrylov_spinhalf_basis_create(N, sec);
    assert(basis != NULL);
    uint64_t dim = qkrylov_basis_dimension(basis);
    std::cout << "Basis dimension for N=4, Sz=0: " << dim << std::endl;
    assert(dim == 6); // 4 choose 2 = 6 states in Sz=0 sector
    assert(qkrylov_basis_nsites(basis) == 4);

    // Test Basis state lookups
    uint64_t s0 = qkrylov_basis_state(basis, 0);
    assert(qkrylov_basis_contains(basis, s0) == 1);
    assert(qkrylov_basis_index(basis, s0) == 0);

    // Test Sector set_n and set_nb functions
    assert(qkrylov_sector_set_n(sec, 2) == QKRYLOV_SUCCESS);
    assert(qkrylov_sector_set_nb(sec, 1) == QKRYLOV_SUCCESS);

    // 3. Test Site API
    qkrylov_site_h site = qkrylov_spinhalf_site_create();
    assert(site != NULL);

    // 4. Test OpSum API (4-site Heisenberg chain)
    qkrylov_opsum_h opsum = qkrylov_opsum_create();
    assert(opsum != NULL);

    for (int i = 0; i < N - 1; ++i) {
        // Sz_i Sz_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 1.0, 0.0, "Sz", i, "Sz", i+1) == QKRYLOV_SUCCESS);
        // 0.5 Sp_i Sm_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 0.5, 0.0, "Sp", i, "Sm", i+1) == QKRYLOV_SUCCESS);
        // 0.5 Sm_i Sp_{i+1}
        assert(qkrylov_opsum_add_term_2body(opsum, 0.5, 0.0, "Sm", i, "Sp", i+1) == QKRYLOV_SUCCESS);
    }

    // Test 3-Body N-Body Term API on separate OpSum handle
    qkrylov_opsum_h opsum_nbody = qkrylov_opsum_create();
    assert(opsum_nbody != NULL);
    const char* ops3[3] = {"Sz", "Sz", "Sz"};
    int sites3[3] = {0, 1, 2};
    assert(qkrylov_opsum_add_term_nbody(opsum_nbody, 0.1f, 0.0f, 3, ops3, sites3) == QKRYLOV_SUCCESS);
    qkrylov_opsum_destroy(opsum_nbody);

    // 5. Test Device & Hamiltonian API
    int is_gpu = qkrylov_is_gpu_build();
    int gpus = qkrylov_gpu_count();
    const char* gpu_backend = qkrylov_find_gpu();
    std::cout << "Device check: is_gpu=" << is_gpu << ", gpu_count=" << gpus
              << ", backend=" << (gpu_backend ? gpu_backend : "none") << std::endl;
    assert(qkrylov_initialize_device("cpu") == QKRYLOV_SUCCESS);

    qkrylov_hamiltonian_h H = qkrylov_hamiltonian_create(basis, site, opsum);
    assert(H != NULL);
    assert(qkrylov_hamiltonian_dimension(H) == dim);

    qkrylov_hamiltonian_h H_dev = qkrylov_hamiltonian_create_device(basis, site, opsum, "cpu");
    assert(H_dev != NULL);
    assert(qkrylov_hamiltonian_dimension(H_dev) == dim);
    qkrylov_hamiltonian_destroy(H_dev);

    // 6. Test Matrix-Vector Apply
    std::vector<float> x_real(dim, 1.0);
    std::vector<float> x_imag(dim, 0.0);
    std::vector<float> y_real(dim, 0.0);
    std::vector<float> y_imag(dim, 0.0);

    int apply_res = qkrylov_hamiltonian_apply(H, x_real.data(), x_imag.data(), y_real.data(), y_imag.data());
    assert(apply_res == QKRYLOV_SUCCESS);

    // Test Zero-Copy Direct Complex Apply
    std::vector<std::complex<float>> x_cx(dim, std::complex<float>(1.0, 0.0));
    std::vector<std::complex<float>> y_cx(dim, std::complex<float>(0.0, 0.0));
    int apply_cx_res = qkrylov_hamiltonian_apply_complex(H, reinterpret_cast<const float*>(x_cx.data()), reinterpret_cast<float*>(y_cx.data()));
    assert(apply_cx_res == QKRYLOV_SUCCESS);
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(y_cx[i].real() - y_real[i]) < 1e-12);
        assert(std::abs(y_cx[i].imag() - y_imag[i]) < 1e-12);
    }

    // Test Matrix-Free Diagonal Extraction
    std::vector<float> diag(dim);
    int diag_res = qkrylov_hamiltonian_diagonal(H, diag.data());
    assert(diag_res == QKRYLOV_SUCCESS);

    // 7. Test Lanczos Ground State Solver via C API (Energy & Eigenvector)
    qkrylov_lanczos_result_c_t lanczos_res;
    std::vector<std::complex<float>> psi_cx(dim);
    int solver_res = qkrylov_lanczos_ground_state_complex(H, 200, 1e-12, &lanczos_res, reinterpret_cast<float*>(psi_cx.data()));
    assert(solver_res == QKRYLOV_SUCCESS);

    std::cout << "C API Lanczos Ground State Energy: " << lanczos_res.energy << std::endl;
    assert(std::abs(lanczos_res.energy - (-1.6160254038)) < 1e-5);

    // Verify eigenvector normalization: ||psi||^2 == 1.0
    float norm_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        norm_sq += std::norm(psi_cx[i]);
    }
    assert(std::abs(norm_sq - 1.0f) < 1e-4);

    // 8. Test Davidson Solver via C API (Lowest 2 Eigenpairs)
    int n_eig = 2;
    std::vector<float> dav_evals(n_eig);
    std::vector<std::complex<float>> dav_evecs(n_eig * dim);
    qkrylov_davidson_result_c_t dav_info;
    int dav_status = qkrylov_davidson_lowest_complex(H, n_eig, 20, 1e-6f, dav_evals.data(), reinterpret_cast<float*>(dav_evecs.data()), &dav_info);
    assert(dav_status == QKRYLOV_SUCCESS);
    assert(dav_info.converged == 1);

    std::cout << "C API Davidson Lowest Eigenvalues: E0=" << dav_evals[0] << ", E1=" << dav_evals[1] << std::endl;
    assert(std::abs(dav_evals[0] - lanczos_res.energy) < 1e-4);
    assert(dav_evals[0] <= dav_evals[1]);

    // Verify H * psi_k == E_k * psi_k for each computed eigenpair
    for (int k = 0; k < n_eig; ++k) {
        const std::complex<float>* vk = dav_evecs.data() + (k * dim);
        std::vector<std::complex<float>> Hvk(dim);
        assert(qkrylov_hamiltonian_apply_complex(H, reinterpret_cast<const float*>(vk), reinterpret_cast<float*>(Hvk.data())) == QKRYLOV_SUCCESS);
        for (size_t i = 0; i < dim; ++i) {
            std::complex<float> expected = dav_evals[k] * vk[i];
            assert(std::abs(Hvk[i] - expected) < 1e-3);
        }
    }
    // 9. Test Dynamics & Spectral Function C API
    int n_iter = 20;
    std::vector<float> alphas(n_iter);
    std::vector<float> betas(n_iter);
    float norm_phi0 = 0.0f;
    int num_coeffs = 0;

    int dyn_status = qkrylov_continued_fraction_coeffs_complex(H, reinterpret_cast<const float*>(psi_cx.data()), n_iter, alphas.data(), betas.data(), &norm_phi0, &num_coeffs);
    assert(dyn_status == QKRYLOV_SUCCESS);
    assert(num_coeffs > 0);
    assert(std::abs(norm_phi0 - 1.0f) < 1e-3);

    float spec_val = qkrylov_evaluate_spectral_function(alphas.data(), betas.data(), num_coeffs, norm_phi0, 0.5f, lanczos_res.energy, 0.1f);
    std::cout << "C API Spectral function at w=0.5: " << spec_val << std::endl;
    assert(spec_val > 0.0f);

    // 10. Test FTLM C API
    qkrylov_ftlm_result_c_t ftlm_res;
    int ftlm_status = qkrylov_ftlm(H, 1.0f, 10, 20, &ftlm_res);
    assert(ftlm_status == QKRYLOV_SUCCESS);
    std::cout << "C API FTLM Result: Z=" << ftlm_res.partition_function << ", E=" << ftlm_res.internal_energy << ", Cv=" << ftlm_res.specific_heat << std::endl;
    assert(ftlm_res.partition_function > 0.0f);

    // Cleanup handles
    qkrylov_hamiltonian_destroy(H);
    qkrylov_opsum_destroy(opsum);
    qkrylov_site_destroy(site);
    qkrylov_basis_destroy(basis);
    qkrylov_sector_destroy(sec);

    std::cout << "C API tests passed successfully!" << std::endl;
    return 0;
}
