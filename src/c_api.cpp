#include "qkrylov/c_api.h"

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/operators/operator_term.hpp"
#include "qkrylov/operators/opsum.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/hubbard_site.hpp"
#include "qkrylov/sites/tj_site.hpp"
#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"
#include "qkrylov/core/device.hpp"
#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/solvers/davidson.hpp"
#include "qkrylov/solvers/dynamics.hpp"
#include "qkrylov/solvers/ftlm.hpp"

#include <memory>
#include <vector>
#include <complex>
#include <cstring>
#include <exception>

using namespace qkrylov;
using namespace qkrylov::QKRYLOV_PRECISION_NAMESPACE;

// Internal wrapper structs holding shared/unique pointers to C++ objects
struct qkrylov_sector_t {
    Sector sector;
};

struct qkrylov_basis_t {
    std::shared_ptr<Basis> ptr;
};

struct qkrylov_site_t {
    std::shared_ptr<Site> ptr;
};

struct qkrylov_opsum_t {
    OpSum opsum;
};

struct qkrylov_hamiltonian_t {
    std::unique_ptr<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>> ptr;
};

extern "C" {

/* Sector API */
qkrylov_sector_h qkrylov_sector_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_sector_t>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_sector_destroy(qkrylov_sector_h sector) {
    if (sector) delete sector;
}

int qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        sector->sector.use_sz = true;
        sector->sector.sz2 = sz2;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_set_hubbard_particles(qkrylov_sector_h sector, int nup, int ndn) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        sector->sector.use_nup = true;
        sector->sector.use_ndn = true;
        sector->sector.nup = nup;
        sector->sector.ndn = ndn;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_set_n(qkrylov_sector_h sector, int n) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        sector->sector.use_n = true;
        sector->sector.n = n;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_set_nb(qkrylov_sector_h sector, int nb) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        sector->sector.use_nb = true;
        sector->sector.nb = nb;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_get_sz(qkrylov_sector_h sector, int* sz2_out, int* active_out) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        if (active_out) *active_out = sector->sector.use_sz ? 1 : 0;
        if (sz2_out) *sz2_out = sector->sector.sz2;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_get_hubbard_particles(qkrylov_sector_h sector, int* nup_out, int* ndn_out, int* active_out) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        if (active_out) *active_out = (sector->sector.use_nup && sector->sector.use_ndn) ? 1 : 0;
        if (nup_out) *nup_out = sector->sector.nup;
        if (ndn_out) *ndn_out = sector->sector.ndn;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_get_n(qkrylov_sector_h sector, int* n_out, int* active_out) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        if (active_out) *active_out = sector->sector.use_n ? 1 : 0;
        if (n_out) *n_out = sector->sector.n;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_get_nb(qkrylov_sector_h sector, int* nb_out, int* active_out) {
    if (!sector) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        if (active_out) *active_out = sector->sector.use_nb ? 1 : 0;
        if (nb_out) *nb_out = sector->sector.nb;
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* Basis API */
qkrylov_basis_h qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector) {
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        Sector sec = sector ? sector->sector : Sector();
        handle->ptr = std::make_shared<SpinHalfBasis>(num_sites, sec);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector) {
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        Sector sec = sector ? sector->sector : Sector();
        handle->ptr = std::make_shared<FermionBasis>(num_sites, sec);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector) {
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        Sector sec = sector ? sector->sector : Sector();
        handle->ptr = std::make_shared<HubbardBasis>(num_sites, sec);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector) {
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        Sector sec = sector ? sector->sector : Sector();
        handle->ptr = std::make_shared<TJBasis>(num_sites, sec);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_basis_destroy(qkrylov_basis_h basis) {
    if (basis) delete basis;
}

uint64_t qkrylov_basis_dimension(qkrylov_basis_h basis) {
    if (!basis || !basis->ptr) return 0;
    try {
        return basis->ptr->size();
    } catch (...) {
        return 0;
    }
}

int qkrylov_basis_nsites(qkrylov_basis_h basis) {
    if (!basis || !basis->ptr) return 0;
    try {
        if (auto b = std::dynamic_pointer_cast<SpinHalfBasis>(basis->ptr)) return b->nsites();
        if (auto b = std::dynamic_pointer_cast<FermionBasis>(basis->ptr))  return b->nsites();
        if (auto b = std::dynamic_pointer_cast<HubbardBasis>(basis->ptr))  return b->nsites();
        if (auto b = std::dynamic_pointer_cast<TJBasis>(basis->ptr))       return b->nsites();
        return 0;
    } catch (...) {
        return 0;
    }
}

uint64_t qkrylov_basis_state(qkrylov_basis_h basis, uint64_t index) {
    if (!basis || !basis->ptr) return 0;
    try {
        return basis->ptr->state(static_cast<Index>(index));
    } catch (...) {
        return 0;
    }
}

int64_t qkrylov_basis_index(qkrylov_basis_h basis, uint64_t state_bitstring) {
    if (!basis || !basis->ptr) return -1;
    try {
        if (!basis->ptr->contains(static_cast<StateID>(state_bitstring))) return -1;
        return static_cast<int64_t>(basis->ptr->index(static_cast<StateID>(state_bitstring)));
    } catch (...) {
        return -1;
    }
}

int qkrylov_basis_contains(qkrylov_basis_h basis, uint64_t state_bitstring) {
    if (!basis || !basis->ptr) return 0;
    try {
        return basis->ptr->contains(static_cast<StateID>(state_bitstring)) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

/* Site API */
qkrylov_site_h qkrylov_spinhalf_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->ptr = std::make_shared<SpinHalfSite>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_site_h qkrylov_fermion_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->ptr = std::make_shared<FermionSite>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_site_h qkrylov_hubbard_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->ptr = std::make_shared<HubbardSite>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

qkrylov_site_h qkrylov_tj_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->ptr = std::make_shared<TJSite>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_site_destroy(qkrylov_site_h site) {
    if (site) delete site;
}

/* OpSum API */
qkrylov_opsum_h qkrylov_opsum_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_opsum_t>();
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_opsum_destroy(qkrylov_opsum_h opsum) {
    if (opsum) delete opsum;
}

int qkrylov_opsum_clear(qkrylov_opsum_h opsum) {
    if (!opsum) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        opsum->opsum.clear();
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_1body(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                 const char* op1, int site1) {
    if (!opsum || !op1) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        OperatorTerm term;
        term.coeff = Complex(static_cast<Real>(coeff_real), static_cast<Real>(coeff_imag));
        term.factors.push_back({std::string(op1), site1});
        opsum->opsum.add_term(term);
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_2body(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                 const char* op1, int site1,
                                 const char* op2, int site2) {
    if (!opsum || !op1 || !op2) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        OperatorTerm term;
        term.coeff = Complex(static_cast<Real>(coeff_real), static_cast<Real>(coeff_imag));
        term.factors.push_back({std::string(op1), site1});
        term.factors.push_back({std::string(op2), site2});
        opsum->opsum.add_term(term);
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_nbody(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                 int n_factors, const char** ops, const int* sites) {
    if (!opsum || !ops || !sites || n_factors <= 0) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        OperatorTerm term;
        term.coeff = Complex(static_cast<Real>(coeff_real), static_cast<Real>(coeff_imag));
        for (int i = 0; i < n_factors; ++i) {
            if (!ops[i]) return QKRYLOV_ERROR_INVALID_ARG;
            term.factors.push_back({std::string(ops[i]), sites[i]});
        }
        opsum->opsum.add_term(term);
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* Device & Hardware Query API */
int qkrylov_is_gpu_build(void) {
    return Device::is_gpu_build() ? 1 : 0;
}

const char* qkrylov_find_gpu(void) {
    if (Device::is_gpu_build()) {
        static std::string backend = Device::backend_name();
        return backend.c_str();
    }
    return nullptr;
}

int qkrylov_gpu_count(void) {
    return Device::gpu_count();
}

int qkrylov_initialize_device(const char* device_str) {
    try {
        detail::initialize_kokkos(Device(device_str ? device_str : "cpu"));
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace> API */
qkrylov_hamiltonian_h qkrylov_hamiltonian_create(qkrylov_basis_h basis,
                                                qkrylov_site_h site,
                                                qkrylov_opsum_h opsum) {
    return qkrylov_hamiltonian_create_device(basis, site, opsum, "cpu");
}

qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device(qkrylov_basis_h basis,
                                                        qkrylov_site_h site,
                                                        qkrylov_opsum_h opsum,
                                                        const char* device_str) {
    if (!basis || !basis->ptr || !site || !site->ptr || !opsum) return nullptr;
    try {
        std::string dev_str = device_str ? device_str : "cpu";
        auto handle = std::make_unique<qkrylov_hamiltonian_t>();
        handle->ptr = std::make_unique<MatrixFreeHamiltonian<Kokkos::DefaultExecutionSpace>>(
            basis->ptr, site->ptr, opsum->opsum, Device(dev_str)
        );
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h) {
    if (h) delete h;
}

uint64_t qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h) {
    if (!h || !h->ptr) return 0;
    try {
        return h->ptr->dimension();
    } catch (...) {
        return 0;
    }
}

int qkrylov_hamiltonian_apply(qkrylov_hamiltonian_h h,
                              const float* x_real, const float* x_imag,
                              float* y_real, float* y_imag) {
    if (!h || !h->ptr || !x_real || !y_real) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        const uint64_t dim = h->ptr->dimension();
        std::vector<Complex> x(dim);
        std::vector<Complex> y(dim);

        for (uint64_t i = 0; i < dim; ++i) {
            float imag = x_imag ? x_imag[i] : 0.0f;
            x[i] = Complex(static_cast<Real>(x_real[i]), static_cast<Real>(imag));
        }

        h->ptr->apply(x.data(), y.data());

        for (uint64_t i = 0; i < dim; ++i) {
            y_real[i] = static_cast<float>(y[i].real());
            if (y_imag) y_imag[i] = static_cast<float>(y[i].imag());
        }

        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_hamiltonian_apply_complex(qkrylov_hamiltonian_h h,
                                        const float* x_complex,
                                        float* y_complex) {
    if (!h || !h->ptr || !x_complex || !y_complex) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        const uint64_t dim = h->ptr->dimension();
        if constexpr (std::is_same_v<Real, float>) {
            const auto* x_c = reinterpret_cast<const Complex*>(x_complex);
            auto* y_c = reinterpret_cast<Complex*>(y_complex);
            h->ptr->apply(x_c, y_c);
        } else {
            std::vector<Complex> x(dim);
            std::vector<Complex> y(dim);
            const float* src = x_complex;
            for (uint64_t i = 0; i < dim; ++i) {
                x[i] = Complex(static_cast<Real>(src[2 * i]), static_cast<Real>(src[2 * i + 1]));
            }
            h->ptr->apply(x.data(), y.data());
            float* dst = y_complex;
            for (uint64_t i = 0; i < dim; ++i) {
                dst[2 * i]     = static_cast<float>(y[i].real());
                dst[2 * i + 1] = static_cast<float>(y[i].imag());
            }
        }
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_hamiltonian_diagonal(qkrylov_hamiltonian_h h, float* diag_out) {
    if (!h || !h->ptr || !diag_out) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        auto diag = h->ptr->diagonal();
        const uint64_t dim = h->ptr->dimension();
        for (uint64_t i = 0; i < dim && i < diag.size(); ++i) {
            diag_out[i] = static_cast<float>(diag[i].real());
        }
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* Solvers API */
int qkrylov_lanczos_ground_state(qkrylov_hamiltonian_h h,
                                 int maxiter,
                                 float tol,
                                 qkrylov_lanczos_result_c_t* result) {
    return qkrylov_lanczos_ground_state_complex(h, maxiter, tol, result, nullptr);
}

int qkrylov_lanczos_ground_state_complex(qkrylov_hamiltonian_h h,
                                         int maxiter,
                                         float tol,
                                         qkrylov_lanczos_result_c_t* result,
                                         float* eigenvector_complex) {
    if (!h || !h->ptr || !result) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        auto res = lanczos_ground_state(*(h->ptr), maxiter, static_cast<Real>(tol));
        result->energy     = static_cast<float>(res.energy);
        result->iterations = res.iterations;
        result->converged  = res.converged ? 1 : 0;
        if (eigenvector_complex && !res.eigenvector.empty()) {
            for (size_t i = 0; i < res.eigenvector.size(); ++i) {
                eigenvector_complex[2 * i]     = static_cast<float>(res.eigenvector[i].real());
                eigenvector_complex[2 * i + 1] = static_cast<float>(res.eigenvector[i].imag());
            }
        }
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_davidson_lowest_complex(qkrylov_hamiltonian_h h,
                                    int n_eig,
                                    int max_subspace,
                                    float tol,
                                    float* eigenvalues_out,
                                    float* eigenvectors_complex_out,
                                    qkrylov_davidson_result_c_t* result_info) {
    if (!h || !h->ptr || !eigenvalues_out || n_eig <= 0) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        auto res = davidson_lowest(*(h->ptr), n_eig, max_subspace, static_cast<Real>(tol));
        const size_t k = std::min(static_cast<size_t>(n_eig), res.eigenvalues.size());
        for (size_t i = 0; i < k; ++i) {
            eigenvalues_out[i] = static_cast<float>(res.eigenvalues[i]);
        }

        if (result_info) {
            result_info->iterations = res.iterations;
            result_info->converged  = res.converged ? 1 : 0;
        }

        if (eigenvectors_complex_out) {
            const uint64_t dim = h->ptr->dimension();
            for (size_t idx = 0; idx < k; ++idx) {
                const auto& vec = res.eigenvectors[idx];
                float* dst = eigenvectors_complex_out + (idx * 2 * dim);
                for (size_t i = 0; i < dim && i < vec.size(); ++i) {
                    dst[2 * i]     = static_cast<float>(vec[i].real());
                    dst[2 * i + 1] = static_cast<float>(vec[i].imag());
                }
            }
        }
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_continued_fraction_coeffs_complex(qkrylov_hamiltonian_h h,
                                              const float* phi0_complex,
                                              int n_iter,
                                              float* alphas_out,
                                              float* betas_out,
                                              float* norm_phi0_out,
                                              int* num_coeffs_out) {
    if (!h || !h->ptr || !phi0_complex || !alphas_out || !betas_out || !norm_phi0_out || n_iter <= 0) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        const uint64_t dim = h->ptr->dimension();
        HostVector phi0(dim);
        for (uint64_t i = 0; i < dim; ++i) {
            phi0[i] = Complex(static_cast<Real>(phi0_complex[2 * i]), static_cast<Real>(phi0_complex[2 * i + 1]));
        }

        auto res = continued_fraction_coeffs(*(h->ptr), phi0, n_iter);
        *norm_phi0_out = static_cast<float>(res.norm_phi0);

        const size_t n_alpha = res.alphas.size();
        const size_t n_beta  = res.betas.size();

        for (size_t i = 0; i < n_alpha && i < static_cast<size_t>(n_iter); ++i) {
            alphas_out[i] = static_cast<float>(res.alphas[i]);
        }
        for (size_t i = 0; i < n_beta && i < static_cast<size_t>(n_iter); ++i) {
            betas_out[i] = static_cast<float>(res.betas[i]);
        }

        if (num_coeffs_out) {
            *num_coeffs_out = static_cast<int>(n_alpha);
        }

        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

float qkrylov_evaluate_spectral_function(const float* alphas,
                                         const float* betas,
                                         size_t n,
                                         float norm_phi0,
                                         float omega,
                                         float E0,
                                         float eta) {
    if (!alphas || !betas || n == 0) return 0.0f;
    try {
        std::vector<Real> alphas_real(n);
        std::vector<Real> betas_real(n > 0 ? n - 1 : 0);
        for (size_t i = 0; i < n; ++i) {
            alphas_real[i] = static_cast<Real>(alphas[i]);
        }
        size_t n_beta = n > 0 ? n - 1 : 0;
        for (size_t i = 0; i < n_beta; ++i) {
            betas_real[i] = static_cast<float>(betas[i]);
        }
        Real val = evaluate_spectral_function(alphas_real.data(), betas_real.data(), n,
                                              static_cast<Real>(norm_phi0),
                                              static_cast<Real>(omega),
                                              static_cast<Real>(E0),
                                              static_cast<Real>(eta));
        return static_cast<float>(val);
    } catch (...) {
        return 0.0f;
    }
}

int qkrylov_ftlm(qkrylov_hamiltonian_h h,
                 float beta,
                 int n_random,
                 int n_steps,
                 qkrylov_ftlm_result_c_t* result) {
    if (!h || !h->ptr || !result) return QKRYLOV_ERROR_INVALID_ARG;
    try {
        auto res = ftlm(*(h->ptr), static_cast<Real>(beta), n_random, n_steps);
        result->beta               = static_cast<float>(res.beta);
        result->partition_function = static_cast<float>(res.partition_function);
        result->internal_energy    = static_cast<float>(res.internal_energy);
        result->specific_heat      = static_cast<float>(res.specific_heat);
        return QKRYLOV_SUCCESS;
    } catch (...) {
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

} // extern "C"
