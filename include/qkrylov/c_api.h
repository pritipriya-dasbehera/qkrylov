#ifndef QKRYLOV_C_API_H
#define QKRYLOV_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 * Return Error Codes
 * ----------------------------------------------------------------------------- */
#define QKRYLOV_SUCCESS                0
#define QKRYLOV_ERROR_INVALID_ARG     -1
#define QKRYLOV_ERROR_EXCEPTION       -2

/* -----------------------------------------------------------------------------
 * Opaque Handles
 * ----------------------------------------------------------------------------- */
typedef struct qkrylov_sector_t*      qkrylov_sector_h;
typedef struct qkrylov_basis_t*       qkrylov_basis_h;
typedef struct qkrylov_site_t*        qkrylov_site_h;
typedef struct qkrylov_opsum_t*       qkrylov_opsum_h;
typedef struct qkrylov_hamiltonian_t* qkrylov_hamiltonian_h;

/* -----------------------------------------------------------------------------
 * Sector API
 * ----------------------------------------------------------------------------- */
qkrylov_sector_h qkrylov_sector_create(void);
void             qkrylov_sector_destroy(qkrylov_sector_h sector);
int              qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2);
int              qkrylov_sector_set_hubbard_particles(qkrylov_sector_h sector, int nup, int ndn);
int              qkrylov_sector_set_n(qkrylov_sector_h sector, int n);
int              qkrylov_sector_set_nb(qkrylov_sector_h sector, int nb);
int              qkrylov_sector_get_sz(qkrylov_sector_h sector, int* sz2_out, int* active_out);
int              qkrylov_sector_get_hubbard_particles(qkrylov_sector_h sector, int* nup_out, int* ndn_out, int* active_out);
int              qkrylov_sector_get_n(qkrylov_sector_h sector, int* n_out, int* active_out);
int              qkrylov_sector_get_nb(qkrylov_sector_h sector, int* nb_out, int* active_out);

/* -----------------------------------------------------------------------------
 * Basis API
 * ----------------------------------------------------------------------------- */
qkrylov_basis_h  qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h  qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h  qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector);
qkrylov_basis_h  qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector);
void             qkrylov_basis_destroy(qkrylov_basis_h basis);
uint64_t         qkrylov_basis_dimension(qkrylov_basis_h basis);
int              qkrylov_basis_nsites(qkrylov_basis_h basis);
uint64_t         qkrylov_basis_state(qkrylov_basis_h basis, uint64_t index);
int64_t          qkrylov_basis_index(qkrylov_basis_h basis, uint64_t state_bitstring);
int              qkrylov_basis_contains(qkrylov_basis_h basis, uint64_t state_bitstring);

/* -----------------------------------------------------------------------------
 * Site API
 * ----------------------------------------------------------------------------- */
qkrylov_site_h   qkrylov_spinhalf_site_create(void);
qkrylov_site_h   qkrylov_fermion_site_create(void);
qkrylov_site_h   qkrylov_hubbard_site_create(void);
qkrylov_site_h   qkrylov_tj_site_create(void);
void             qkrylov_site_destroy(qkrylov_site_h site);

/* -----------------------------------------------------------------------------
 * OpSum API
 * ----------------------------------------------------------------------------- */
qkrylov_opsum_h  qkrylov_opsum_create(void);
void             qkrylov_opsum_destroy(qkrylov_opsum_h opsum);
int              qkrylov_opsum_clear(qkrylov_opsum_h opsum);
int              qkrylov_opsum_add_term_1body(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                             const char* op1, int site1);
int              qkrylov_opsum_add_term_2body(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                             const char* op1, int site1,
                                             const char* op2, int site2);
int              qkrylov_opsum_add_term_nbody(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                             int n_factors, const char** ops, const int* sites);

/* -----------------------------------------------------------------------------
 * Device & Hardware Query API
 * ----------------------------------------------------------------------------- */
int         qkrylov_is_gpu_build(void);
const char* qkrylov_find_gpu(void);
int         qkrylov_gpu_count(void);
int         qkrylov_initialize_device(const char* device_str);

/* -----------------------------------------------------------------------------
 * Matrix-Free Hamiltonian API
 * ----------------------------------------------------------------------------- */
qkrylov_hamiltonian_h qkrylov_hamiltonian_create(qkrylov_basis_h basis,
                                                qkrylov_site_h site,
                                                qkrylov_opsum_h opsum);
qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device(qkrylov_basis_h basis,
                                                        qkrylov_site_h site,
                                                        qkrylov_opsum_h opsum,
                                                        const char* device_str);
void                  qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h);
uint64_t              qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h);

/* Zero-copy matrix-vector apply: y = H * x */
/* x and y are pointers to complex float arrays of size dimension() */
int                   qkrylov_hamiltonian_apply(qkrylov_hamiltonian_h h,
                                                const float* x_real, const float* x_imag,
                                                float* y_real, float* y_imag);

/* Direct zero-copy complex apply: x_complex and y_complex are contiguous arrays of 2*dimension() doubles [re, im, re, im...] */
int                   qkrylov_hamiltonian_apply_complex(qkrylov_hamiltonian_h h,
                                                        const float* x_complex,
                                                        float* y_complex);

/* Extract matrix-free diagonal elements H_ii into caller-allocated array diag_out of size dimension() */
int                   qkrylov_hamiltonian_diagonal(qkrylov_hamiltonian_h h,
                                                   float* diag_out);

/* -----------------------------------------------------------------------------
 * Solvers API
 * ----------------------------------------------------------------------------- */
typedef struct {
    float energy;
    int iterations;
    int converged;
} qkrylov_lanczos_result_c_t;

typedef struct {
    int iterations;
    int converged;
} qkrylov_davidson_result_c_t;

int qkrylov_lanczos_ground_state(qkrylov_hamiltonian_h h,
                                 int maxiter,
                                 float tol,
                                 qkrylov_lanczos_result_c_t* result);

/* Ground state Lanczos solver with optional contiguous complex eigenvector output buffer [re, im, re, im...] */
int qkrylov_lanczos_ground_state_complex(qkrylov_hamiltonian_h h,
                                         int maxiter,
                                         float tol,
                                         qkrylov_lanczos_result_c_t* result,
                                         float* eigenvector_complex);

/* Davidson solver computing n_eig lowest eigenvalues and optional n_eig contiguous complex eigenvectors [re, im, re, im...] */
int qkrylov_davidson_lowest_complex(qkrylov_hamiltonian_h h,
                                    int n_eig,
                                    int max_subspace,
                                    float tol,
                                    float* eigenvalues_out,
                                    float* eigenvectors_complex_out,
                                    qkrylov_davidson_result_c_t* result_info);

/* Compute continued fraction tridiagonal coefficients (alphas, betas) starting from vector phi0_complex */
int qkrylov_continued_fraction_coeffs_complex(qkrylov_hamiltonian_h h,
                                              const float* phi0_complex,
                                              int n_iter,
                                              float* alphas_out,
                                              float* betas_out,
                                              float* norm_phi0_out,
                                              int* num_coeffs_out);

/* Evaluate spectral function I(omega) from continued fraction coefficients */
float qkrylov_evaluate_spectral_function(const float* alphas,
                                         const float* betas,
                                         size_t n,
                                         float norm_phi0,
                                         float omega,
                                         float E0,
                                         float eta);

/* Finite Temperature Lanczos Method (FTLM) Result */
typedef struct {
    float beta;                /* Inverse temperature beta = 1 / (kB * T) */
    float partition_function;  /* Thermal partition function Z(beta) */
    float internal_energy;     /* Internal energy E(beta) */
    float specific_heat;       /* Specific heat Cv(beta) */
} qkrylov_ftlm_result_c_t;

/* Finite Temperature Lanczos Method solver */
int qkrylov_ftlm(qkrylov_hamiltonian_h h,
                 float beta,
                 int n_random,
                 int n_steps,
                 qkrylov_ftlm_result_c_t* result);

#ifdef __cplusplus
}
#endif

#endif /* QKRYLOV_C_API_H */
