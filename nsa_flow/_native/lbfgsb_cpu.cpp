#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <tuple>
#include <string>
#include <stdexcept>
#include <chrono>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
extern "C" {
    void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, int* info);
    void sgesv_(const int* n, const int* nrhs, float* a, const int* lda, int* ipiv, float* b, const int* ldb, int* info);
}
#endif

namespace nsa_flow {

// --------------------------------------------------------------------------
// CBLAS and LAPACK zero-allocation wrappers
// --------------------------------------------------------------------------

template <typename T>
inline void cblas_gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                       int M, int N, int K,
                       T alpha, const T* A, int lda,
                       const T* B, int ldb,
                       T beta, T* C, int ldc);

template <>
inline void cblas_gemm<double>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                              int M, int N, int K,
                              double alpha, const double* A, int lda,
                              const double* B, int ldb,
                              double beta, double* C, int ldc) {
    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline void cblas_gemm<float>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                             int M, int N, int K,
                             float alpha, const float* A, int lda,
                             const float* B, int ldb,
                             float beta, float* C, int ldc) {
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
inline void cblas_gemv(CBLAS_TRANSPOSE TransA, int M, int N,
                       T alpha, const T* A, int lda,
                       const T* X, int incX,
                       T beta, T* Y, int incY);

template <>
inline void cblas_gemv<double>(CBLAS_TRANSPOSE TransA, int M, int N,
                               double alpha, const double* A, int lda,
                               const double* X, int incX,
                               double beta, double* Y, int incY) {
    cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template <>
inline void cblas_gemv<float>(CBLAS_TRANSPOSE TransA, int M, int N,
                              float alpha, const float* A, int lda,
                              const float* X, int incX,
                              float beta, float* Y, int incY) {
    cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template <typename T>
inline void lapack_gesv_raw(int n, int nrhs, T* a, int lda, int* ipiv, T* b, int ldb, int* info);

template <>
inline void lapack_gesv_raw<double>(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb, int* info) {
    dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
inline void lapack_gesv_raw<float>(int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb, int* info) {
    sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <typename T>
inline bool lapack_solve_row_major(int n, int nrhs, const T* A, const T* B, T* X) {
    T A_col_stack[64 * 64];
    T B_col_stack[64 * 64];
    int ipiv_stack[64];
    std::vector<T> A_col_heap, B_col_heap;
    std::vector<int> ipiv_heap;
    T* A_col = (n * n <= 4096) ? A_col_stack : (A_col_heap.resize(n * n), A_col_heap.data());
    T* B_col = (n * nrhs <= 4096) ? B_col_stack : (B_col_heap.resize(n * nrhs), B_col_heap.data());
    int* ipiv = (n <= 64) ? ipiv_stack : (ipiv_heap.resize(n), ipiv_heap.data());

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_col[j * n + i] = A[i * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nrhs; ++j) {
            B_col[j * n + i] = B[i * nrhs + j];
        }
    }
    int info = 0;
    lapack_gesv_raw<T>(n, nrhs, A_col, n, ipiv, B_col, n, &info);
    if (info != 0) return false;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nrhs; ++j) {
            X[i * nrhs + j] = B_col[j * n + i];
        }
    }
    return true;
}

template <typename T>
inline T safe_div(T num, T den, T tiny) {
    if (den >= static_cast<T>(0)) {
        return num / std::max(den, tiny);
    }
    return num / std::min(den, -tiny);
}

template <typename T>
inline bool solve_dense_system(int64_t dim, const T* A, const T* b, T* x) {
    if (dim > 64) {
        return lapack_solve_row_major<T>(static_cast<int>(dim), 1, A, b, x);
    }
    T M[64][65];
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            M[i][j] = A[i * dim + j];
        }
        M[i][dim] = b[i];
    }
    const T tiny = std::numeric_limits<T>::min();
    for (int64_t i = 0; i < dim; ++i) {
        int64_t max_row = i;
        T max_val = std::abs(M[i][i]);
        for (int64_t r = i + 1; r < dim; ++r) {
            T val = std::abs(M[r][i]);
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }
        if (max_val < tiny) return false;
        if (max_row != i) {
            for (int64_t j = i; j <= dim; ++j) {
                std::swap(M[i][j], M[max_row][j]);
            }
        }
        T diag = M[i][i];
        for (int64_t r = i + 1; r < dim; ++r) {
            T factor = M[r][i] / diag;
            for (int64_t j = i; j <= dim; ++j) {
                M[r][j] -= factor * M[i][j];
            }
        }
    }
    for (int64_t i = dim - 1; i >= 0; --i) {
        T sum = M[i][dim];
        for (int64_t j = i + 1; j < dim; ++j) {
            sum -= M[i][j] * x[j];
        }
        x[i] = sum / M[i][i];
    }
    return true;
}

template <typename T>
inline bool invert_dense_system(int64_t dim, const T* A, T* invA) {
    if (dim > 64) {
        int d = static_cast<int>(dim);
        std::vector<T> eye(d * d, static_cast<T>(0));
        for (int i = 0; i < d; ++i) eye[i * d + i] = static_cast<T>(1);
        return lapack_solve_row_major<T>(d, d, A, eye.data(), invA);
    }
    T M[64][128];
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            M[i][j] = A[i * dim + j];
            M[i][dim + j] = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
    const T tiny = std::numeric_limits<T>::min();
    for (int64_t i = 0; i < dim; ++i) {
        int64_t max_row = i;
        T max_val = std::abs(M[i][i]);
        for (int64_t r = i + 1; r < dim; ++r) {
            T val = std::abs(M[r][i]);
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }
        if (max_val < tiny) return false;
        if (max_row != i) {
            for (int64_t j = i; j < 2 * dim; ++j) {
                std::swap(M[i][j], M[max_row][j]);
            }
        }
        T diag = M[i][i];
        T inv_diag = static_cast<T>(1) / diag;
        for (int64_t j = i; j < 2 * dim; ++j) {
            M[i][j] *= inv_diag;
        }
        for (int64_t r = 0; r < dim; ++r) {
            if (r != i) {
                T factor = M[r][i];
                if (factor != static_cast<T>(0)) {
                    for (int64_t j = i; j < 2 * dim; ++j) {
                        M[r][j] -= factor * M[i][j];
                    }
                }
            }
        }
    }
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            invA[i * dim + j] = M[i][dim + j];
        }
    }
    return true;
}

template <typename scalar_t>
inline void clip_raw(int64_t n, const scalar_t* x, scalar_t* out,
                     bool has_lo, const scalar_t* lo,
                     bool has_hi, const scalar_t* hi,
                     bool has_mask, const scalar_t* mask) {
    for (int64_t i = 0; i < n; ++i) {
        scalar_t v = x[i];
        if (has_lo && v < lo[i]) v = lo[i];
        if (has_hi && v > hi[i]) v = hi[i];
        if (has_mask) v *= mask[i];
        out[i] = v;
    }
}

template <typename scalar_t>
inline double compute_a_max_raw(int64_t n, const scalar_t* d, const scalar_t* x,
                                bool has_lo, const scalar_t* lo,
                                bool has_hi, const scalar_t* hi) {
    double a_m = 1.0;
    const scalar_t tiny = std::numeric_limits<scalar_t>::min();
    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();

    if (has_lo) {
        scalar_t min_lim = inf;
        bool any_neg = false;
        for (int64_t i = 0; i < n; ++i) {
            if (d[i] < 0 && x[i] > lo[i]) {
                any_neg = true;
                scalar_t lim = (lo[i] - x[i]) / std::min(d[i], -tiny);
                if (lim < 0) lim = 0;
                if (lim < min_lim) min_lim = lim;
            }
        }
        if (any_neg) a_m = std::max(a_m, static_cast<double>(min_lim));
    }
    if (has_hi) {
        scalar_t min_lim = inf;
        bool any_pos = false;
        for (int64_t i = 0; i < n; ++i) {
            if (d[i] > 0 && x[i] < hi[i]) {
                any_pos = true;
                scalar_t lim = (hi[i] - x[i]) / std::max(d[i], tiny);
                if (lim < 0) lim = 0;
                if (lim < min_lim) min_lim = lim;
            }
        }
        if (any_pos) a_m = std::max(a_m, static_cast<double>(min_lim));
    }
    return a_m;
}

template <typename scalar_t>
inline double compute_gmap_raw(int64_t n, const scalar_t* x, const scalar_t* g,
                               bool has_lo, const scalar_t* lo,
                               bool has_hi, const scalar_t* hi,
                               bool has_mask, const scalar_t* mask) {
    double nY2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double xi = static_cast<double>(x[i]);
        nY2 += xi * xi;
    }
    double nY = std::sqrt(nY2);
    if (!std::isfinite(nY) || nY <= 0.0) return std::numeric_limits<double>::infinity();

    double diff_norm2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        scalar_t step_i = x[i] - static_cast<scalar_t>(nY2) * g[i];
        if (has_lo && step_i < lo[i]) step_i = lo[i];
        if (has_hi && step_i > hi[i]) step_i = hi[i];
        if (has_mask) step_i *= mask[i];
        double diff = static_cast<double>(step_i - x[i]);
        diff_norm2 += diff * diff;
    }
    return std::sqrt(diff_norm2) / nY;
}

// --------------------------------------------------------------------------
// L-BFGS-B Compact Matrix M Builder
// --------------------------------------------------------------------------

template <typename scalar_t>
inline bool lbfgsb_build_M_contiguous(int64_t n, int64_t m,
                                     const scalar_t* S_hist,
                                     const scalar_t* Y_hist,
                                     double theta, scalar_t* M_out) {
    if (m == 0) return true;
    int64_t two_m = 2 * m;
    scalar_t SY[32 * 32] = {0};
    scalar_t SS[32 * 32] = {0};

    cblas_gemm(CblasNoTrans, CblasTrans,
               static_cast<int>(m), static_cast<int>(m), static_cast<int>(n),
               static_cast<scalar_t>(1), S_hist, static_cast<int>(n),
               Y_hist, static_cast<int>(n),
               static_cast<scalar_t>(0), SY, static_cast<int>(m));

    cblas_gemm(CblasNoTrans, CblasTrans,
               static_cast<int>(m), static_cast<int>(m), static_cast<int>(n),
               static_cast<scalar_t>(1), S_hist, static_cast<int>(n),
               S_hist, static_cast<int>(n),
               static_cast<scalar_t>(0), SS, static_cast<int>(m));

    scalar_t K_stack[64 * 64];
    std::vector<scalar_t> K_heap;
    scalar_t* K = (two_m <= 64) ? K_stack : (K_heap.resize(two_m * two_m), K_heap.data());
    scalar_t th = static_cast<scalar_t>(theta);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            K[i * two_m + j] = (i == j) ? -SY[i * m + i] : static_cast<scalar_t>(0);
            K[i * two_m + (m + j)] = (j > i) ? SY[j * m + i] : static_cast<scalar_t>(0);
            K[(m + i) * two_m + j] = (i > j) ? SY[i * m + j] : static_cast<scalar_t>(0);
            K[(m + i) * two_m + (m + j)] = th * SS[i * m + j];
        }
    }

    return invert_dense_system<scalar_t>(two_m, K, M_out);
}

template <typename scalar_t>
inline bool lbfgsb_build_M_raw(int64_t n, int64_t m,
                              const scalar_t* const* S_ptrs,
                              const scalar_t* const* Y_ptrs,
                              double theta, scalar_t* M_out) {
    if (m == 0) return true;
    std::vector<scalar_t> S_contig(m * n);
    std::vector<scalar_t> Y_contig(m * n);
    for (int64_t i = 0; i < m; ++i) {
        std::memcpy(&S_contig[i * n], S_ptrs[i], n * sizeof(scalar_t));
        std::memcpy(&Y_contig[i * n], Y_ptrs[i], n * sizeof(scalar_t));
    }
    return lbfgsb_build_M_contiguous<scalar_t>(n, m, S_contig.data(), Y_contig.data(), theta, M_out);
}

template <typename scalar_t>
at::Tensor lbfgsb_build_M_impl(const at::Tensor& S_in, const at::Tensor& Y_in, double theta) {
    if (!S_in.defined() || S_in.numel() == 0 || S_in.size(1) == 0) {
        return at::empty({0, 0}, S_in.options());
    }
    int64_t n = S_in.size(0);
    int64_t m = S_in.size(1);
    int64_t two_m = 2 * m;

    auto S = S_in.contiguous();
    auto Y = Y_in.contiguous();
    const scalar_t* S_ptr = S.data_ptr<scalar_t>();
    const scalar_t* Y_ptr = Y.data_ptr<scalar_t>();

    scalar_t SY[32 * 32] = {0};
    scalar_t SS[32 * 32] = {0};

    cblas_gemm(CblasTrans, CblasNoTrans,
               static_cast<int>(m), static_cast<int>(m), static_cast<int>(n),
               static_cast<scalar_t>(1), S_ptr, static_cast<int>(m),
               Y_ptr, static_cast<int>(m),
               static_cast<scalar_t>(0), SY, static_cast<int>(m));

    cblas_gemm(CblasTrans, CblasNoTrans,
               static_cast<int>(m), static_cast<int>(m), static_cast<int>(n),
               static_cast<scalar_t>(1), S_ptr, static_cast<int>(m),
               S_ptr, static_cast<int>(m),
               static_cast<scalar_t>(0), SS, static_cast<int>(m));

    std::vector<scalar_t> K_vec(two_m * two_m, 0);
    scalar_t* K = K_vec.data();
    scalar_t th = static_cast<scalar_t>(theta);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            K[i * two_m + j] = (i == j) ? -SY[i * m + i] : static_cast<scalar_t>(0);
            K[i * two_m + (m + j)] = (j > i) ? SY[j * m + i] : static_cast<scalar_t>(0);
            K[(m + i) * two_m + j] = (i > j) ? SY[i * m + j] : static_cast<scalar_t>(0);
            K[(m + i) * two_m + (m + j)] = th * SS[i * m + j];
        }
    }

    auto M = at::empty({two_m, two_m}, S_in.options());
    scalar_t* M_out = M.data_ptr<scalar_t>();
    std::vector<scalar_t> eye(two_m * two_m, 0);
    for (int64_t i = 0; i < two_m; ++i) eye[i * two_m + i] = static_cast<scalar_t>(1);
    bool ok = lapack_solve_row_major<scalar_t>(static_cast<int>(two_m), static_cast<int>(two_m), K, eye.data(), M_out);
    if (!ok) {
        throw std::runtime_error("singular K in lbfgsb_build_M");
    }
    return M;
}

at::Tensor lbfgsb_build_M(const at::Tensor& S, const at::Tensor& Y, double theta) {
    if (!S.defined() || S.numel() == 0 || S.size(1) == 0) {
        return at::empty({0, 0}, S.options());
    }
    return AT_DISPATCH_FLOATING_TYPES(S.scalar_type(), "lbfgsb_build_M", ([&] {
        return lbfgsb_build_M_impl<scalar_t>(S, Y, theta);
    }));
}

// --------------------------------------------------------------------------
// L-BFGS-B Direction Computation (Cauchy Point & Subspace Minimisation)
// --------------------------------------------------------------------------

struct Breakpoint {
    double t;
    int64_t idx;
};

template <typename scalar_t>
struct DirectionWorkspace {
    std::vector<int64_t> free_idx;
    std::vector<scalar_t> W_free;
    std::vector<scalar_t> r_free;
    std::vector<scalar_t> Wfv;
    std::vector<scalar_t> WfWf;
    std::vector<scalar_t> N;
    std::vector<scalar_t> Wfr;
    std::vector<scalar_t> v0;
    std::vector<scalar_t> v;
    std::vector<scalar_t> Mc;
    std::vector<scalar_t> wmc;
    std::vector<scalar_t> W_buf;
    std::vector<Breakpoint> cand;
    std::vector<scalar_t> t;
    std::vector<uint8_t> fixed;
    std::vector<scalar_t> x_cp;
    std::vector<scalar_t> d_raw;
    std::vector<scalar_t> r;
    std::vector<scalar_t> d_hat;

    void resize(int64_t n, int64_t mem_cap) {
        int64_t max_2m = std::max<int64_t>(2 * mem_cap, 2);
        if (static_cast<int64_t>(free_idx.size()) < n) {
            free_idx.resize(n);
            r_free.resize(n);
            Wfv.resize(n);
            wmc.resize(n);
            cand.resize(n);
            t.resize(n);
            fixed.resize(n);
            x_cp.resize(n);
            d_raw.resize(n);
            r.resize(n);
            d_hat.resize(n);
        }
        if (static_cast<int64_t>(W_buf.size()) < max_2m * n) {
            W_buf.resize(max_2m * n);
            W_free.resize(max_2m * n);
        }
        if (static_cast<int64_t>(WfWf.size()) < max_2m * max_2m) {
            WfWf.resize(max_2m * max_2m);
            N.resize(max_2m * max_2m);
            Wfr.resize(max_2m);
            v0.resize(max_2m);
            v.resize(max_2m);
            Mc.resize(max_2m);
        }
    }
};

template <typename scalar_t>
inline void lbfgsb_direction_core(
    int64_t n, const scalar_t* x_ptr, const scalar_t* g_ptr,
    bool has_lo, const scalar_t* lo_ptr,
    bool has_hi, const scalar_t* hi_ptr,
    int64_t m, double theta, const scalar_t* M_ptr,
    const scalar_t* W_ptr,
    int64_t max_breakpoints,
    DirectionWorkspace<scalar_t>& ws,
    scalar_t* dir_ptr, double& a_max,
    uint8_t* fixed_out = nullptr,
    scalar_t* xcp_out = nullptr) {

    const scalar_t tiny = std::numeric_limits<scalar_t>::min();
    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();
    int64_t two_m = 2 * m;

    scalar_t* t = ws.t.data();
    Breakpoint* cand = ws.cand.data();
    uint8_t* fixed_ptr = fixed_out ? fixed_out : ws.fixed.data();
    scalar_t* x_cp_ptr = xcp_out ? xcp_out : ws.x_cp.data();
    scalar_t* d_ptr = ws.d_raw.data();
    scalar_t* r = ws.r.data();
    scalar_t* d_hat = ws.d_hat.data();

    for (int64_t i = 0; i < n; ++i) {
        scalar_t ti = inf;
        if (has_lo && g_ptr[i] > 0) {
            scalar_t den = std::max(g_ptr[i], tiny);
            ti = (x_ptr[i] - lo_ptr[i]) / den;
        }
        if (has_hi && g_ptr[i] < 0) {
            scalar_t den = std::min(g_ptr[i], -tiny);
            ti = (x_ptr[i] - hi_ptr[i]) / den;
        }
        if (std::isnan(ti) || std::isinf(ti)) {
            ti = inf;
        } else if (ti < 0) {
            ti = 0;
        }
        t[i] = ti;

        if (ti <= 0) {
            fixed_ptr[i] = 1;
            scalar_t bound = has_lo ? lo_ptr[i] : (has_hi ? hi_ptr[i] : static_cast<scalar_t>(0));
            x_cp_ptr[i] = bound;
            d_ptr[i] = 0.0;
        } else {
            fixed_ptr[i] = 0;
            x_cp_ptr[i] = x_ptr[i];
            d_ptr[i] = -g_ptr[i];
        }
    }

    scalar_t p[64] = {0};
    scalar_t c[64] = {0};

    if (two_m > 0 && W_ptr) {
        cblas_gemv(CblasNoTrans, static_cast<int>(two_m), static_cast<int>(n),
                   static_cast<scalar_t>(1), W_ptr, static_cast<int>(n),
                   d_ptr, 1, static_cast<scalar_t>(0), p, 1);
    }

    scalar_t fp = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        fp -= d_ptr[i] * d_ptr[i];
    }
    scalar_t fpp = -static_cast<scalar_t>(theta) * fp;

    if (two_m > 0 && M_ptr) {
        scalar_t pMp = 0;
        for (int64_t r_idx = 0; r_idx < two_m; ++r_idx) {
            scalar_t Mp_r = 0;
            for (int64_t k = 0; k < two_m; ++k) {
                Mp_r += M_ptr[r_idx * two_m + k] * p[k];
            }
            pMp += Mp_r * p[r_idx];
        }
        fpp -= pMp;
    }
    fpp = std::max(fpp, tiny);
    scalar_t dt_min = safe_div(-fp, fpp, tiny);

    int64_t n_cand = 0;
    for (int64_t i = 0; i < n; ++i) {
        scalar_t ti = t[i];
        if (ti > 0 && ti < inf) {
            cand[n_cand].t = static_cast<double>(ti);
            cand[n_cand].idx = i;
            n_cand++;
        }
    }

    scalar_t t_old = 0.0;
    if (n_cand > 0) {
        int64_t num_bk = std::min(n_cand, max_breakpoints);
        std::partial_sort(cand, cand + num_bk, cand + n_cand,
                          [](const Breakpoint& a, const Breakpoint& b) {
                              return a.t < b.t;
                          });

        for (int64_t j = 0; j < num_bk; ++j) {
            scalar_t tj = static_cast<scalar_t>(cand[j].t);
            int64_t i = cand[j].idx;
            scalar_t dt = tj - t_old;
            if (dt_min < dt || dt <= 0.0) {
                break;
            }

            scalar_t xcp_b = 0;
            if (has_lo && has_hi) {
                xcp_b = (g_ptr[i] > 0) ? lo_ptr[i] : hi_ptr[i];
            } else if (has_lo) {
                xcp_b = lo_ptr[i];
            } else {
                xcp_b = hi_ptr[i];
            }
            scalar_t zb = xcp_b - x_ptr[i];
            scalar_t gi = g_ptr[i];

            if (two_m > 0) {
                for (int64_t k = 0; k < two_m; ++k) {
                    c[k] += dt * p[k];
                }
            }

            scalar_t wMc = 0;
            scalar_t dfpp = 0;
            scalar_t fpp_prev = fpp;

            if (two_m > 0 && M_ptr && W_ptr) {
                scalar_t w_col[64];
                for (int64_t k = 0; k < two_m; ++k) {
                    w_col[k] = W_ptr[k * n + i];
                }
                scalar_t MW_i[64] = {0};
                for (int64_t r_idx = 0; r_idx < two_m; ++r_idx) {
                    scalar_t sum_val = 0;
                    for (int64_t k = 0; k < two_m; ++k) {
                        sum_val += M_ptr[r_idx * two_m + k] * w_col[k];
                    }
                    MW_i[r_idx] = sum_val;
                }
                scalar_t wMp = 0;
                scalar_t wMw = 0;
                for (int64_t k = 0; k < two_m; ++k) {
                    wMc += MW_i[k] * c[k];
                    wMp += MW_i[k] * p[k];
                    wMw += MW_i[k] * w_col[k];
                }
                dfpp = -static_cast<scalar_t>(theta) * gi * gi - 2.0 * gi * wMp - gi * gi * wMw;
            } else {
                wMc = 0;
                dfpp = -static_cast<scalar_t>(theta) * gi * gi;
            }

            fpp = std::max(fpp + dfpp, tiny);
            scalar_t dfp = dt * fpp_prev + gi * gi + static_cast<scalar_t>(theta) * gi * zb - gi * wMc;
            fp += dfp;
            dt_min = safe_div(-fp, fpp, tiny);

            if (two_m > 0 && W_ptr) {
                for (int64_t k = 0; k < two_m; ++k) {
                    p[k] += gi * W_ptr[k * n + i];
                }
            }

            d_ptr[i] = 0.0;
            x_cp_ptr[i] = xcp_b;
            fixed_ptr[i] = 1;
            t_old = tj;
        }
    }

    dt_min = std::max(dt_min, static_cast<scalar_t>(0));
    t_old += dt_min;
    for (int64_t i = 0; i < n; ++i) {
        if (!fixed_ptr[i] && t[i] > t_old) {
            x_cp_ptr[i] = x_ptr[i] + t_old * d_ptr[i];
        }
    }
    if (two_m > 0) {
        for (int64_t k = 0; k < two_m; ++k) {
            c[k] += dt_min * p[k];
        }
    }

    int64_t n_free = 0;
    int64_t* free_idx = ws.free_idx.data();
    for (int64_t i = 0; i < n; ++i) {
        if (!fixed_ptr[i]) {
            free_idx[n_free++] = i;
        }
    }

    if (n_free > 0) {
        scalar_t th = static_cast<scalar_t>(theta);
        scalar_t* r_free = ws.r_free.data();
        scalar_t* W_free = ws.W_free.data();

        if (two_m == 0 || !M_ptr || !W_ptr) {
            for (int64_t j = 0; j < n_free; ++j) {
                int64_t i = free_idx[j];
                scalar_t zi = x_cp_ptr[i] - x_ptr[i];
                r[i] = g_ptr[i] + th * zi;
                d_hat[i] = -r[i] / th;
            }
            for (int64_t i = 0; i < n; ++i) {
                if (fixed_ptr[i]) d_hat[i] = 0.0;
            }
        } else {
            scalar_t* Mc = ws.Mc.data();
            cblas_gemv(CblasNoTrans, static_cast<int>(two_m), static_cast<int>(two_m),
                       static_cast<scalar_t>(1), M_ptr, static_cast<int>(two_m),
                       c, 1, static_cast<scalar_t>(0), Mc, 1);

            // Pack W_free: rows 0..2m-1, columns 0..n_free-1
            for (int64_t k = 0; k < two_m; ++k) {
                const scalar_t* W_row = &W_ptr[k * n];
                scalar_t* Wf_row = &W_free[k * n_free];
                for (int64_t j = 0; j < n_free; ++j) {
                    Wf_row[j] = W_row[free_idx[j]];
                }
            }

            scalar_t* wmc = ws.wmc.data();
            cblas_gemv(CblasTrans, static_cast<int>(two_m), static_cast<int>(n_free),
                       static_cast<scalar_t>(1), W_free, static_cast<int>(n_free),
                       Mc, 1, static_cast<scalar_t>(0), wmc, 1);

            for (int64_t j = 0; j < n_free; ++j) {
                int64_t i = free_idx[j];
                scalar_t zi = x_cp_ptr[i] - x_ptr[i];
                scalar_t ri = g_ptr[i] + th * zi - wmc[j];
                r[i] = ri;
                r_free[j] = ri;
            }
            for (int64_t i = 0; i < n; ++i) {
                if (fixed_ptr[i]) r[i] = 0.0;
            }

            scalar_t* WfWf = ws.WfWf.data();
            cblas_gemm(CblasNoTrans, CblasTrans,
                       static_cast<int>(two_m), static_cast<int>(two_m), static_cast<int>(n_free),
                       static_cast<scalar_t>(1), W_free, static_cast<int>(n_free),
                       W_free, static_cast<int>(n_free),
                       static_cast<scalar_t>(0), WfWf, static_cast<int>(two_m));

            scalar_t* Wfr = ws.Wfr.data();
            cblas_gemv(CblasNoTrans, static_cast<int>(two_m), static_cast<int>(n_free),
                       static_cast<scalar_t>(1), W_free, static_cast<int>(n_free),
                       r_free, 1, static_cast<scalar_t>(0), Wfr, 1);

            scalar_t* v0 = ws.v0.data();
            cblas_gemv(CblasNoTrans, static_cast<int>(two_m), static_cast<int>(two_m),
                       static_cast<scalar_t>(1), M_ptr, static_cast<int>(two_m),
                       Wfr, 1, static_cast<scalar_t>(0), v0, 1);

            scalar_t* N = ws.N.data();
            scalar_t inv_th = static_cast<scalar_t>(1.0) / th;
            cblas_gemm(CblasNoTrans, CblasNoTrans,
                       static_cast<int>(two_m), static_cast<int>(two_m), static_cast<int>(two_m),
                       -inv_th, M_ptr, static_cast<int>(two_m),
                       WfWf, static_cast<int>(two_m),
                       static_cast<scalar_t>(0), N, static_cast<int>(two_m));
            for (int64_t k = 0; k < two_m; ++k) {
                N[k * two_m + k] += static_cast<scalar_t>(1.0);
            }

            scalar_t* v = ws.v.data();
            bool solved = solve_dense_system<scalar_t>(two_m, N, v0, v);
            if (!solved) {
                std::fill(v, v + two_m, static_cast<scalar_t>(0));
            }

            scalar_t* Wfv = ws.Wfv.data();
            cblas_gemv(CblasTrans, static_cast<int>(two_m), static_cast<int>(n_free),
                       static_cast<scalar_t>(1), W_free, static_cast<int>(n_free),
                       v, 1, static_cast<scalar_t>(0), Wfv, 1);

            scalar_t inv_th2 = static_cast<scalar_t>(1.0) / (th * th);
            for (int64_t j = 0; j < n_free; ++j) {
                int64_t i = free_idx[j];
                d_hat[i] = -r_free[j] * inv_th - Wfv[j] * inv_th2;
            }
            for (int64_t i = 0; i < n; ++i) {
                if (fixed_ptr[i]) d_hat[i] = 0.0;
            }
        }

        scalar_t alpha = 1.0;
        if (has_lo) {
            for (int64_t j = 0; j < n_free; ++j) {
                int64_t i = free_idx[j];
                if (d_hat[i] < 0) {
                    scalar_t lim = (lo_ptr[i] - x_cp_ptr[i]) / std::min(d_hat[i], -tiny);
                    if (lim < 0) lim = 0;
                    if (lim < alpha) alpha = lim;
                }
            }
        }
        if (has_hi) {
            for (int64_t j = 0; j < n_free; ++j) {
                int64_t i = free_idx[j];
                if (d_hat[i] > 0) {
                    scalar_t lim = (hi_ptr[i] - x_cp_ptr[i]) / std::max(d_hat[i], tiny);
                    if (lim < 0) lim = 0;
                    if (lim < alpha) alpha = lim;
                }
            }
        }

        for (int64_t i = 0; i < n; ++i) {
            scalar_t d_val = (x_cp_ptr[i] + alpha * d_hat[i]) - x_ptr[i];
            if (has_lo && x_ptr[i] <= lo_ptr[i] && d_val < 0) d_val = 0.0;
            if (has_hi && x_ptr[i] >= hi_ptr[i] && d_val > 0) d_val = 0.0;
            dir_ptr[i] = d_val;
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            scalar_t d_val = x_cp_ptr[i] - x_ptr[i];
            if (has_lo && x_ptr[i] <= lo_ptr[i] && d_val < 0) d_val = 0.0;
            if (has_hi && x_ptr[i] >= hi_ptr[i] && d_val > 0) d_val = 0.0;
            dir_ptr[i] = d_val;
        }
    }

    a_max = compute_a_max_raw(n, dir_ptr, x_ptr, has_lo, lo_ptr, has_hi, hi_ptr);
}

template <typename scalar_t>
std::tuple<at::Tensor, double, at::Tensor, at::Tensor>
lbfgsb_direction_impl(const at::Tensor& x_in, const at::Tensor& g_in,
                      const c10::optional<at::Tensor>& lo_opt,
                      const c10::optional<at::Tensor>& hi_opt,
                      const at::Tensor& S_in, const at::Tensor& Y_in, double theta,
                      const at::Tensor& M_in, int64_t max_breakpoints) {
    auto x = x_in.contiguous();
    auto g = g_in.contiguous();
    int64_t n = x.numel();

    bool has_lo = lo_opt.has_value() && lo_opt->defined() && lo_opt->numel() > 0;
    bool has_hi = hi_opt.has_value() && hi_opt->defined() && hi_opt->numel() > 0;
    at::Tensor lo = has_lo ? lo_opt->contiguous() : at::Tensor();
    at::Tensor hi = has_hi ? hi_opt->contiguous() : at::Tensor();

    const scalar_t* x_ptr = x.data_ptr<scalar_t>();
    const scalar_t* g_ptr = g.data_ptr<scalar_t>();
    const scalar_t* lo_ptr = has_lo ? lo.data_ptr<scalar_t>() : nullptr;
    const scalar_t* hi_ptr = has_hi ? hi.data_ptr<scalar_t>() : nullptr;

    int64_t m = (S_in.defined() && S_in.numel() > 0) ? S_in.size(1) : 0;

    auto S = (m > 0) ? S_in.contiguous() : at::Tensor();
    auto Y = (m > 0) ? Y_in.contiguous() : at::Tensor();
    const scalar_t* S_ptr = (m > 0) ? S.data_ptr<scalar_t>() : nullptr;
    const scalar_t* Y_ptr = (m > 0) ? Y.data_ptr<scalar_t>() : nullptr;

    at::Tensor M_contig = (m > 0 && M_in.defined() && M_in.numel() > 0) ? M_in.contiguous() : at::Tensor();
    const scalar_t* M_ptr = (M_contig.defined() && M_contig.numel() > 0) ? M_contig.data_ptr<scalar_t>() : nullptr;

    DirectionWorkspace<scalar_t> ws;
    ws.resize(n, m);

    if (m > 0) {
        for (int64_t j = 0; j < m; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                ws.W_buf[j * n + i] = Y_ptr[i * m + j];
                ws.W_buf[(m + j) * n + i] = static_cast<scalar_t>(theta) * S_ptr[i * m + j];
            }
        }
    }

    at::Tensor fixed = at::zeros({n}, x.options().dtype(at::kBool));
    uint8_t* fixed_ptr = reinterpret_cast<uint8_t*>(fixed.data_ptr<bool>());

    at::Tensor x_cp = at::empty_like(x);
    scalar_t* x_cp_ptr = x_cp.data_ptr<scalar_t>();

    at::Tensor dir = at::empty_like(x);
    scalar_t* dir_ptr = dir.data_ptr<scalar_t>();
    double a_max = 1.0;

    lbfgsb_direction_core<scalar_t>(
        n, x_ptr, g_ptr, has_lo, lo_ptr, has_hi, hi_ptr,
        m, theta, M_ptr, (m > 0) ? ws.W_buf.data() : nullptr,
        max_breakpoints, ws, dir_ptr, a_max, fixed_ptr, x_cp_ptr);

    return std::make_tuple(dir, a_max, fixed, x_cp);
}

std::tuple<at::Tensor, double, at::Tensor>
lbfgsb_direction(const at::Tensor& x, const at::Tensor& g,
                 const c10::optional<at::Tensor>& lo,
                 const c10::optional<at::Tensor>& hi,
                 const at::Tensor& S, const at::Tensor& Y, double theta,
                 const at::Tensor& M, int64_t max_breakpoints) {
    auto res = AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lbfgsb_direction", ([&] {
        return lbfgsb_direction_impl<scalar_t>(x, g, lo, hi, S, Y, theta, M, max_breakpoints);
    }));
    return std::make_tuple(std::get<0>(res), std::get<1>(res), std::get<2>(res));
}

std::tuple<at::Tensor, double, at::Tensor, at::Tensor>
lbfgsb_direction_internal(const at::Tensor& x, const at::Tensor& g,
                          const c10::optional<at::Tensor>& lo,
                          const c10::optional<at::Tensor>& hi,
                          const at::Tensor& S, const at::Tensor& Y, double theta,
                          const at::Tensor& M, int64_t max_breakpoints) {
    return AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lbfgsb_direction_internal", ([&] {
        return lbfgsb_direction_impl<scalar_t>(x, g, lo, hi, S, Y, theta, M, max_breakpoints);
    }));
}

// --------------------------------------------------------------------------
// Objective Workspaces and High-Performance Fused Objectives
// --------------------------------------------------------------------------

template <typename scalar_t>
struct ObjectiveWorkspace {
    std::vector<scalar_t> SV;  // p * k
    std::vector<scalar_t> XV;  // n_rows * k
    std::vector<scalar_t> A;   // k * k
    std::vector<scalar_t> B;   // k * k
    std::vector<scalar_t> G;   // k * k

    std::vector<scalar_t> V;   // p * k
    std::vector<scalar_t> gV;  // p * k
    std::vector<scalar_t> BW;  // 2k * 2k
    std::vector<scalar_t> GW;  // 2k * 2k

    std::vector<scalar_t> rhs;  // k * k
    std::vector<scalar_t> coef; // k * k
    std::vector<scalar_t> R;    // p * k
    std::vector<scalar_t> gF;   // p * k

    void init_data(int64_t p, int64_t k, int64_t n_rows) {
        SV.assign(p * k, 0);
        if (n_rows > 0) XV.assign(n_rows * k, 0);
        A.assign(k * k, 0);
        B.assign(k * k, 0);
        G.assign(k * k, 0);
    }

    void init_signed(int64_t p, int64_t k, int64_t n_rows) {
        init_data(p, k, n_rows);
        V.assign(p * k, 0);
        gV.assign(p * k, 0);
        int64_t two_k = 2 * k;
        BW.assign(two_k * two_k, 0);
        GW.assign(two_k * two_k, 0);
    }

    void init_anchored(int64_t p, int64_t k, const std::string& fidelity) {
        A.assign(k * k, 0);
        B.assign(k * k, 0);
        G.assign(k * k, 0);
        gF.assign(p * k, 0);
        if (fidelity == "subspace") {
            rhs.assign(k * k, 0);
            coef.assign(k * k, 0);
            R.assign(p * k, 0);
        }
    }
};

template <typename scalar_t>
inline double eval_data_objective_raw(
    int64_t p, int64_t k, const scalar_t* V,
    int64_t n_rows, const scalar_t* X, const scalar_t* S,
    double c_val, double w_val, const std::string& orth,
    bool eval_grad, scalar_t* grad_out, ObjectiveWorkspace<scalar_t>& ws) {

    scalar_t c = static_cast<scalar_t>(c_val);
    scalar_t w = static_cast<scalar_t>(w_val);
    bool is_orth_cg = (orth == "Cg");
    const scalar_t tiny = std::numeric_limits<scalar_t>::min();

    scalar_t* A = ws.A.data();
    scalar_t* B = ws.B.data();
    scalar_t* G = ws.G.data();
    scalar_t* SV = ws.SV.data();
    scalar_t* XV = ws.XV.data();

    std::fill(A, A + k * k, static_cast<scalar_t>(0));

    if (S != nullptr) {
        cblas_gemm(CblasNoTrans, CblasNoTrans, static_cast<int>(p), static_cast<int>(k), static_cast<int>(p),
                   static_cast<scalar_t>(1), S, static_cast<int>(p), V, static_cast<int>(k),
                   static_cast<scalar_t>(0), SV, static_cast<int>(k));
        cblas_gemm(CblasTrans, CblasNoTrans, static_cast<int>(k), static_cast<int>(k), static_cast<int>(p),
                   static_cast<scalar_t>(1), V, static_cast<int>(k), SV, static_cast<int>(k),
                   static_cast<scalar_t>(0), A, static_cast<int>(k));
    } else if (X != nullptr && n_rows > 0) {
        cblas_gemm(CblasNoTrans, CblasNoTrans, static_cast<int>(n_rows), static_cast<int>(k), static_cast<int>(p),
                   static_cast<scalar_t>(1), X, static_cast<int>(p), V, static_cast<int>(k),
                   static_cast<scalar_t>(0), XV, static_cast<int>(k));
        cblas_gemm(CblasTrans, CblasNoTrans, static_cast<int>(k), static_cast<int>(k), static_cast<int>(n_rows),
                   static_cast<scalar_t>(1), XV, static_cast<int>(k), XV, static_cast<int>(k),
                   static_cast<scalar_t>(0), A, static_cast<int>(k));
        if (eval_grad) {
            cblas_gemm(CblasTrans, CblasNoTrans, static_cast<int>(p), static_cast<int>(k), static_cast<int>(n_rows),
                       static_cast<scalar_t>(1), X, static_cast<int>(p), XV, static_cast<int>(k),
                       static_cast<scalar_t>(0), SV, static_cast<int>(k));
        }
    }
    cblas_gemm(CblasTrans, CblasNoTrans, static_cast<int>(k), static_cast<int>(k), static_cast<int>(p),
               static_cast<scalar_t>(1), V, static_cast<int>(k), V, static_cast<int>(k),
               static_cast<scalar_t>(0), B, static_cast<int>(k));

    scalar_t trA = 0;
    for (int64_t i = 0; i < k; ++i) trA += A[i * k + i];
    scalar_t AB = 0;
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            AB += A[i * k + j] * B[j * k + i];
        }
    }
    scalar_t F = (c - 2.0 * trA + AB) / c;

    scalar_t D = 0;
    scalar_t t = 0;
    for (int64_t i = 0; i < k; ++i) t += B[i * k + i];
    scalar_t a2 = 0;
    scalar_t scale_k_cg = (k > 1) ? (1.0 - 1.0 / static_cast<scalar_t>(k)) : static_cast<scalar_t>(1);
    scalar_t scale_k_D = (k > 1) ? (1.0 - 1.0 / static_cast<scalar_t>(k)) : static_cast<scalar_t>(1);
    scalar_t N2 = 0;

    if (k > 1) {
        if (is_orth_cg) {
            if (t < tiny) t = tiny;
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    if (i != j) {
                        scalar_t val = B[i * k + j];
                        a2 += val * val;
                    }
                }
            }
            D = a2 / (t * t * scale_k_cg);
        } else {
            if (t < tiny) t = tiny;
            scalar_t inv_k = 1.0 / static_cast<scalar_t>(k);
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    scalar_t g_val = B[i * k + j] / t;
                    G[i * k + j] = g_val;
                    N2 += g_val * g_val;
                    scalar_t gc = g_val - (i == j ? inv_k : static_cast<scalar_t>(0));
                    D += gc * gc;
                }
            }
            D /= scale_k_D;
        }
    }

    double E = static_cast<double>((1.0 - w) * F + (k > 1 ? w * D : static_cast<scalar_t>(0)));

    if (eval_grad && grad_out != nullptr) {
        scalar_t factor_F = 2.0 / c;
        scalar_t factor_D_D = (k > 1) ? (4.0 / (scale_k_D * t)) : static_cast<scalar_t>(0);
        scalar_t factor_D_cg = (k > 1) ? (4.0 / (t * t * scale_k_cg)) : static_cast<scalar_t>(0);
        scalar_t scale_cg = (t > 0) ? (a2 / t) : static_cast<scalar_t>(0);

        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                scalar_t svb = 0, va = 0;
                for (int64_t m = 0; m < k; ++m) {
                    svb += SV[r * k + m] * B[m * k + j];
                    va += V[r * k + m] * A[m * k + j];
                }
                scalar_t gF = factor_F * (-2.0 * SV[r * k + j] + svb + va);

                scalar_t gD = 0;
                if (k > 1 && w > 0.0) {
                    if (is_orth_cg) {
                        scalar_t v_off = 0;
                        for (int64_t m = 0; m < k; ++m) {
                            if (m != j) v_off += V[r * k + m] * B[m * k + j];
                        }
                        gD = factor_D_cg * (v_off - scale_cg * V[r * k + j]);
                    } else {
                        scalar_t vg = 0;
                        for (int64_t m = 0; m < k; ++m) {
                            vg += V[r * k + m] * G[m * k + j];
                        }
                        gD = factor_D_D * (vg - N2 * V[r * k + j]);
                    }
                }
                grad_out[r * k + j] = (1.0 - w) * gF + (w > 0.0 && k > 1 ? w * gD : static_cast<scalar_t>(0));
            }
        }
    }
    return E;
}

template <typename scalar_t>
inline double eval_signed_objective_raw(
    int64_t p, int64_t two_k, const scalar_t* W,
    int64_t n_rows, const scalar_t* X, const scalar_t* S,
    double c_val, double w_val, const std::string& orth, double lobe_val,
    bool eval_grad, scalar_t* grad_out, ObjectiveWorkspace<scalar_t>& ws) {

    int64_t k = two_k / 2;
    scalar_t c = static_cast<scalar_t>(c_val);
    scalar_t w = static_cast<scalar_t>(w_val);
    const scalar_t tiny = std::numeric_limits<scalar_t>::min();

    scalar_t* V = ws.V.data();
    scalar_t* gV = ws.gV.data();

    for (int64_t r = 0; r < p; ++r) {
        for (int64_t j = 0; j < k; ++j) {
            V[r * k + j] = W[r * two_k + j] - W[r * two_k + k + j];
        }
    }

    double F = eval_data_objective_raw(p, k, V, n_rows, X, S, c_val, 0.0, "", eval_grad, gV, ws);

    scalar_t* BW = ws.BW.data();
    scalar_t* GW = ws.GW.data();

    cblas_gemm(CblasTrans, CblasNoTrans, static_cast<int>(two_k), static_cast<int>(two_k), static_cast<int>(p),
               static_cast<scalar_t>(1), W, static_cast<int>(two_k), W, static_cast<int>(two_k),
               static_cast<scalar_t>(0), BW, static_cast<int>(two_k));

    scalar_t D = 0;
    scalar_t tW = 0;
    for (int64_t i = 0; i < two_k; ++i) tW += BW[i * two_k + i];
    scalar_t a2 = 0;
    scalar_t scale_k_cg = 1.0 - 1.0 / static_cast<scalar_t>(two_k);
    scalar_t scale_k_D = 1.0 - 1.0 / static_cast<scalar_t>(two_k);
    scalar_t N2 = 0;

    if (orth == "D") {
        if (tW < tiny) tW = tiny;
        scalar_t inv_k = 1.0 / static_cast<scalar_t>(two_k);
        for (int64_t i = 0; i < two_k; ++i) {
            for (int64_t j = 0; j < two_k; ++j) {
                scalar_t g_val = BW[i * two_k + j] / tW;
                GW[i * two_k + j] = g_val;
                N2 += g_val * g_val;
                scalar_t gc = g_val - (i == j ? inv_k : static_cast<scalar_t>(0));
                D += gc * gc;
            }
        }
        D /= scale_k_D;
    } else { // "Cg" default for signed
        if (tW < tiny) tW = tiny;
        for (int64_t i = 0; i < two_k; ++i) {
            for (int64_t j = 0; j < two_k; ++j) {
                if (i != j) {
                    scalar_t val = BW[i * two_k + j];
                    a2 += val * val;
                }
            }
        }
        D = a2 / (tW * tW * scale_k_cg);
    }

    double E_lobe = 0.0;
    if (lobe_val > 0.0 && w > 0.0) {
        scalar_t dot = 0;
        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                dot += W[r * two_k + j] * W[r * two_k + k + j];
            }
        }
        E_lobe = static_cast<double>((w * static_cast<scalar_t>(lobe_val) / c) * dot);
    }

    double E = (1.0 - w_val) * F + w_val * static_cast<double>(D) + E_lobe;

    if (eval_grad && grad_out != nullptr) {
        scalar_t factor_D_D = 4.0 / (scale_k_D * tW);
        scalar_t factor_D_cg = 4.0 / (tW * tW * scale_k_cg);
        scalar_t scale_cg = (tW > 0) ? (a2 / tW) : static_cast<scalar_t>(0);
        scalar_t lobe_coef = static_cast<scalar_t>(w_val * lobe_val / c_val);

        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < two_k; ++j) {
                scalar_t gD = 0;
                if (w > 0.0) {
                    if (orth == "D") {
                        scalar_t wg = 0;
                        for (int64_t m = 0; m < two_k; ++m) {
                            wg += W[r * two_k + m] * GW[m * two_k + j];
                        }
                        gD = factor_D_D * (wg - N2 * W[r * two_k + j]);
                    } else {
                        scalar_t w_off = 0;
                        for (int64_t m = 0; m < two_k; ++m) {
                            if (m != j) w_off += W[r * two_k + m] * BW[m * two_k + j];
                        }
                        gD = factor_D_cg * (w_off - scale_cg * W[r * two_k + j]);
                    }
                }
                scalar_t gW_F = (j < k) ? gV[r * k + j] : -gV[r * k + (j - k)];
                scalar_t g_lobe = 0;
                if (lobe_val > 0.0 && w > 0.0) {
                    g_lobe = lobe_coef * ((j < k) ? W[r * two_k + k + j] : W[r * two_k + (j - k)]);
                }
                grad_out[r * two_k + j] = static_cast<scalar_t>(1.0 - w_val) * gW_F + static_cast<scalar_t>(w_val) * gD + g_lobe;
            }
        }
    }
    return E;
}

template <typename scalar_t>
inline double eval_anchored_objective_raw(
    int64_t p, int64_t k, const scalar_t* Y,
    const scalar_t* X0, double denom_val,
    const std::string& fidelity, const scalar_t* chol,
    double w_val, const std::string& orth,
    bool eval_grad, scalar_t* grad_out, ObjectiveWorkspace<scalar_t>& ws) {

    scalar_t denom = static_cast<scalar_t>(denom_val);
    scalar_t w = static_cast<scalar_t>(w_val);
    bool is_orth_cg = (orth == "Cg");
    const scalar_t tiny = std::numeric_limits<scalar_t>::min();

    scalar_t* gF = ws.gF.data();
    scalar_t F = 0.0;

    if (fidelity == "subspace") {
        scalar_t* rhs = ws.rhs.data();
        scalar_t* coef = ws.coef.data();
        scalar_t* R = ws.R.data();

        cblas_gemm(CblasTrans, CblasNoTrans, static_cast<int>(k), static_cast<int>(k), static_cast<int>(p),
                   static_cast<scalar_t>(1), X0, static_cast<int>(k), Y, static_cast<int>(k),
                   static_cast<scalar_t>(0), rhs, static_cast<int>(k));

        for (int64_t col = 0; col < k; ++col) {
            for (int64_t i = 0; i < k; ++i) {
                scalar_t s = rhs[i * k + col];
                for (int64_t j = 0; j < i; ++j) {
                    s -= chol[i * k + j] * coef[j * k + col];
                }
                coef[i * k + col] = s / chol[i * k + i];
            }
            for (int64_t i = k - 1; i >= 0; --i) {
                scalar_t s = coef[i * k + col];
                for (int64_t j = i + 1; j < k; ++j) {
                    s -= chol[j * k + i] * coef[j * k + col];
                }
                coef[i * k + col] = s / chol[i * k + i];
            }
        }

        scalar_t R2 = 0, nY2 = 0;
        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                scalar_t py = 0;
                for (int64_t m = 0; m < k; ++m) {
                    py += X0[r * k + m] * coef[m * k + j];
                }
                scalar_t r_val = Y[r * k + j] - py;
                R[r * k + j] = r_val;
                R2 += r_val * r_val;
                nY2 += Y[r * k + j] * Y[r * k + j];
            }
        }
        if (nY2 < tiny) nY2 = tiny;
        F = R2 / nY2;

        if (eval_grad) {
            scalar_t factor_sub = 2.0 / nY2;
            for (int64_t idx = 0; idx < p * k; ++idx) {
                gF[idx] = factor_sub * (R[idx] - F * Y[idx]);
            }
        }
    } else { // "anchor"
        scalar_t R2 = 0;
        for (int64_t idx = 0; idx < p * k; ++idx) {
            scalar_t diff = Y[idx] - X0[idx];
            R2 += diff * diff;
            if (eval_grad) gF[idx] = (2.0 / denom) * diff;
        }
        F = R2 / denom;
    }

    scalar_t* B = ws.B.data();
    scalar_t* G = ws.G.data();

    cblas_gemm(CblasTrans, CblasNoTrans, static_cast<int>(k), static_cast<int>(k), static_cast<int>(p),
               static_cast<scalar_t>(1), Y, static_cast<int>(k), Y, static_cast<int>(k),
               static_cast<scalar_t>(0), B, static_cast<int>(k));

    scalar_t D = 0;
    scalar_t t = 0;
    for (int64_t i = 0; i < k; ++i) t += B[i * k + i];
    scalar_t a2 = 0;
    scalar_t scale_k_cg = (k > 1) ? (1.0 - 1.0 / static_cast<scalar_t>(k)) : static_cast<scalar_t>(1);
    scalar_t scale_k_D = (k > 1) ? (1.0 - 1.0 / static_cast<scalar_t>(k)) : static_cast<scalar_t>(1);
    scalar_t N2 = 0;

    if (k > 1) {
        if (is_orth_cg) {
            if (t < tiny) t = tiny;
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    if (i != j) {
                        scalar_t val = B[i * k + j];
                        a2 += val * val;
                    }
                }
            }
            D = a2 / (t * t * scale_k_cg);
        } else {
            if (t < tiny) t = tiny;
            scalar_t inv_k = 1.0 / static_cast<scalar_t>(k);
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    scalar_t g_val = B[i * k + j] / t;
                    G[i * k + j] = g_val;
                    N2 += g_val * g_val;
                    scalar_t gc = g_val - (i == j ? inv_k : static_cast<scalar_t>(0));
                    D += gc * gc;
                }
            }
            D /= scale_k_D;
        }
    }

    double E = static_cast<double>((1.0 - w) * F + (k > 1 ? w * D : static_cast<scalar_t>(0)));

    if (eval_grad && grad_out != nullptr) {
        scalar_t factor_D_D = (k > 1) ? (4.0 / (scale_k_D * t)) : static_cast<scalar_t>(0);
        scalar_t factor_D_cg = (k > 1) ? (4.0 / (t * t * scale_k_cg)) : static_cast<scalar_t>(0);
        scalar_t scale_cg = (t > 0) ? (a2 / t) : static_cast<scalar_t>(0);

        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                scalar_t gF_val = gF[r * k + j];
                scalar_t gD = 0;
                if (k > 1 && w > 0.0) {
                    if (is_orth_cg) {
                        scalar_t v_off = 0;
                        for (int64_t m = 0; m < k; ++m) {
                            if (m != j) v_off += Y[r * k + m] * B[m * k + j];
                        }
                        gD = factor_D_cg * (v_off - scale_cg * Y[r * k + j]);
                    } else {
                        scalar_t vg = 0;
                        for (int64_t m = 0; m < k; ++m) {
                            vg += Y[r * k + m] * G[m * k + j];
                        }
                        gD = factor_D_D * (vg - N2 * Y[r * k + j]);
                    }
                }
                grad_out[r * k + j] = (1.0 - w) * gF_val + (w > 0.0 && k > 1 ? w * gD : static_cast<scalar_t>(0));
            }
        }
    }
    return E;
}

struct ObjectiveResult {
    double energy;
    at::Tensor grad;
};

template <typename scalar_t>
inline ObjectiveResult eval_data_objective_impl(
    const at::Tensor& V_in, const at::Tensor& S_in, const at::Tensor& X_in,
    double c_val, double w_val, const std::string& orth, bool eval_grad) {

    auto V = V_in.contiguous();
    int64_t p = V.size(0);
    int64_t k = V.size(1);
    const scalar_t* V_ptr = V.data_ptr<scalar_t>();

    int64_t n_rows = 0;
    const scalar_t* X_ptr = nullptr;
    const scalar_t* S_ptr = nullptr;
    at::Tensor X_contig, S_contig;

    if (S_in.defined() && S_in.numel() > 0) {
        S_contig = S_in.contiguous();
        S_ptr = S_contig.data_ptr<scalar_t>();
    } else if (X_in.defined() && X_in.numel() > 0) {
        X_contig = X_in.contiguous();
        n_rows = X_contig.size(0);
        X_ptr = X_contig.data_ptr<scalar_t>();
    }

    ObjectiveWorkspace<scalar_t> ws;
    ws.init_data(p, k, n_rows);

    at::Tensor grad;
    scalar_t* grad_ptr = nullptr;
    if (eval_grad) {
        grad = at::empty({p, k}, V.options());
        grad_ptr = grad.data_ptr<scalar_t>();
    }

    double E = eval_data_objective_raw(p, k, V_ptr, n_rows, X_ptr, S_ptr, c_val, w_val, orth, eval_grad, grad_ptr, ws);
    return {E, grad};
}

inline ObjectiveResult eval_data_objective(
    const at::Tensor& V, const at::Tensor& S, const at::Tensor& X,
    double c, double w, const std::string& orth, bool eval_grad) {
    return AT_DISPATCH_FLOATING_TYPES(V.scalar_type(), "eval_data_objective", ([&] {
        return eval_data_objective_impl<scalar_t>(V, S, X, c, w, orth, eval_grad);
    }));
}

inline double cubic_min_scalar(double a, double fa, double ga, double b, double fb, double gb) {
    if (a == b) return std::numeric_limits<double>::quiet_NaN();
    double d1 = ga + gb - 3.0 * (fa - fb) / (a - b);
    double q = d1 * d1 - ga * gb;
    if (q < 0.0) return std::numeric_limits<double>::quiet_NaN();
    double d2 = std::sqrt(q) * (b > a ? 1.0 : -1.0);
    double denom = gb - ga + 2.0 * d2;
    if (denom == 0.0) return std::numeric_limits<double>::quiet_NaN();
    double cand = b - (b - a) * ((gb + d2 - d1) / denom);
    double lo = a < b ? a : b;
    double hi = a < b ? b : a;
    if (lo < cand && cand < hi) return cand;
    return std::numeric_limits<double>::quiet_NaN();
}

// --------------------------------------------------------------------------
// Zero-Allocation High-Speed Fused L-BFGS-B Solver
// --------------------------------------------------------------------------

template <typename scalar_t>
std::tuple<at::Tensor, double, int64_t, int64_t, int64_t, std::string, double, double, double>
lbfgsb_solve_impl(const at::Tensor& x0_in,
                  const py::dict& spec,
                  const c10::optional<at::Tensor>& lo_opt,
                  const c10::optional<at::Tensor>& hi_opt,
                  const c10::optional<at::Tensor>& mask_opt,
                  int64_t max_grad,
                  double tol,
                  int64_t memory,
                  double sigma,
                  int64_t patience,
                  double rtol,
                  double stall_slack,
                  const py::object& callback) {

    auto x0 = x0_in.contiguous();
    auto orig_shape = x0.sizes().vec();
    int64_t n = x0.numel();

    int64_t p = orig_shape[0];
    int64_t k_vars = orig_shape[1];

    bool has_lo = lo_opt.has_value() && lo_opt->defined() && lo_opt->numel() > 0;
    bool has_hi = hi_opt.has_value() && hi_opt->defined() && hi_opt->numel() > 0;
    bool has_mask = mask_opt.has_value() && mask_opt->defined() && mask_opt->numel() > 0;

    std::vector<scalar_t> lo_vec(n, -std::numeric_limits<scalar_t>::infinity());
    std::vector<scalar_t> hi_vec(n, std::numeric_limits<scalar_t>::infinity());
    std::vector<scalar_t> mask_vec(n, static_cast<scalar_t>(1));

    if (has_lo) {
        auto lo_t = lo_opt->contiguous().reshape({-1});
        const scalar_t* p_lo = lo_t.data_ptr<scalar_t>();
        std::copy(p_lo, p_lo + n, lo_vec.begin());
    }
    if (has_hi) {
        auto hi_t = hi_opt->contiguous().reshape({-1});
        const scalar_t* p_hi = hi_t.data_ptr<scalar_t>();
        std::copy(p_hi, p_hi + n, hi_vec.begin());
    }
    if (has_mask) {
        auto mask_t = mask_opt->contiguous().reshape({-1});
        const scalar_t* p_mask = mask_t.data_ptr<scalar_t>();
        std::copy(p_mask, p_mask + n, mask_vec.begin());
        if (!has_lo) {
            std::fill(lo_vec.begin(), lo_vec.end(), static_cast<scalar_t>(0));
            has_lo = true;
        }
        for (int64_t i = 0; i < n; ++i) {
            bool mb = (mask_vec[i] != static_cast<scalar_t>(0));
            lo_vec[i] = mb ? lo_vec[i] : static_cast<scalar_t>(0);
            if (!has_hi) {
                hi_vec[i] = mb ? std::numeric_limits<scalar_t>::infinity() : static_cast<scalar_t>(0);
            } else {
                hi_vec[i] = mb ? hi_vec[i] : static_cast<scalar_t>(0);
            }
        }
        has_hi = true;
    }

    std::string mode = spec["mode"].cast<std::string>();
    double w = spec["w"].cast<double>();
    std::string orth = spec.contains("orth") ? spec["orth"].cast<std::string>() : "D";

    at::Tensor S_data, X_data, X0_anchored, chol_anchored;
    const scalar_t* S_ptr = nullptr;
    const scalar_t* X_ptr = nullptr;
    const scalar_t* X0_ptr = nullptr;
    const scalar_t* chol_ptr = nullptr;
    int64_t n_rows = 0;

    double c_data = 0.0, denom_anchored = 0.0, lobe_signed = 0.0;
    std::string fidelity_anchored = "anchor";

    ObjectiveWorkspace<scalar_t> ws;

    if (mode == "data") {
        if (spec.contains("S") && !spec["S"].is_none()) {
            S_data = spec["S"].cast<at::Tensor>().contiguous();
            S_ptr = S_data.data_ptr<scalar_t>();
        }
        if (spec.contains("X") && !spec["X"].is_none()) {
            X_data = spec["X"].cast<at::Tensor>().contiguous();
            n_rows = X_data.size(0);
            X_ptr = X_data.data_ptr<scalar_t>();
        }
        c_data = spec["c"].cast<double>();
        ws.init_data(p, k_vars, n_rows);
    } else if (mode == "signed") {
        if (spec.contains("S") && !spec["S"].is_none()) {
            S_data = spec["S"].cast<at::Tensor>().contiguous();
            S_ptr = S_data.data_ptr<scalar_t>();
        }
        if (spec.contains("X") && !spec["X"].is_none()) {
            X_data = spec["X"].cast<at::Tensor>().contiguous();
            n_rows = X_data.size(0);
            X_ptr = X_data.data_ptr<scalar_t>();
        }
        c_data = spec["c"].cast<double>();
        lobe_signed = spec.contains("lobe") ? spec["lobe"].cast<double>() : 0.0;
        int64_t k = k_vars / 2;
        ws.init_signed(p, k, n_rows);
    } else if (mode == "anchored") {
        fidelity_anchored = spec.contains("fidelity") ? spec["fidelity"].cast<std::string>() : "anchor";
        X0_anchored = spec["target"].cast<at::Tensor>().contiguous();
        X0_ptr = X0_anchored.data_ptr<scalar_t>();
        denom_anchored = spec["denom"].cast<double>();
        if (spec.contains("chol") && !spec["chol"].is_none()) {
            chol_anchored = spec["chol"].cast<at::Tensor>().contiguous();
            chol_ptr = chol_anchored.data_ptr<scalar_t>();
        }
        ws.init_anchored(p, k_vars, fidelity_anchored);
    }

    int64_t n_grad = 0;
    int64_t n_fun = 0;

    auto eval_fg_raw = [&](const scalar_t* x_buf, scalar_t* g_buf) -> double {
        n_grad += 1;
        if (mode == "data") {
            return eval_data_objective_raw<scalar_t>(p, k_vars, x_buf, n_rows, X_ptr, S_ptr, c_data, w, orth, true, g_buf, ws);
        } else if (mode == "signed") {
            return eval_signed_objective_raw<scalar_t>(p, k_vars, x_buf, n_rows, X_ptr, S_ptr, c_data, w, orth, lobe_signed, true, g_buf, ws);
        } else {
            return eval_anchored_objective_raw<scalar_t>(p, k_vars, x_buf, X0_ptr, denom_anchored, fidelity_anchored, chol_ptr, w, orth, true, g_buf, ws);
        }
    };

    auto eval_f_raw = [&](const scalar_t* x_buf) -> double {
        n_fun += 1;
        if (mode == "data") {
            return eval_data_objective_raw<scalar_t>(p, k_vars, x_buf, n_rows, X_ptr, S_ptr, c_data, w, orth, false, nullptr, ws);
        } else if (mode == "signed") {
            return eval_signed_objective_raw<scalar_t>(p, k_vars, x_buf, n_rows, X_ptr, S_ptr, c_data, w, orth, lobe_signed, false, nullptr, ws);
        } else {
            return eval_anchored_objective_raw<scalar_t>(p, k_vars, x_buf, X0_ptr, denom_anchored, fidelity_anchored, chol_ptr, w, orth, false, nullptr, ws);
        }
    };

    // Workspace buffers for L-BFGS-B
    std::vector<scalar_t> xf(n);
    const scalar_t* x0_ptr = x0.data_ptr<scalar_t>();
    clip_raw(n, x0_ptr, xf.data(), has_lo, lo_vec.data(), has_hi, hi_vec.data(), has_mask, mask_vec.data());

    std::vector<scalar_t> g(n);
    double f = eval_fg_raw(xf.data(), g.data());
    double gmap = compute_gmap_raw(n, xf.data(), g.data(), has_lo, lo_vec.data(), has_hi, hi_vec.data(), has_mask, mask_vec.data());
    double energy_start = f;
    double grad_map_start = gmap;
    std::string stop = "max_iter";
    int64_t it = 0;

    bool has_callback = !callback.is_none();
    if (has_callback) {
        callback(0, energy_start, grad_map_start, 1, 0);
    }

    std::vector<double> E_win;
    std::vector<double> g_win;
    double eps = (sizeof(scalar_t) == sizeof(double)) ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();

    auto stalled = [&]() -> bool {
        if (g_win.size() < 5) return false;
        double first = g_win[0];
        double best = *std::min_element(g_win.begin(), g_win.end());
        return std::isfinite(first) && first > 0.0 && best >= 0.9 * first;
    };

    auto classify_stall = [&](double f_try) -> std::string {
        if (!std::isfinite(gmap) || gmap > stall_slack * std::max(tol, 0.0)) {
            return "line_search";
        }
        if (stalled()) {
            return "plateau";
        }
        if (std::isfinite(f_try) && (f_try - f) <= 64.0 * eps * (1.0 + std::abs(f))) {
            return "plateau";
        }
        return "line_search";
    };

    // Preallocated solver workspaces
    std::vector<scalar_t> d(n);
    std::vector<scalar_t> xcp(n);
    std::vector<uint8_t> fixed(n);

    int64_t mem_cap = std::max<int64_t>(memory, 1);
    if (mem_cap > 32) {
        PyErr_WarnEx(PyExc_RuntimeWarning, "L-BFGS-B memory capped at 32", 1);
        mem_cap = 32;
    }
    DirectionWorkspace<scalar_t> ws_dir;
    ws_dir.resize(n, mem_cap);

    std::vector<scalar_t> x_trial(n);
    std::vector<scalar_t> g_trial(n);
    std::vector<scalar_t> x_prev(n);
    std::vector<scalar_t> g_prev(n);
    std::vector<scalar_t> x_lo(n);
    std::vector<scalar_t> g_lo(n);
    std::vector<scalar_t> x_new(n);
    std::vector<scalar_t> g_new(n);
    std::vector<scalar_t> s_temp(n);
    std::vector<scalar_t> y_temp(n);

    std::vector<scalar_t> S_hist(mem_cap * n, 0);
    std::vector<scalar_t> Y_hist(mem_cap * n, 0);
    std::vector<scalar_t> M_buf((2 * mem_cap) * (2 * mem_cap), 0);
    int64_t m_hist = 0;
    double theta = 1.0;
    int64_t t_dir_us = 0, t_phi_us = 0, t_mem_us = 0, t_gmap_us = 0;
    const bool profile_enabled = spec.contains("profile") && spec["profile"].cast<bool>();

    while (n_grad < max_grad) {
        it += 1;
        if (gmap <= tol) {
            stop = "grad_map";
            break;
        }

        std::chrono::time_point<std::chrono::high_resolution_clock> t0_step;
        if (profile_enabled) t0_step = std::chrono::high_resolution_clock::now();
        double a_max = 1.0;

        if (m_hist > 0) {
            std::memcpy(ws_dir.W_buf.data(), Y_hist.data(), m_hist * n * sizeof(scalar_t));
            scalar_t th_s = static_cast<scalar_t>(theta);
            const scalar_t* s_src = S_hist.data();
            scalar_t* w_dst = &ws_dir.W_buf[m_hist * n];
            for (int64_t i = 0; i < m_hist * n; ++i) {
                w_dst[i] = th_s * s_src[i];
            }
        }

        lbfgsb_direction_core<scalar_t>(
            n, xf.data(), g.data(),
            has_lo, lo_vec.data(), has_hi, hi_vec.data(),
            m_hist, theta, (m_hist > 0) ? M_buf.data() : nullptr,
            (m_hist > 0) ? ws_dir.W_buf.data() : nullptr,
            512, ws_dir, d.data(), a_max, fixed.data(), xcp.data());

        double g0_dir = 0.0;
        for (int64_t i = 0; i < n; ++i) g0_dir += static_cast<double>(d[i]) * static_cast<double>(g[i]);

        if (g0_dir >= 0.0) {
            m_hist = 0;
            theta = 1.0;
            lbfgsb_direction_core<scalar_t>(
                n, xf.data(), g.data(),
                has_lo, lo_vec.data(), has_hi, hi_vec.data(),
                0, 1.0, nullptr, nullptr,
                512, ws_dir, d.data(), a_max, fixed.data(), xcp.data());

            clip_raw(n, xcp.data(), xcp.data(), has_lo, lo_vec.data(), has_hi, hi_vec.data(), has_mask, mask_vec.data());
            double dg_reset = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                d[i] = xcp[i] - xf[i];
                dg_reset += static_cast<double>(d[i]) * static_cast<double>(g[i]);
            }
            if (dg_reset >= 0.0) {
                double g_norm2 = 0.0;
                for (int64_t i = 0; i < n; ++i) g_norm2 += static_cast<double>(g[i]) * static_cast<double>(g[i]);
                double g_norm = std::sqrt(g_norm2);
                double tiny_s = std::numeric_limits<scalar_t>::min();
                scalar_t denom_g = static_cast<scalar_t>(std::max(g_norm, tiny_s));
                for (int64_t i = 0; i < n; ++i) d[i] = -g[i] / denom_g;
            }
            a_max = compute_a_max_raw(n, d.data(), xf.data(), has_lo, lo_vec.data(), has_hi, hi_vec.data());
            g0_dir = 0.0;
            for (int64_t i = 0; i < n; ++i) g0_dir += static_cast<double>(d[i]) * static_cast<double>(g[i]);
        }
        if (profile_enabled) {
            auto t1_step = std::chrono::high_resolution_clock::now();
            t_dir_us += std::chrono::duration_cast<std::chrono::microseconds>(t1_step - t0_step).count();
        }

        auto phi = [&](double a, double& fa_out, double& ga_d_out) {
            std::chrono::time_point<std::chrono::high_resolution_clock> t_p0;
            if (profile_enabled) t_p0 = std::chrono::high_resolution_clock::now();
            for (int64_t i = 0; i < n; ++i) {
                scalar_t val = xf[i] + static_cast<scalar_t>(a) * d[i];
                if (has_lo && val < lo_vec[i]) val = lo_vec[i];
                if (has_hi && val > hi_vec[i]) val = hi_vec[i];
                if (has_mask) val *= mask_vec[i];
                x_trial[i] = val;
            }
            fa_out = eval_fg_raw(x_trial.data(), g_trial.data());
            double res_d = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                res_d += static_cast<double>(g_trial[i]) * static_cast<double>(d[i]);
            }
            ga_d_out = res_d;
            if (profile_enabled) {
                auto t_p1 = std::chrono::high_resolution_clock::now();
                t_phi_us += std::chrono::duration_cast<std::chrono::microseconds>(t_p1 - t_p0).count();
            }
        };

        if (std::isnan(a_max)) {
            a_max = 1.0;
            a_max = compute_a_max_raw(n, d.data(), xf.data(), has_lo, lo_vec.data(), has_hi, hi_vec.data());
        }

        int64_t remaining = std::max<int64_t>(max_grad - n_grad - 1, 1);
        int64_t max_eval = std::min<int64_t>(20, remaining);
        int64_t ls_eval = 0;
        double a_prev = 0.0, f_prev = f, d_prev = g0_dir;
        std::copy(xf.begin(), xf.end(), x_prev.begin());
        std::copy(g.begin(), g.end(), g_prev.begin());

        double a_i = std::min(1.0, a_max);
        double a_lo_val = 0.0, a_hi_val = 0.0, f_lo_val = f, f_hi_val = 0.0, d_lo_val = g0_dir, d_hi_val = 0.0;
        std::copy(xf.begin(), xf.end(), x_lo.begin());
        std::copy(g.begin(), g.end(), g_lo.begin());

        bool has_bracket = false, has_f_hi = false;
        bool wolfe_success = false;
        double accepted_f = f;
        double c1 = 1e-4, c2 = 0.9;

        while (ls_eval < max_eval) {
            double f_i = 0.0, d_i = 0.0;
            phi(a_i, f_i, d_i);
            ls_eval += 1;
            if (!std::isfinite(f_i) || !std::isfinite(d_i)) {
                a_lo_val = a_prev; f_lo_val = f_prev; d_lo_val = d_prev; a_hi_val = a_i;
                std::copy(x_prev.begin(), x_prev.end(), x_lo.begin());
                std::copy(g_prev.begin(), g_prev.end(), g_lo.begin());
                has_bracket = true; has_f_hi = false;
                break;
            }
            if (f_i > f + c1 * a_i * g0_dir || (ls_eval > 1 && f_i >= f_prev)) {
                a_lo_val = a_prev; f_lo_val = f_prev; d_lo_val = d_prev; a_hi_val = a_i;
                std::copy(x_prev.begin(), x_prev.end(), x_lo.begin());
                std::copy(g_prev.begin(), g_prev.end(), g_lo.begin());
                has_bracket = true; has_f_hi = false;
                break;
            }
            if (std::abs(d_i) <= -c2 * g0_dir) {
                wolfe_success = true;
                accepted_f = f_i;
                std::copy(x_trial.begin(), x_trial.end(), x_new.begin());
                std::copy(g_trial.begin(), g_trial.end(), g_new.begin());
                break;
            }
            if (d_i >= 0.0) {
                a_lo_val = a_i; f_lo_val = f_i; d_lo_val = d_i; a_hi_val = a_prev;
                std::copy(x_trial.begin(), x_trial.end(), x_lo.begin());
                std::copy(g_trial.begin(), g_trial.end(), g_lo.begin());
                has_bracket = true; has_f_hi = false;
                break;
            }
            a_prev = a_i; f_prev = f_i; d_prev = d_i;
            std::copy(x_trial.begin(), x_trial.end(), x_prev.begin());
            std::copy(g_trial.begin(), g_trial.end(), g_prev.begin());
            if (a_i >= a_max - 1e-16) {
                wolfe_success = true;
                accepted_f = f_i;
                std::copy(x_trial.begin(), x_trial.end(), x_new.begin());
                std::copy(g_trial.begin(), g_trial.end(), g_new.begin());
                break;
            }
            a_i = std::min(2.0 * a_i, a_max);
        }

        if (!wolfe_success && has_bracket) {
            while (ls_eval < max_eval && std::abs(a_hi_val - a_lo_val) > 1e-16) {
                double a_j = std::numeric_limits<double>::quiet_NaN();
                if (has_f_hi) {
                    a_j = cubic_min_scalar(a_lo_val, f_lo_val, d_lo_val, a_hi_val, f_hi_val, d_hi_val);
                }
                if (std::isnan(a_j)) {
                    a_j = 0.5 * (a_lo_val + a_hi_val);
                }
                double f_j = 0.0, d_j = 0.0;
                phi(a_j, f_j, d_j);
                ls_eval += 1;
                if (!std::isfinite(f_j) || !std::isfinite(d_j)) {
                    a_hi_val = a_j; has_f_hi = false;
                    continue;
                }
                if (f_j > f + c1 * a_j * g0_dir || f_j >= f_lo_val) {
                    a_hi_val = a_j; f_hi_val = f_j; d_hi_val = d_j; has_f_hi = true;
                } else {
                    if (std::abs(d_j) <= -c2 * g0_dir) {
                        wolfe_success = true;
                        accepted_f = f_j;
                        std::copy(x_trial.begin(), x_trial.end(), x_new.begin());
                        std::copy(g_trial.begin(), g_trial.end(), g_new.begin());
                        break;
                    }
                    if (d_j * (a_hi_val - a_lo_val) >= 0.0) {
                        a_hi_val = a_lo_val; f_hi_val = f_lo_val; d_hi_val = d_lo_val; has_f_hi = true;
                    }
                    a_lo_val = a_j; f_lo_val = f_j; d_lo_val = d_j;
                    std::copy(x_trial.begin(), x_trial.end(), x_lo.begin());
                }
            }
        }

        double f_try = std::numeric_limits<double>::infinity();
        if (!wolfe_success) {
            double step = 1.0;
            bool accepted = false;
            for (int64_t ls = 0; ls < 30; ++ls) {
                double dn2 = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    scalar_t val = xf[i] + static_cast<scalar_t>(step) * d[i];
                    if (has_lo && val < lo_vec[i]) val = lo_vec[i];
                    if (has_hi && val > hi_vec[i]) val = hi_vec[i];
                    if (has_mask) val *= mask_vec[i];
                    x_trial[i] = val;
                    double diff = static_cast<double>(val - xf[i]);
                    dn2 += diff * diff;
                }
                if (dn2 == 0.0 || (sizeof(scalar_t) == sizeof(float) && dn2 <= 1e-16)) break;
                double fb = eval_f_raw(x_trial.data());
                if (!std::isfinite(fb)) {
                    step *= 0.5;
                    continue;
                }
                f_try = std::min(f_try, fb);
                if (fb <= f - sigma * dn2 / step) {
                    accepted = true;
                    std::copy(x_trial.begin(), x_trial.end(), x_new.begin());
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) {
                stop = classify_stall(f_try);
                break;
            }
            if (n_grad >= max_grad) {
                stop = "max_iter";
                break;
            }
            accepted_f = eval_fg_raw(x_new.data(), g_new.data());
        }

        bool g_finite = true;
        for (int64_t i = 0; i < n; ++i) {
            if (!std::isfinite(g_new[i])) {
                g_finite = false;
                break;
            }
        }
        if (!std::isfinite(accepted_f) || !g_finite) {
            stop = "line_search";
            break;
        }

        // Push correction pair
        std::chrono::time_point<std::chrono::high_resolution_clock> t_m0;
        if (profile_enabled) t_m0 = std::chrono::high_resolution_clock::now();
        double sy = 0.0, yy = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            scalar_t si = x_new[i] - xf[i];
            scalar_t yi = g_new[i] - g[i];
            s_temp[i] = si;
            y_temp[i] = yi;
            sy += static_cast<double>(si) * static_cast<double>(yi);
            yy += static_cast<double>(yi) * static_cast<double>(yi);
        }
        if (yy > 0.0 && sy > 2.2e-16 * yy) {
            if (m_hist < mem_cap) {
                std::memcpy(&S_hist[m_hist * n], s_temp.data(), n * sizeof(scalar_t));
                std::memcpy(&Y_hist[m_hist * n], y_temp.data(), n * sizeof(scalar_t));
                m_hist += 1;
            } else {
                std::memmove(&S_hist[0], &S_hist[n], (mem_cap - 1) * n * sizeof(scalar_t));
                std::memmove(&Y_hist[0], &Y_hist[n], (mem_cap - 1) * n * sizeof(scalar_t));
                std::memcpy(&S_hist[(mem_cap - 1) * n], s_temp.data(), n * sizeof(scalar_t));
                std::memcpy(&Y_hist[(mem_cap - 1) * n], y_temp.data(), n * sizeof(scalar_t));
            }
            theta = yy / sy;
            bool ok = lbfgsb_build_M_contiguous<scalar_t>(n, m_hist, S_hist.data(), Y_hist.data(), theta, M_buf.data());
            if (!ok) {
                m_hist = 0;
                theta = 1.0;
            }
        }
        if (profile_enabled) {
            auto t_m1 = std::chrono::high_resolution_clock::now();
            t_mem_us += std::chrono::duration_cast<std::chrono::microseconds>(t_m1 - t_m0).count();
        }

        std::copy(x_new.begin(), x_new.end(), xf.begin());
        std::copy(g_new.begin(), g_new.end(), g.begin());
        f = accepted_f;
        std::chrono::time_point<std::chrono::high_resolution_clock> t_gm0;
        if (profile_enabled) t_gm0 = std::chrono::high_resolution_clock::now();
        gmap = compute_gmap_raw(n, xf.data(), g.data(), has_lo, lo_vec.data(), has_hi, hi_vec.data(), has_mask, mask_vec.data());
        if (profile_enabled) {
            auto t_gm1 = std::chrono::high_resolution_clock::now();
            t_gmap_us += std::chrono::duration_cast<std::chrono::microseconds>(t_gm1 - t_gm0).count();
        }

        if (has_callback) {
            callback(it, f, gmap, n_grad, n_fun);
        }

        if (gmap <= tol) {
            stop = "grad_map";
            break;
        }

        E_win.push_back(f);
        g_win.push_back(gmap);
        if (static_cast<int64_t>(E_win.size()) > patience) {
            E_win.erase(E_win.begin());
            g_win.erase(g_win.begin());
        }
        if (static_cast<int64_t>(E_win.size()) == patience) {
            bool all_nan = true;
            for (double val : E_win) {
                if (std::isfinite(val)) {
                    all_nan = false;
                    break;
                }
            }
            if (!all_nan) {
                double min_E = *std::min_element(E_win.begin(), E_win.end());
                double max_E = *std::max_element(E_win.begin(), E_win.end());
                if ((max_E - min_E) / (1.0 + std::abs(min_E)) < rtol && stalled()) {
                    stop = "plateau";
                    break;
                }
            }
        }
    }

    if (stop == "max_iter" && std::isfinite(gmap) && gmap <= stall_slack * std::max(tol, 0.0) && stalled()) {
        stop = "plateau";
    }

    if (spec.contains("profile") && spec["profile"].cast<bool>()) {
        printf("PROFILE iters=%lld: dir=%lld us (%.1f/it), phi=%lld us (%.1f/it), mem=%lld us (%.1f/it), gmap=%lld us (%.1f/it)\n",
               (long long)it,
               (long long)t_dir_us, (double)t_dir_us / std::max<int64_t>(it, 1),
               (long long)t_phi_us, (double)t_phi_us / std::max<int64_t>(it, 1),
               (long long)t_mem_us, (double)t_mem_us / std::max<int64_t>(it, 1),
               (long long)t_gmap_us, (double)t_gmap_us / std::max<int64_t>(it, 1));
    }

    at::Tensor x_out = at::empty_like(x0);
    scalar_t* x_out_ptr = x_out.data_ptr<scalar_t>();
    std::copy(xf.begin(), xf.end(), x_out_ptr);

    return std::make_tuple(x_out, f, n_grad, n_fun, it, stop, gmap, energy_start, grad_map_start);
}

std::tuple<at::Tensor, double, int64_t, int64_t, int64_t, std::string, double, double, double>
lbfgsb_solve(const at::Tensor& x0_in,
             const py::dict& spec,
             const c10::optional<at::Tensor>& lo_opt,
             const c10::optional<at::Tensor>& hi_opt,
             const c10::optional<at::Tensor>& mask_opt,
             int64_t max_grad,
             double tol,
             int64_t memory,
             double sigma,
             int64_t patience,
             double rtol,
             double stall_slack,
             const py::object& callback) {
    return AT_DISPATCH_FLOATING_TYPES(x0_in.scalar_type(), "lbfgsb_solve", ([&] {
        return lbfgsb_solve_impl<scalar_t>(x0_in, spec, lo_opt, hi_opt, mask_opt,
                                           max_grad, tol, memory, sigma,
                                           patience, rtol, stall_slack, callback);
    }));
}

} // namespace nsa_flow

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lbfgsb_build_M", &nsa_flow::lbfgsb_build_M, "Rebuild M compact L-BFGS matrix");
    m.def("lbfgsb_direction", &nsa_flow::lbfgsb_direction, "L-BFGS-B search direction and a_max",
          py::arg("x"), py::arg("g"), py::arg("lo") = py::none(), py::arg("hi") = py::none(),
          py::arg("S") = at::Tensor(), py::arg("Y") = at::Tensor(), py::arg("theta") = 1.0,
          py::arg("M") = at::Tensor(), py::arg("max_breakpoints") = 512);
    m.def("eval_data_objective", [](const at::Tensor& V, const at::Tensor& S, const at::Tensor& X,
                                    double c, double w, const std::string& orth, bool eval_grad) {
        auto res = nsa_flow::eval_data_objective(V, S, X, c, w, orth, eval_grad);
        return py::make_tuple(res.energy, res.grad);
    }, py::arg("V"), py::arg("S"), py::arg("X"), py::arg("c"), py::arg("w"), py::arg("orth"), py::arg("eval_grad") = true);
    m.def("lbfgsb_solve", &nsa_flow::lbfgsb_solve, "Whole L-BFGS-B iteration with fused objective",
          py::arg("x0"), py::arg("problem_spec"),
          py::arg("lo") = py::none(), py::arg("hi") = py::none(), py::arg("mask") = py::none(),
          py::arg("max_grad") = 2000, py::arg("tol") = 1e-9, py::arg("memory") = 10,
          py::arg("sigma") = 1e-4, py::arg("patience") = 50, py::arg("rtol") = 1e-12,
          py::arg("stall_slack") = 1e3, py::arg("callback") = py::none());
}
