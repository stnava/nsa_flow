#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <tuple>
#include <string>

namespace nsa_flow {

template <typename T>
inline T safe_div(T num, T den, T tiny) {
    if (den >= static_cast<T>(0)) {
        return num / std::max(den, tiny);
    }
    return num / std::min(den, -tiny);
}

template <typename T>
inline bool solve_dense_system(int64_t dim, const T* A, const T* b, T* x) {
    if (dim > 64) return false;
    T M[64][65];
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            M[i][j] = A[i * dim + j];
        }
        M[i][dim] = b[i];
    }
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
        if (max_val < 1e-300) return false;
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
    if (dim > 64) return false;
    T M[64][128];
    for (int64_t i = 0; i < dim; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            M[i][j] = A[i * dim + j];
            M[i][dim + j] = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
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
        if (max_val < 1e-300) return false;
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
at::Tensor lbfgsb_build_M_impl(const at::Tensor& S_in, const at::Tensor& Y_in, double theta) {
    if (!S_in.defined() || S_in.numel() == 0 || S_in.size(1) == 0) {
        return at::empty({0, 0}, S_in.options());
    }
    int64_t n = S_in.size(0);
    int64_t m = S_in.size(1);
    int64_t two_m = 2 * m;

    if (two_m > 64) {
        auto SY = S_in.t().matmul(Y_in);
        auto SS = S_in.t().matmul(S_in);
        auto L = SY.tril(-1);
        auto D = at::diag_embed(SY.diagonal());
        auto top = at::cat({-D, L.t()}, 1);
        auto bot = at::cat({L, theta * SS}, 1);
        auto K = at::cat({top, bot}, 0);
        auto eye = at::eye(two_m, S_in.options());
        return at::linalg_solve(K, eye);
    }

    auto S = S_in.contiguous();
    auto Y = Y_in.contiguous();
    const scalar_t* S_ptr = S.data_ptr<scalar_t>();
    const scalar_t* Y_ptr = Y.data_ptr<scalar_t>();

    scalar_t SY[32][32] = {{0}};
    scalar_t SS[32][32] = {{0}};
    for (int64_t r = 0; r < n; ++r) {
        const scalar_t* s_row = &S_ptr[r * m];
        const scalar_t* y_row = &Y_ptr[r * m];
        for (int64_t i = 0; i < m; ++i) {
            scalar_t si = s_row[i];
            for (int64_t j = 0; j < m; ++j) {
                SY[i][j] += si * y_row[j];
            }
            for (int64_t j = i; j < m; ++j) {
                SS[i][j] += si * s_row[j];
            }
        }
    }
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            SS[i][j] = SS[j][i];
        }
    }

    scalar_t K[64 * 64] = {0};
    scalar_t th = static_cast<scalar_t>(theta);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            K[i * two_m + j] = (i == j) ? -SY[i][i] : static_cast<scalar_t>(0);
            K[i * two_m + (m + j)] = (j > i) ? SY[j][i] : static_cast<scalar_t>(0);
            K[(m + i) * two_m + j] = (i > j) ? SY[i][j] : static_cast<scalar_t>(0);
            K[(m + i) * two_m + (m + j)] = th * SS[i][j];
        }
    }

    auto M = at::empty({two_m, two_m}, S_in.options());
    scalar_t* M_out = M.data_ptr<scalar_t>();
    bool inverted = invert_dense_system<scalar_t>(two_m, K, M_out);
    if (!inverted) {
        auto SY_t = S_in.t().matmul(Y_in);
        auto SS_t = S_in.t().matmul(S_in);
        auto L = SY_t.tril(-1);
        auto D = at::diag_embed(SY_t.diagonal());
        auto top = at::cat({-D, L.t()}, 1);
        auto bot = at::cat({L, theta * SS_t}, 1);
        auto K_t = at::cat({top, bot}, 0);
        auto eye = at::eye(two_m, S_in.options());
        return at::linalg_solve(K_t, eye);
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

    const scalar_t tiny = std::numeric_limits<scalar_t>::min();
    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();

    int64_t m = (S_in.defined() && S_in.numel() > 0) ? S_in.size(1) : 0;
    int64_t two_m = 2 * m;

    auto S = (m > 0) ? S_in.contiguous() : at::Tensor();
    auto Y = (m > 0) ? Y_in.contiguous() : at::Tensor();
    const scalar_t* S_ptr = (m > 0) ? S.data_ptr<scalar_t>() : nullptr;
    const scalar_t* Y_ptr = (m > 0) ? Y.data_ptr<scalar_t>() : nullptr;

    auto get_W = [&](int64_t row, int64_t col) -> scalar_t {
        if (col < m) return Y_ptr[row * m + col];
        return static_cast<scalar_t>(theta) * S_ptr[row * m + (col - m)];
    };

    at::Tensor M_contig = (m > 0 && M_in.defined() && M_in.numel() > 0) ? M_in.contiguous() : at::Tensor();
    const scalar_t* M_ptr = (M_contig.defined() && M_contig.numel() > 0) ? M_contig.data_ptr<scalar_t>() : nullptr;

    // 1. Breakpoint computation
    scalar_t t_buf[1024];
    std::vector<scalar_t> t_vec;
    scalar_t* t = (n <= 1024) ? t_buf : (t_vec.assign(n, inf), t_vec.data());
    if (n <= 1024) std::fill(t, t + n, inf);

    at::Tensor fixed = at::zeros({n}, x.options().dtype(at::kBool));
    bool* fixed_ptr = fixed.data_ptr<bool>();

    at::Tensor x_cp = at::empty_like(x);
    scalar_t* x_cp_ptr = x_cp.data_ptr<scalar_t>();

    at::Tensor d = at::zeros_like(x);
    scalar_t* d_ptr = d.data_ptr<scalar_t>();

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
            fixed_ptr[i] = true;
            scalar_t bound = has_lo ? lo_ptr[i] : (has_hi ? hi_ptr[i] : 0.0);
            x_cp_ptr[i] = bound;
            d_ptr[i] = 0.0;
        } else {
            fixed_ptr[i] = false;
            x_cp_ptr[i] = x_ptr[i];
            d_ptr[i] = -g_ptr[i];
        }
    }

    scalar_t p[64] = {0};
    scalar_t c[64] = {0};

    if (two_m > 0) {
        for (int64_t k = 0; k < two_m; ++k) {
            scalar_t sum_val = 0;
            for (int64_t i = 0; i < n; ++i) {
                sum_val += get_W(i, k) * d_ptr[i];
            }
            p[k] = sum_val;
        }
    }

    scalar_t fp = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        fp -= d_ptr[i] * d_ptr[i];
    }
    scalar_t fpp = -static_cast<scalar_t>(theta) * fp;

    if (two_m > 0 && M_ptr) {
        scalar_t pMp = 0;
        for (int64_t r = 0; r < two_m; ++r) {
            scalar_t Mp_r = 0;
            for (int64_t k = 0; k < two_m; ++k) {
                Mp_r += M_ptr[r * two_m + k] * p[k];
            }
            pMp += Mp_r * p[r];
        }
        fpp -= pMp;
    }
    fpp = std::max(fpp, tiny);
    scalar_t dt_min = safe_div(-fp, fpp, tiny);
    scalar_t t_old = 0.0;

    struct Breakpoint {
        scalar_t t;
        int64_t idx;
    };
    Breakpoint cand_buf[1024];
    std::vector<Breakpoint> cand_vec;
    Breakpoint* cand = (n <= 1024) ? cand_buf : (cand_vec.resize(n), cand_vec.data());
    int64_t cand_size = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (std::isfinite(t[i]) && t[i] > 0) {
            cand[cand_size++] = {t[i], i};
        }
    }

    if (cand_size > 0) {
        int64_t num_bk = std::min<int64_t>(cand_size, max_breakpoints);
        std::partial_sort(cand, cand + num_bk, cand + cand_size,
                          [](const Breakpoint& a, const Breakpoint& b) {
                              if (a.t != b.t) return a.t < b.t;
                              return a.idx < b.idx;
                          });

        for (int64_t j = 0; j < num_bk; ++j) {
            scalar_t tj = cand[j].t;
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

            if (two_m > 0 && M_ptr) {
                scalar_t MW_i[64] = {0};
                for (int64_t r = 0; r < two_m; ++r) {
                    scalar_t sum_val = 0;
                    for (int64_t k = 0; k < two_m; ++k) {
                        sum_val += M_ptr[r * two_m + k] * get_W(i, k);
                    }
                    MW_i[r] = sum_val;
                }
                scalar_t wMp = 0;
                scalar_t wMw = 0;
                for (int64_t k = 0; k < two_m; ++k) {
                    wMc += MW_i[k] * c[k];
                    wMp += MW_i[k] * p[k];
                    wMw += MW_i[k] * get_W(i, k);
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

            if (two_m > 0) {
                for (int64_t k = 0; k < two_m; ++k) {
                    p[k] += gi * get_W(i, k);
                }
            }

            d_ptr[i] = 0.0;
            x_cp_ptr[i] = xcp_b;
            fixed_ptr[i] = true;
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

    // 2. Subspace Minimisation
    bool any_free = false;
    for (int64_t i = 0; i < n; ++i) {
        if (!fixed_ptr[i]) {
            any_free = true;
            break;
        }
    }

    at::Tensor dir = at::empty_like(x);
    scalar_t* dir_ptr = dir.data_ptr<scalar_t>();

    if (any_free) {
        scalar_t r_buf[1024];
        std::vector<scalar_t> r_vec;
        scalar_t* r = (n <= 1024) ? r_buf : (r_vec.resize(n), r_vec.data());

        // Mc = M * c (size 2m)
        scalar_t Mc[64] = {0};
        if (m > 0 && M_ptr) {
            for (int64_t r_idx = 0; r_idx < two_m; ++r_idx) {
                scalar_t sum_val = 0;
                for (int64_t k = 0; k < two_m; ++k) {
                    sum_val += M_ptr[r_idx * two_m + k] * c[k];
                }
                Mc[r_idx] = sum_val;
            }
        }

        // r = (g + theta * (x_cp - x) - W * Mc) * free
        scalar_t th = static_cast<scalar_t>(theta);
        for (int64_t i = 0; i < n; ++i) {
            if (fixed_ptr[i]) {
                r[i] = 0.0;
            } else {
                scalar_t zi = x_cp_ptr[i] - x_ptr[i];
                scalar_t ri = g_ptr[i] + th * zi;
                if (m > 0 && M_ptr) {
                    scalar_t wmc = 0;
                    for (int64_t k = 0; k < two_m; ++k) {
                        wmc += get_W(i, k) * Mc[k];
                    }
                    ri -= wmc;
                }
                r[i] = ri;
            }
        }

        scalar_t d_hat_buf[1024];
        std::vector<scalar_t> d_hat_vec;
        scalar_t* d_hat = (n <= 1024) ? d_hat_buf : (d_hat_vec.resize(n), d_hat_vec.data());

        if (m == 0 || !M_ptr) {
            for (int64_t i = 0; i < n; ++i) {
                d_hat[i] = fixed_ptr[i] ? 0.0 : (-r[i] / th);
            }
        } else if (two_m <= 64) {
            // Fused Wf^T * r and Wf^T * Wf in a single sequential pass
            scalar_t Wfr[64] = {0};
            scalar_t WfWf[64 * 64] = {0};
            for (int64_t i = 0; i < n; ++i) {
                if (!fixed_ptr[i]) {
                    scalar_t w_i[64];
                    const scalar_t* y_row = &Y_ptr[i * m];
                    const scalar_t* s_row = &S_ptr[i * m];
                    for (int64_t k = 0; k < m; ++k) {
                        w_i[k] = y_row[k];
                        w_i[m + k] = th * s_row[k];
                    }
                    scalar_t ri = r[i];
                    for (int64_t k = 0; k < two_m; ++k) {
                        Wfr[k] += w_i[k] * ri;
                    }
                    for (int64_t j = 0; j < two_m; ++j) {
                        scalar_t wj = w_i[j];
                        scalar_t* W_row = &WfWf[j * two_m];
                        for (int64_t l = j; l < two_m; ++l) {
                            W_row[l] += wj * w_i[l];
                        }
                    }
                }
            }
            for (int64_t j = 0; j < two_m; ++j) {
                for (int64_t l = 0; l < j; ++l) {
                    WfWf[j * two_m + l] = WfWf[l * two_m + j];
                }
            }

            // v0 = M * (Wf^T * r) (size 2m)
            scalar_t v0[64] = {0};
            for (int64_t i = 0; i < two_m; ++i) {
                scalar_t sum_val = 0;
                for (int64_t k = 0; k < two_m; ++k) {
                    sum_val += M_ptr[i * two_m + k] * Wfr[k];
                }
                v0[i] = sum_val;
            }

            // MWf = M * (Wf^T * Wf) (size 2m x 2m)
            // N = I - (1 / theta) * MWf
            scalar_t N[64 * 64] = {0};
            scalar_t inv_th = 1.0 / th;
            for (int64_t i = 0; i < two_m; ++i) {
                for (int64_t j = 0; j < two_m; ++j) {
                    scalar_t mw_val = 0;
                    for (int64_t k = 0; k < two_m; ++k) {
                        mw_val += M_ptr[i * two_m + k] * WfWf[k * two_m + j];
                    }
                    N[i * two_m + j] = (i == j ? 1.0 : 0.0) - inv_th * mw_val;
                }
            }

            // Solve N * v = v0 using LAPACK for bit-exact parity even on ill-conditioned systems
            scalar_t v[64] = {0};
            auto N_t = at::from_blob(N, {two_m, two_m}, x.options());
            auto v0_t = at::from_blob(v0, {two_m, 1}, x.options());
            auto v_t = at::linalg_solve(N_t, v0_t);
            const scalar_t* vt_p = v_t.template data_ptr<scalar_t>();
            std::copy(vt_p, vt_p + two_m, v);

            // d_hat = -(r / theta) - (Wf * v) / (theta^2)
            scalar_t inv_th2 = 1.0 / (th * th);
            for (int64_t i = 0; i < n; ++i) {
                if (fixed_ptr[i]) {
                    d_hat[i] = 0.0;
                } else {
                    scalar_t wfv = 0;
                    for (int64_t k = 0; k < two_m; ++k) {
                        wfv += get_W(i, k) * v[k];
                    }
                    d_hat[i] = -r[i] * inv_th - wfv * inv_th2;
                }
            }
        } else {
            // Fallback if two_m > 64
            at::Tensor free_f = fixed.logical_not().to(x.dtype());
            at::Tensor W_t = at::empty({n, two_m}, x.options());
            scalar_t* W_t_ptr = W_t.template data_ptr<scalar_t>();
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t k = 0; k < two_m; ++k) {
                    W_t_ptr[i * two_m + k] = get_W(i, k);
                }
            }
            at::Tensor r_t = at::from_blob(r, {n}, x.options()).clone();
            at::Tensor Wf = W_t * free_f.unsqueeze(1);
            at::Tensor v = M_in.matmul(Wf.t().matmul(r_t));
            at::Tensor N_t = at::eye(two_m, x.options()) - (M_in.matmul(Wf.t().matmul(Wf))) / static_cast<scalar_t>(theta);
            v = at::linalg_solve(N_t, v);
            at::Tensor d_hat_t = -(r_t / static_cast<scalar_t>(theta)) - (Wf.matmul(v)) / static_cast<scalar_t>(theta * theta);
            d_hat_t = d_hat_t * free_f;
            const scalar_t* dht_p = d_hat_t.template data_ptr<scalar_t>();
            std::copy(dht_p, dht_p + n, d_hat);
        }

        scalar_t alpha = 1.0;
        if (has_lo) {
            for (int64_t i = 0; i < n; ++i) {
                if (!fixed_ptr[i] && d_hat[i] < 0) {
                    scalar_t lim = (lo_ptr[i] - x_cp_ptr[i]) / std::min(d_hat[i], -tiny);
                    if (lim < 0) lim = 0;
                    if (lim < alpha) alpha = lim;
                }
            }
        }
        if (has_hi) {
            for (int64_t i = 0; i < n; ++i) {
                if (!fixed_ptr[i] && d_hat[i] > 0) {
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

    // 3. Feasibility limit a_max along dir
    double a_max = 1.0;
    if (has_lo) {
        scalar_t min_lim = inf;
        bool any_neg = false;
        for (int64_t i = 0; i < n; ++i) {
            if (dir_ptr[i] < 0 && x_ptr[i] > lo_ptr[i]) {
                any_neg = true;
                scalar_t lim = (lo_ptr[i] - x_ptr[i]) / std::min(dir_ptr[i], -tiny);
                if (lim < 0) lim = 0;
                if (lim < min_lim) min_lim = lim;
            }
        }
        if (any_neg) {
            a_max = std::max(a_max, static_cast<double>(min_lim));
        }
    }
    if (has_hi) {
        scalar_t min_lim = inf;
        bool any_pos = false;
        for (int64_t i = 0; i < n; ++i) {
            if (dir_ptr[i] > 0 && x_ptr[i] < hi_ptr[i]) {
                any_pos = true;
                scalar_t lim = (hi_ptr[i] - x_ptr[i]) / std::max(dir_ptr[i], tiny);
                if (lim < 0) lim = 0;
                if (lim < min_lim) min_lim = lim;
            }
        }
        if (any_pos) {
            a_max = std::max(a_max, static_cast<double>(min_lim));
        }
    }

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
// Phase 2: Fused objective evaluations and whole L-BFGS-B iteration
// --------------------------------------------------------------------------

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
    scalar_t c = static_cast<scalar_t>(c_val);
    scalar_t w = static_cast<scalar_t>(w_val);
    bool is_orth_cg = (orth == "Cg");

    scalar_t SV_stack[1024] = {0};
    scalar_t A_stack[256] = {0};
    scalar_t B_stack[256] = {0};
    scalar_t G_stack[256] = {0};
    std::vector<scalar_t> SV_heap, A_heap, B_heap, G_heap;
    scalar_t* SV = (p * k <= 1024) ? SV_stack : (SV_heap.assign(p * k, 0), SV_heap.data());
    scalar_t* A = (k * k <= 256) ? A_stack : (A_heap.assign(k * k, 0), A_heap.data());
    scalar_t* B = (k * k <= 256) ? B_stack : (B_heap.assign(k * k, 0), B_heap.data());
    scalar_t* G = (k * k <= 256) ? G_stack : (G_heap.assign(k * k, 0), G_heap.data());

    if (S_in.defined() && S_in.numel() > 0) {
        auto S = S_in.contiguous();
        if (p > 128) {
            at::Tensor SV_t = at::mm(S, V);
            const scalar_t* sv_p = SV_t.data_ptr<scalar_t>();
            std::copy(sv_p, sv_p + p * k, SV);
            at::Tensor A_t = at::mm(V.t(), SV_t);
            const scalar_t* a_p = A_t.data_ptr<scalar_t>();
            std::copy(a_p, a_p + k * k, A);
        } else {
            const scalar_t* S_ptr = S.data_ptr<scalar_t>();
            for (int64_t i = 0; i < p; ++i) {
                const scalar_t* S_row = &S_ptr[i * p];
                scalar_t* SV_row = &SV[i * k];
                for (int64_t r = 0; r < p; ++r) {
                    scalar_t s_ir = S_row[r];
                    const scalar_t* V_row = &V_ptr[r * k];
                    for (int64_t j = 0; j < k; ++j) {
                        SV_row[j] += s_ir * V_row[j];
                    }
                }
            }
            for (int64_t r = 0; r < p; ++r) {
                const scalar_t* V_row = &V_ptr[r * k];
                const scalar_t* SV_row = &SV[r * k];
                for (int64_t i = 0; i < k; ++i) {
                    scalar_t v_ri = V_row[i];
                    scalar_t* A_row = &A[i * k];
                    for (int64_t j = 0; j < k; ++j) {
                        A_row[j] += v_ri * SV_row[j];
                    }
                }
            }
        }
    } else if (X_in.defined() && X_in.numel() > 0) {
        auto X = X_in.contiguous();
        int64_t n_rows = X.size(0);
        if (p > 128) {
            at::Tensor XV_t = at::mm(X, V);
            at::Tensor A_t = at::mm(XV_t.t(), XV_t);
            const scalar_t* a_p = A_t.data_ptr<scalar_t>();
            std::copy(a_p, a_p + k * k, A);
            if (eval_grad) {
                at::Tensor SV_t = at::mm(X.t(), XV_t);
                const scalar_t* sv_p = SV_t.data_ptr<scalar_t>();
                std::copy(sv_p, sv_p + p * k, SV);
            }
        } else {
            const scalar_t* X_ptr = X.data_ptr<scalar_t>();
            std::vector<scalar_t> XV(n_rows * k, 0);
            for (int64_t i = 0; i < n_rows; ++i) {
                const scalar_t* X_row = &X_ptr[i * p];
                scalar_t* XV_row = &XV[i * k];
                for (int64_t r = 0; r < p; ++r) {
                    scalar_t x_ir = X_row[r];
                    const scalar_t* V_row = &V_ptr[r * k];
                    for (int64_t j = 0; j < k; ++j) {
                        XV_row[j] += x_ir * V_row[j];
                    }
                }
            }
            for (int64_t r = 0; r < n_rows; ++r) {
                const scalar_t* XV_row = &XV[r * k];
                for (int64_t i = 0; i < k; ++i) {
                    scalar_t xv_ri = XV_row[i];
                    scalar_t* A_row = &A[i * k];
                    for (int64_t j = 0; j < k; ++j) {
                        A_row[j] += xv_ri * XV_row[j];
                    }
                }
            }
            if (eval_grad) {
                for (int64_t r = 0; r < n_rows; ++r) {
                    const scalar_t* X_row = &X_ptr[r * p];
                    const scalar_t* XV_row = &XV[r * k];
                    for (int64_t i = 0; i < p; ++i) {
                        scalar_t x_ri = X_row[i];
                        scalar_t* SV_row = &SV[i * k];
                        for (int64_t j = 0; j < k; ++j) {
                            SV_row[j] += x_ri * XV_row[j];
                        }
                    }
                }
            }
        }
    }

    if (p > 128) {
        at::Tensor B_t = at::mm(V.t(), V);
        const scalar_t* b_p = B_t.data_ptr<scalar_t>();
        std::copy(b_p, b_p + k * k, B);
    } else {
        for (int64_t r = 0; r < p; ++r) {
            const scalar_t* V_row = &V_ptr[r * k];
            for (int64_t i = 0; i < k; ++i) {
                scalar_t v_ri = V_row[i];
                scalar_t* B_row = &B[i * k];
                for (int64_t j = 0; j < k; ++j) {
                    B_row[j] += v_ri * V_row[j];
                }
            }
        }
    }

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
    scalar_t scale_k_cg = (k > 1) ? (1.0 - 1.0 / static_cast<scalar_t>(k)) : 1.0;
    scalar_t scale_k_D = (k > 1) ? (1.0 - 1.0 / static_cast<scalar_t>(k)) : 1.0;
    scalar_t N2 = 0;

    if (k > 1) {
        if (is_orth_cg) {
            if (t < 1e-12) t = 1e-12;
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
            if (t < 1e-300) t = 1e-300;
            scalar_t inv_k = 1.0 / static_cast<scalar_t>(k);
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    scalar_t g_val = B[i * k + j] / t;
                    G[i * k + j] = g_val;
                    N2 += g_val * g_val;
                    scalar_t gc = g_val - (i == j ? inv_k : 0.0);
                    D += gc * gc;
                }
            }
            D /= scale_k_D;
        }
    }

    double E = static_cast<double>((1.0 - w) * F + (k > 1 ? w * D : 0.0));

    at::Tensor grad;
    if (eval_grad) {
        grad = at::empty({p, k}, V.options());
        scalar_t* g_ptr = grad.data_ptr<scalar_t>();
        scalar_t factor_F = 2.0 / c;
        scalar_t factor_D_D = (k > 1) ? (4.0 / (scale_k_D * t)) : 0.0;
        scalar_t factor_D_cg = (k > 1) ? (4.0 / (t * t * scale_k_cg)) : 0.0;
        scalar_t scale_cg = (t > 0) ? (a2 / t) : 0.0;

        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                scalar_t svb = 0, va = 0;
                for (int64_t m = 0; m < k; ++m) {
                    svb += SV[r * k + m] * B[m * k + j];
                    va += V_ptr[r * k + m] * A[m * k + j];
                }
                scalar_t gF = factor_F * (-2.0 * SV[r * k + j] + svb + va);

                scalar_t gD = 0;
                if (k > 1 && w > 0.0) {
                    if (is_orth_cg) {
                        scalar_t v_off = 0;
                        for (int64_t m = 0; m < k; ++m) {
                            if (m != j) v_off += V_ptr[r * k + m] * B[m * k + j];
                        }
                        gD = factor_D_cg * (v_off - scale_cg * V_ptr[r * k + j]);
                    } else {
                        scalar_t vg = 0;
                        for (int64_t m = 0; m < k; ++m) {
                            vg += V_ptr[r * k + m] * G[m * k + j];
                        }
                        gD = factor_D_D * (vg - N2 * V_ptr[r * k + j]);
                    }
                }
                g_ptr[r * k + j] = (1.0 - w) * gF + (w > 0.0 && k > 1 ? w * gD : 0.0);
            }
        }
    }
    return {E, grad};
}

inline ObjectiveResult eval_data_objective(
    const at::Tensor& V, const at::Tensor& S, const at::Tensor& X,
    double c, double w, const std::string& orth, bool eval_grad) {
    return AT_DISPATCH_FLOATING_TYPES(V.scalar_type(), "eval_data_objective", ([&] {
        return eval_data_objective_impl<scalar_t>(V, S, X, c, w, orth, eval_grad);
    }));
}

template <typename scalar_t>
inline ObjectiveResult eval_signed_objective_impl(
    const at::Tensor& W_in, const at::Tensor& S, const at::Tensor& X,
    double c, double w, const std::string& orth, double lobe, bool eval_grad) {

    auto W = W_in.contiguous();
    int64_t p = W.size(0);
    int64_t two_k = W.size(1);
    int64_t k = two_k / 2;
    const scalar_t* W_ptr = W.data_ptr<scalar_t>();

    at::Tensor V = at::empty({p, k}, W.options());
    scalar_t* V_ptr = V.data_ptr<scalar_t>();
    for (int64_t r = 0; r < p; ++r) {
        for (int64_t j = 0; j < k; ++j) {
            V_ptr[r * k + j] = W_ptr[r * two_k + j] - W_ptr[r * two_k + k + j];
        }
    }

    auto data_res = eval_data_objective(V, S, X, c, 0.0, "", eval_grad);
    double F = data_res.energy;

    std::vector<scalar_t> BW(two_k * two_k, 0);
    for (int64_t i = 0; i < two_k; ++i) {
        for (int64_t j = 0; j < two_k; ++j) {
            scalar_t sum = 0;
            for (int64_t r = 0; r < p; ++r) {
                sum += W_ptr[r * two_k + i] * W_ptr[r * two_k + j];
            }
            BW[i * two_k + j] = sum;
        }
    }

    scalar_t D = 0;
    scalar_t tW = 0;
    for (int64_t i = 0; i < two_k; ++i) tW += BW[i * two_k + i];
    scalar_t a2 = 0;
    scalar_t scale_k_cg = 1.0 - 1.0 / static_cast<scalar_t>(two_k);
    scalar_t scale_k_D = 1.0 - 1.0 / static_cast<scalar_t>(two_k);
    std::vector<scalar_t> GW(two_k * two_k, 0);
    scalar_t N2 = 0;

    if (orth == "D") {
        if (tW < 1e-300) tW = 1e-300;
        scalar_t inv_k = 1.0 / static_cast<scalar_t>(two_k);
        for (int64_t i = 0; i < two_k; ++i) {
            for (int64_t j = 0; j < two_k; ++j) {
                scalar_t g_val = BW[i * two_k + j] / tW;
                GW[i * two_k + j] = g_val;
                N2 += g_val * g_val;
                scalar_t gc = g_val - (i == j ? inv_k : 0.0);
                D += gc * gc;
            }
        }
        D /= scale_k_D;
    } else { // "Cg" default for signed
        if (tW < 1e-12) tW = 1e-12;
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
    if (lobe > 0.0 && w > 0.0) {
        scalar_t dot = 0;
        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                dot += W_ptr[r * two_k + j] * W_ptr[r * two_k + k + j];
            }
        }
        E_lobe = static_cast<double>((w * lobe / c) * dot);
    }

    double E = (1.0 - w) * F + w * static_cast<double>(D) + E_lobe;

    at::Tensor grad;
    if (eval_grad) {
        grad = at::empty({p, two_k}, W.options());
        scalar_t* g_ptr = grad.data_ptr<scalar_t>();
        const scalar_t* gV_ptr = data_res.grad.data_ptr<scalar_t>();
        scalar_t factor_D_D = 4.0 / (scale_k_D * tW);
        scalar_t factor_D_cg = 4.0 / (tW * tW * scale_k_cg);
        scalar_t scale_cg = (tW > 0) ? (a2 / tW) : 0.0;
        scalar_t lobe_coef = static_cast<scalar_t>(w * lobe / c);

        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < two_k; ++j) {
                scalar_t gD = 0;
                if (w > 0.0) {
                    if (orth == "D") {
                        scalar_t wg = 0;
                        for (int64_t m = 0; m < two_k; ++m) {
                            wg += W_ptr[r * two_k + m] * GW[m * two_k + j];
                        }
                        gD = factor_D_D * (wg - N2 * W_ptr[r * two_k + j]);
                    } else {
                        scalar_t w_off = 0;
                        for (int64_t m = 0; m < two_k; ++m) {
                            if (m != j) w_off += W_ptr[r * two_k + m] * BW[m * two_k + j];
                        }
                        gD = factor_D_cg * (w_off - scale_cg * W_ptr[r * two_k + j]);
                    }
                }
                scalar_t gW_F = (j < k) ? gV_ptr[r * k + j] : -gV_ptr[r * k + (j - k)];
                scalar_t g_lobe = 0;
                if (lobe > 0.0 && w > 0.0) {
                    g_lobe = lobe_coef * ((j < k) ? W_ptr[r * two_k + k + j] : W_ptr[r * two_k + (j - k)]);
                }
                g_ptr[r * two_k + j] = static_cast<scalar_t>(1.0 - w) * gW_F + static_cast<scalar_t>(w) * gD + g_lobe;
            }
        }
    }
    return {E, grad};
}

inline ObjectiveResult eval_signed_objective(
    const at::Tensor& W, const at::Tensor& S, const at::Tensor& X,
    double c, double w, const std::string& orth, double lobe, bool eval_grad) {
    return AT_DISPATCH_FLOATING_TYPES(W.scalar_type(), "eval_signed_objective", ([&] {
        return eval_signed_objective_impl<scalar_t>(W, S, X, c, w, orth, lobe, eval_grad);
    }));
}

template <typename scalar_t>
inline ObjectiveResult eval_anchored_objective_impl(
    const at::Tensor& Y_in, const at::Tensor& X0_in, double denom_val,
    const std::string& fidelity, const at::Tensor& chol_in,
    double w_val, const std::string& orth, bool eval_grad) {

    auto Y = Y_in.contiguous();
    auto X0 = X0_in.contiguous();
    int64_t p = Y.size(0);
    int64_t k = Y.size(1);
    const scalar_t* Y_ptr = Y.data_ptr<scalar_t>();
    const scalar_t* X0_ptr = X0.data_ptr<scalar_t>();
    scalar_t denom = static_cast<scalar_t>(denom_val);
    scalar_t w = static_cast<scalar_t>(w_val);
    bool is_orth_cg = (orth == "Cg");

    scalar_t F = 0.0;
    at::Tensor grad_F;
    if (eval_grad) grad_F = at::empty({p, k}, Y.options());
    scalar_t* gF_ptr = eval_grad ? grad_F.data_ptr<scalar_t>() : nullptr;

    if (fidelity == "subspace") {
        auto chol = chol_in.contiguous();
        const scalar_t* chol_ptr = chol.data_ptr<scalar_t>();
        std::vector<scalar_t> rhs(k * k, 0);
        for (int64_t i = 0; i < k; ++i) {
            for (int64_t j = 0; j < k; ++j) {
                scalar_t sum = 0;
                for (int64_t r = 0; r < p; ++r) {
                    sum += X0_ptr[r * k + i] * Y_ptr[r * k + j];
                }
                rhs[i * k + j] = sum;
            }
        }
        std::vector<scalar_t> coef(k * k, 0);
        for (int64_t col = 0; col < k; ++col) {
            for (int64_t i = 0; i < k; ++i) {
                scalar_t s = rhs[i * k + col];
                for (int64_t j = 0; j < i; ++j) {
                    s -= chol_ptr[i * k + j] * coef[j * k + col];
                }
                coef[i * k + col] = s / chol_ptr[i * k + i];
            }
            for (int64_t i = k - 1; i >= 0; --i) {
                scalar_t s = coef[i * k + col];
                for (int64_t j = i + 1; j < k; ++j) {
                    s -= chol_ptr[j * k + i] * coef[j * k + col];
                }
                coef[i * k + col] = s / chol_ptr[i * k + i];
            }
        }
        scalar_t R2 = 0, nY2 = 0;
        std::vector<scalar_t> R(p * k, 0);
        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                scalar_t py = 0;
                for (int64_t m = 0; m < k; ++m) {
                    py += X0_ptr[r * k + m] * coef[m * k + j];
                }
                scalar_t r_val = Y_ptr[r * k + j] - py;
                R[r * k + j] = r_val;
                R2 += r_val * r_val;
                nY2 += Y_ptr[r * k + j] * Y_ptr[r * k + j];
            }
        }
        if (nY2 < 1e-30) nY2 = 1e-30;
        F = R2 / nY2;
        if (eval_grad) {
            scalar_t factor_sub = 2.0 / nY2;
            for (int64_t idx = 0; idx < p * k; ++idx) {
                gF_ptr[idx] = factor_sub * (R[idx] - F * Y_ptr[idx]);
            }
        }
    } else { // "anchor"
        scalar_t R2 = 0;
        for (int64_t idx = 0; idx < p * k; ++idx) {
            scalar_t diff = Y_ptr[idx] - X0_ptr[idx];
            R2 += diff * diff;
            if (eval_grad) gF_ptr[idx] = (2.0 / denom) * diff;
        }
        F = R2 / denom;
    }

    std::vector<scalar_t> B(k * k, 0);
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            scalar_t sum = 0;
            for (int64_t r = 0; r < p; ++r) {
                sum += Y_ptr[r * k + i] * Y_ptr[r * k + j];
            }
            B[i * k + j] = sum;
        }
    }

    scalar_t D = 0;
    scalar_t t = 0;
    for (int64_t i = 0; i < k; ++i) t += B[i * k + i];
    scalar_t a2 = 0;
    scalar_t scale_k_cg = (k > 1) ? (1.0 - 1.0 / static_cast<scalar_t>(k)) : 1.0;
    scalar_t scale_k_D = (k > 1) ? (1.0 - 1.0 / static_cast<scalar_t>(k)) : 1.0;
    std::vector<scalar_t> G(k * k, 0);
    scalar_t N2 = 0;

    if (k > 1) {
        if (is_orth_cg) {
            if (t < 1e-12) t = 1e-12;
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
            if (t < 1e-300) t = 1e-300;
            scalar_t inv_k = 1.0 / static_cast<scalar_t>(k);
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    scalar_t g_val = B[i * k + j] / t;
                    G[i * k + j] = g_val;
                    N2 += g_val * g_val;
                    scalar_t gc = g_val - (i == j ? inv_k : 0.0);
                    D += gc * gc;
                }
            }
            D /= scale_k_D;
        }
    }

    double E = static_cast<double>((1.0 - w) * F + (k > 1 ? w * D : 0.0));

    at::Tensor grad;
    if (eval_grad) {
        grad = at::empty({p, k}, Y.options());
        scalar_t* g_ptr = grad.data_ptr<scalar_t>();
        scalar_t factor_D_D = (k > 1) ? (4.0 / (scale_k_D * t)) : 0.0;
        scalar_t factor_D_cg = (k > 1) ? (4.0 / (t * t * scale_k_cg)) : 0.0;
        scalar_t scale_cg = (t > 0) ? (a2 / t) : 0.0;

        for (int64_t r = 0; r < p; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                scalar_t gF = gF_ptr[r * k + j];
                scalar_t gD = 0;
                if (k > 1 && w > 0.0) {
                    if (is_orth_cg) {
                        scalar_t v_off = 0;
                        for (int64_t m = 0; m < k; ++m) {
                            if (m != j) v_off += Y_ptr[r * k + m] * B[m * k + j];
                        }
                        gD = factor_D_cg * (v_off - scale_cg * Y_ptr[r * k + j]);
                    } else {
                        scalar_t vg = 0;
                        for (int64_t m = 0; m < k; ++m) {
                            vg += Y_ptr[r * k + m] * G[m * k + j];
                        }
                        gD = factor_D_D * (vg - N2 * Y_ptr[r * k + j]);
                    }
                }
                g_ptr[r * k + j] = (1.0 - w) * gF + (w > 0.0 && k > 1 ? w * gD : 0.0);
            }
        }
    }
    return {E, grad};
}

inline ObjectiveResult eval_anchored_objective(
    const at::Tensor& Y, const at::Tensor& X0, double denom,
    const std::string& fidelity, const at::Tensor& chol,
    double w, const std::string& orth, bool eval_grad) {
    return AT_DISPATCH_FLOATING_TYPES(Y.scalar_type(), "eval_anchored_objective", ([&] {
        return eval_anchored_objective_impl<scalar_t>(Y, X0, denom, fidelity, chol, w, orth, eval_grad);
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

    auto x0 = x0_in.contiguous();
    auto orig_shape = x0.sizes().vec();
    int64_t n = x0.numel();

    bool has_lo = lo_opt.has_value() && lo_opt->defined() && lo_opt->numel() > 0;
    bool has_hi = hi_opt.has_value() && hi_opt->defined() && hi_opt->numel() > 0;
    bool has_mask = mask_opt.has_value() && mask_opt->defined() && mask_opt->numel() > 0;

    at::Tensor lo = has_lo ? lo_opt->contiguous().reshape({-1}) : at::Tensor();
    at::Tensor hi = has_hi ? hi_opt->contiguous().reshape({-1}) : at::Tensor();
    at::Tensor mask = has_mask ? mask_opt->contiguous().reshape({-1}) : at::Tensor();

    if (has_mask) {
        if (!has_lo) {
            lo = at::zeros({n}, x0.options());
            has_lo = true;
        }
        auto mb = mask.to(at::kBool);
        lo = at::where(mb, lo, at::zeros_like(lo));
        if (!has_hi) {
            hi = at::full_like(lo, std::numeric_limits<double>::infinity());
            has_hi = true;
        }
        hi = at::where(mb, hi, at::zeros_like(hi));
    }

    auto clip = [&](const at::Tensor& v) {
        auto res = v;
        if (has_lo) res = at::maximum(res, lo);
        if (has_hi) res = at::minimum(res, hi);
        return res;
    };

    std::string mode = spec["mode"].cast<std::string>();
    double w = spec["w"].cast<double>();
    std::string orth = spec.contains("orth") ? spec["orth"].cast<std::string>() : "D";

    at::Tensor S_data, X_data, X0_anchored, chol_anchored;
    double c_data = 0.0, denom_anchored = 0.0, lobe_signed = 0.0;
    std::string fidelity_anchored = "anchor";

    if (mode == "data") {
        if (spec.contains("S") && !spec["S"].is_none()) S_data = spec["S"].cast<at::Tensor>().contiguous();
        if (spec.contains("X") && !spec["X"].is_none()) X_data = spec["X"].cast<at::Tensor>().contiguous();
        c_data = spec["c"].cast<double>();
    } else if (mode == "signed") {
        if (spec.contains("S") && !spec["S"].is_none()) S_data = spec["S"].cast<at::Tensor>().contiguous();
        if (spec.contains("X") && !spec["X"].is_none()) X_data = spec["X"].cast<at::Tensor>().contiguous();
        c_data = spec["c"].cast<double>();
        lobe_signed = spec.contains("lobe") ? spec["lobe"].cast<double>() : 0.0;
    } else if (mode == "anchored") {
        fidelity_anchored = spec.contains("fidelity") ? spec["fidelity"].cast<std::string>() : "anchor";
        X0_anchored = spec["target"].cast<at::Tensor>().contiguous();
        denom_anchored = spec["denom"].cast<double>();
        if (spec.contains("chol") && !spec["chol"].is_none()) {
            chol_anchored = spec["chol"].cast<at::Tensor>().contiguous();
        }
    }

    int64_t n_grad = 0;
    int64_t n_fun = 0;

    auto eval_fg = [&](const at::Tensor& x_flat) -> std::pair<double, at::Tensor> {
        n_grad += 1;
        auto x_shaped = x_flat.reshape(orig_shape);
        ObjectiveResult res;
        if (mode == "data") {
            res = eval_data_objective(x_shaped, S_data, X_data, c_data, w, orth, true);
        } else if (mode == "signed") {
            res = eval_signed_objective(x_shaped, S_data, X_data, c_data, w, orth, lobe_signed, true);
        } else {
            res = eval_anchored_objective(x_shaped, X0_anchored, denom_anchored, fidelity_anchored, chol_anchored, w, orth, true);
        }
        return {res.energy, res.grad.reshape({-1})};
    };

    auto eval_f = [&](const at::Tensor& x_flat) -> double {
        n_fun += 1;
        auto x_shaped = x_flat.reshape(orig_shape);
        ObjectiveResult res;
        if (mode == "data") {
            res = eval_data_objective(x_shaped, S_data, X_data, c_data, w, orth, false);
        } else if (mode == "signed") {
            res = eval_signed_objective(x_shaped, S_data, X_data, c_data, w, orth, lobe_signed, false);
        } else {
            res = eval_anchored_objective(x_shaped, X0_anchored, denom_anchored, fidelity_anchored, chol_anchored, w, orth, false);
        }
        return res.energy;
    };

    auto compute_gmap = [&](const at::Tensor& x_curr, const at::Tensor& g_curr) -> double {
        double nY = x_curr.norm().item<double>();
        if (!std::isfinite(nY) || nY <= 0.0) return std::numeric_limits<double>::infinity();
        auto step = x_curr - (nY * nY) * g_curr;
        step = clip(step);
        if (has_mask) {
            step = step * mask;
        }
        return (step - x_curr).norm().item<double>() / nY;
    };

    at::Tensor xf = clip(x0.reshape({-1}));
    if (has_mask) xf = xf * mask;

    auto [f, g] = eval_fg(xf);
    double gmap = compute_gmap(xf, g);
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
    double eps = (xf.scalar_type() == at::kDouble) ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();
    double eps_floor = (xf.scalar_type() == at::kDouble) ? 1e-9 : 1e-5;

    auto stalled = [&]() -> bool {
        if (g_win.size() < 5) return false;
        double first = g_win[0];
        double best = *std::min_element(g_win.begin(), g_win.end());
        return std::isfinite(first) && first > 0.0 && best >= 0.9 * first;
    };

    auto classify_stall = [&](double f_try) -> std::string {
        if (!std::isfinite(gmap) || gmap > stall_slack * std::max(tol, eps_floor)) {
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

    at::Tensor S_stack = at::empty({n, 0}, xf.options());
    at::Tensor Y_stack = at::empty({n, 0}, xf.options());
    double theta = 1.0;
    at::Tensor M = at::empty({0, 0}, xf.options());

    auto compute_a_max = [&](const at::Tensor& dir_t) -> double {
        double a_m = 1.0;
        if (has_lo) {
            auto d_contig = dir_t.contiguous();
            auto x_contig = xf.contiguous();
            auto lo_contig = lo.contiguous();
            AT_DISPATCH_FLOATING_TYPES(dir_t.scalar_type(), "compute_a_max_lo", ([&] {
                const scalar_t* d_p = d_contig.data_ptr<scalar_t>();
                const scalar_t* x_p = x_contig.data_ptr<scalar_t>();
                const scalar_t* lo_p = lo_contig.data_ptr<scalar_t>();
                scalar_t tiny_s = std::numeric_limits<scalar_t>::min();
                scalar_t min_lim = std::numeric_limits<scalar_t>::infinity();
                bool any_neg = false;
                for (int64_t i = 0; i < n; ++i) {
                    if (d_p[i] < 0 && x_p[i] > lo_p[i]) {
                        any_neg = true;
                        scalar_t lim = (lo_p[i] - x_p[i]) / std::min(d_p[i], -tiny_s);
                        if (lim < 0) lim = 0;
                        if (lim < min_lim) min_lim = lim;
                    }
                }
                if (any_neg) a_m = std::max(a_m, static_cast<double>(min_lim));
            }));
        }
        if (has_hi) {
            auto d_contig = dir_t.contiguous();
            auto x_contig = xf.contiguous();
            auto hi_contig = hi.contiguous();
            AT_DISPATCH_FLOATING_TYPES(dir_t.scalar_type(), "compute_a_max_hi", ([&] {
                const scalar_t* d_p = d_contig.data_ptr<scalar_t>();
                const scalar_t* x_p = x_contig.data_ptr<scalar_t>();
                const scalar_t* hi_p = hi_contig.data_ptr<scalar_t>();
                scalar_t tiny_s = std::numeric_limits<scalar_t>::min();
                scalar_t min_lim = std::numeric_limits<scalar_t>::infinity();
                bool any_pos = false;
                for (int64_t i = 0; i < n; ++i) {
                    if (d_p[i] > 0 && x_p[i] < hi_p[i]) {
                        any_pos = true;
                        scalar_t lim = (hi_p[i] - x_p[i]) / std::max(d_p[i], tiny_s);
                        if (lim < 0) lim = 0;
                        if (lim < min_lim) min_lim = lim;
                    }
                }
                if (any_pos) a_m = std::max(a_m, static_cast<double>(min_lim));
            }));
        }
        return a_m;
    };
    int64_t t_dir_us = 0, t_phi_us = 0, t_mem_us = 0, t_gmap_us = 0;

    while (n_grad < max_grad) {
        it += 1;
        if (gmap <= tol) {
            stop = "grad_map";
            break;
        }

        auto t0_step = std::chrono::high_resolution_clock::now();
        auto [d, a_max, fixed, xcp] = lbfgsb_direction_internal(xf, g, lo, hi, S_stack, Y_stack, theta, M, 512);

        double g0_dir = (d * g).sum().item<double>();
        if (g0_dir >= 0.0) {
            S_stack = at::empty({n, 0}, xf.options());
            Y_stack = at::empty({n, 0}, xf.options());
            theta = 1.0;
            M = at::empty({0, 0}, xf.options());
            auto [d_reset, a_max_reset, fixed_reset, xcp_reset] = lbfgsb_direction_internal(xf, g, lo, hi, S_stack, Y_stack, theta, M, 512);
            d = clip(xcp_reset) - xf;
            double dg_reset = (d * g).sum().item<double>();
            double tiny_val = (xf.scalar_type() == at::kDouble) ? std::numeric_limits<double>::min() : std::numeric_limits<float>::min();
            if (dg_reset >= 0.0) {
                double g_norm = g.norm().item<double>();
                d = -g / std::max(g_norm, tiny_val);
            }
            a_max = compute_a_max(d);
            g0_dir = (d * g).sum().item<double>();
        }
        auto t1_step = std::chrono::high_resolution_clock::now();
        t_dir_us += std::chrono::duration_cast<std::chrono::microseconds>(t1_step - t0_step).count();

        at::Tensor state_x, state_g;
        auto phi = [&](double a) -> std::pair<double, double> {
            auto t_p0 = std::chrono::high_resolution_clock::now();
            auto xa = clip(xf + a * d);
            auto [fa, ga] = eval_fg(xa);
            state_x = xa;
            state_g = ga;
            double res_d = (ga * d).sum().item<double>();
            auto t_p1 = std::chrono::high_resolution_clock::now();
            t_phi_us += std::chrono::duration_cast<std::chrono::microseconds>(t_p1 - t_p0).count();
            return {fa, res_d};
        };

        if (std::isnan(a_max)) {
            a_max = 1.0;
            a_max = compute_a_max(d);
        }

        int64_t remaining = std::max<int64_t>(max_grad - n_grad - 1, 1);
        int64_t max_eval = std::min<int64_t>(20, remaining);
        int64_t ls_eval = 0;
        double a_prev = 0.0, f_prev = f, d_prev = g0_dir;
        at::Tensor x_prev = xf.clone(), g_prev = g.clone();
        double a_i = std::min(1.0, a_max);
        double a_lo = 0.0, a_hi = 0.0, f_lo = f, f_hi = 0.0, d_lo = g0_dir, d_hi = 0.0;
        at::Tensor x_lo = xf.clone(), g_lo = g.clone();
        bool has_bracket = false, has_f_hi = false, has_a_lo = false;
        bool wolfe_success = false;
        double accepted_alpha = 0.0, accepted_f = f;
        at::Tensor x_new, g_new;
        double c1 = 1e-4, c2 = 0.9;
        while (ls_eval < max_eval) {
            auto [f_i, d_i] = phi(a_i);
            ls_eval += 1;
            if (!std::isfinite(f_i) || !std::isfinite(d_i)) {
                a_lo = a_prev; f_lo = f_prev; d_lo = d_prev; a_hi = a_i;
                x_lo = x_prev; g_lo = g_prev;
                has_bracket = true; has_f_hi = false; has_a_lo = true;
                break;
            }
            if (f_i > f + c1 * a_i * g0_dir || (ls_eval > 1 && f_i >= f_prev)) {
                a_lo = a_prev; f_lo = f_prev; d_lo = d_prev; a_hi = a_i;
                x_lo = x_prev; g_lo = g_prev;
                has_bracket = true; has_f_hi = false; has_a_lo = true;
                break;
            }
            if (std::abs(d_i) <= -c2 * g0_dir) {
                wolfe_success = true;
                accepted_alpha = a_i; accepted_f = f_i;
                x_new = state_x; g_new = state_g;
                break;
            }
            if (d_i >= 0.0) {
                a_lo = a_i; f_lo = f_i; d_lo = d_i; a_hi = a_prev;
                x_lo = state_x.clone(); g_lo = state_g.clone();
                has_bracket = true; has_f_hi = false; has_a_lo = true;
                break;
            }
            a_prev = a_i; f_prev = f_i; d_prev = d_i;
            x_prev = state_x.clone(); g_prev = state_g.clone();
            if (a_i >= a_max - 1e-16) {
                wolfe_success = true;
                accepted_alpha = a_i; accepted_f = f_i;
                x_new = state_x; g_new = state_g;
                break;
            }
            a_i = std::min(2.0 * a_i, a_max);
        }

        if (!wolfe_success && has_bracket) {
            while (ls_eval < max_eval && std::abs(a_hi - a_lo) > 1e-16) {
                double a_j = std::numeric_limits<double>::quiet_NaN();
                if (has_f_hi) {
                    a_j = cubic_min_scalar(a_lo, f_lo, d_lo, a_hi, f_hi, d_hi);
                }
                if (std::isnan(a_j)) {
                    a_j = 0.5 * (a_lo + a_hi);
                }
                auto [f_j, d_j] = phi(a_j);
                ls_eval += 1;
                if (!std::isfinite(f_j) || !std::isfinite(d_j)) {
                    a_hi = a_j; has_f_hi = false;
                    continue;
                }
                if (f_j > f + c1 * a_j * g0_dir || f_j >= f_lo) {
                    a_hi = a_j; f_hi = f_j; d_hi = d_j; has_f_hi = true;
                } else {
                    if (std::abs(d_j) <= -c2 * g0_dir) {
                        wolfe_success = true;
                        accepted_alpha = a_j; accepted_f = f_j;
                        x_new = state_x; g_new = state_g;
                        break;
                    }
                    if (d_j * (a_hi - a_lo) >= 0.0) {
                        a_hi = a_lo; f_hi = f_lo; d_hi = d_lo; has_f_hi = true;
                    }
                    a_lo = a_j; f_lo = f_j; d_lo = d_j;
                    x_lo = state_x.clone(); g_lo = state_g.clone();
                }
            }
            if (!wolfe_success && has_a_lo && f_lo < f) {
                wolfe_success = true;
                accepted_alpha = a_lo; accepted_f = f_lo;
                x_new = x_lo; g_new = g_lo;
            }
        }

        double f_try = std::numeric_limits<double>::infinity();
        if (!wolfe_success) {
            // Projected-arc backtracking fallback
            double step = 1.0;
            bool accepted = false;
            for (int64_t ls = 0; ls < 30; ++ls) {
                auto x_trial = clip(xf + step * d);
                auto diff = x_trial - xf;
                double dn2 = (diff * diff).sum().item<double>();
                if (dn2 == 0.0) break;
                double fb = eval_f(x_trial);
                if (!std::isfinite(fb)) {
                    step *= 0.5;
                    continue;
                }
                f_try = std::min(f_try, fb);
                if (fb <= f - sigma * dn2 / step) {
                    accepted = true;
                    x_new = x_trial;
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
            auto [fb_eval, gb_eval] = eval_fg(x_new);
            accepted_f = fb_eval;
            g_new = gb_eval;
        }

        if (!std::isfinite(accepted_f) || !at::isfinite(g_new).all().item<bool>()) {
            stop = "line_search";
            break;
        }

        // Push correction pair
        at::Tensor s = x_new - xf;
        at::Tensor y = g_new - g;
        double sy = (s * y).sum().item<double>();
        double yy = (y * y).sum().item<double>();
        if (yy > 0.0 && sy > 2.2e-16 * yy) {
            auto t_m0 = std::chrono::high_resolution_clock::now();
            if (S_stack.size(1) < memory) {
                S_stack = at::cat({S_stack, s.unsqueeze(1)}, 1);
                Y_stack = at::cat({Y_stack, y.unsqueeze(1)}, 1);
            } else {
                S_stack = at::cat({S_stack.narrow(1, 1, memory - 1), s.unsqueeze(1)}, 1);
                Y_stack = at::cat({Y_stack.narrow(1, 1, memory - 1), y.unsqueeze(1)}, 1);
            }
            theta = yy / sy;
            try {
                M = lbfgsb_build_M(S_stack, Y_stack, theta);
            } catch (...) {
                S_stack = at::empty({n, 0}, xf.options());
                Y_stack = at::empty({n, 0}, xf.options());
                theta = 1.0;
                M = at::empty({0, 0}, xf.options());
            }
            auto t_m1 = std::chrono::high_resolution_clock::now();
            t_mem_us += std::chrono::duration_cast<std::chrono::microseconds>(t_m1 - t_m0).count();
        }

        xf = x_new;
        f = accepted_f;
        g = g_new;
        auto t_gm0 = std::chrono::high_resolution_clock::now();
        gmap = compute_gmap(xf, g);
        auto t_gm1 = std::chrono::high_resolution_clock::now();
        t_gmap_us += std::chrono::duration_cast<std::chrono::microseconds>(t_gm1 - t_gm0).count();

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

    if (stop == "max_iter" && std::isfinite(gmap) && gmap <= stall_slack * std::max(tol, eps_floor) && stalled()) {
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

    return std::make_tuple(xf.reshape(orig_shape), f, n_grad, n_fun, it, stop, gmap, energy_start, grad_map_start);
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
