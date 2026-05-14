/*
 * FANNY fuzzy clustering — port of R package cluster src/fanny.c (fuzzy + caddy).
 *
 * Reference: Kaufman & Rousseeuw (1990); cluster::fanny (Maechler et al.).
 * Only the dissimilarity-matrix path (diss=TRUE) is implemented here.
 */

#include "fanny.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

inline int imin2(int a, int b) { return a < b ? a : b; }
inline int imax2(int a, int b) { return a > b ? a : b; }

inline int pair_index(int m, int i, int n) {
    const int lo = imin2(m, i);
    const int hi = imax2(m, i);
    return lo * n - (lo + 1) * (lo + 2) / 2 + hi;
}

inline double pair_dist(const double* diss, int n, int m, int i) {
    if (m == i) {
        return 0.0;
    }
    return diss[m * n + i];
}

// Condensed layout used inside fuzzy() — same indexing as R cluster.
inline double pair_dist_condensed(const double* dss, int n, int m, int i) {
    if (m == i) {
        return 0.0;
    }
    return dss[pair_index(m, i, n)];
}

void square_to_condensed(const double* diss, int n, std::vector<double>& dss) {
    const int nhalf = n * (n - 1) / 2;
    dss.resize(static_cast<size_t>(nhalf));
    int pos = 0;
    for (int lo = 0; lo < n - 1; ++lo) {
        for (int hi = lo + 1; hi < n; ++hi) {
            dss[static_cast<size_t>(pos++)] = diss[lo * n + hi];
        }
    }
}

void fuzzy_core(
    int nn,
    int k,
    double* p,          // column-major nn x k (p[m + j*nn])
    double* dp,
    double* pt,
    const double* dss,  // condensed
    double* esp,
    double* ef,
    double r,
    double tol,
    int maxit,
    bool compute_p,
    int& out_iterations,
    bool& out_converged,
    double& out_objective,
    double& out_pc,
    double& out_npc
) {
    const int p_d = nn;
    const int dp_d = nn;
    const double reen = 1.0 / (r - 1.0);

    if (compute_p) {
        const double p0 = 0.1 / static_cast<double>(k - 1);
        for (int m = 0; m < nn; ++m) {
            for (int j = 0; j < k; ++j) {
                p[m + j * p_d] = p0;
            }
        }
        int ndk = nn / k;
        int nd = ndk;
        int j = 0;
        for (int m = 0; m < nn; ++m) {
            p[m + j * p_d] = 0.9;
            if (m + 1 >= nd) {
                ++j;
                if (j + 1 == k) {
                    nd = nn;
                } else {
                    nd += ndk;
                }
            }
            for (int jj = 0; jj < k; ++jj) {
                p[m + jj * p_d] = std::pow(p[m + jj * p_d], r);
            }
        }
    } else {
        for (int m = 0; m < nn; ++m) {
            for (int j = 0; j < k; ++j) {
                p[m + j * p_d] = std::pow(p[m + j * p_d], r);
            }
        }
    }

    double cryt = 0.0;
    for (int j = 0; j < k; ++j) {
        esp[j] = 0.0;
        ef[j] = 0.0;
        for (int m = 0; m < nn; ++m) {
            esp[j] += p[m + j * p_d];
            for (int i = 0; i < nn; ++i) {
                if (i != m) {
                    const int mi = pair_index(m, i, nn);
                    dp[m + j * dp_d] += p[i + j * p_d] * dss[mi];
                    ef[j] += p[i + j * p_d] * p[m + j * p_d] * dss[mi];
                }
            }
        }
        cryt += ef[j] / (esp[j] * 2.0);
    }
    double crt = cryt;

    int it = 0;
    bool converged = false;
    while (++it <= maxit) {
        for (int m = 0; m < nn; ++m) {
            double dt = 0.0;
            for (int j = 0; j < k; ++j) {
                const double denom = dp[m + j * dp_d] - ef[j] / (2.0 * esp[j]);
                pt[j] = std::pow(esp[j] / denom, reen);
                dt += pt[j];
            }
            double xx = 0.0;
            for (int j = 0; j < k; ++j) {
                pt[j] /= dt;
                if (pt[j] < 0.0) {
                    xx += pt[j];
                }
            }
            for (int j = 0; j < k; ++j) {
                pt[j] = (pt[j] > 0.0) ? std::pow(pt[j] / (1.0 - xx), r) : 0.0;
                const double d_mj = pt[j] - p[m + j * p_d];
                esp[j] += d_mj;
                for (int i = 0; i < nn; ++i) {
                    if (i != m) {
                        const int mi = pair_index(m, i, nn);
                        const double ddd = d_mj * dss[mi];
                        dp[i + j * dp_d] += ddd;
                        ef[j] += p[i + j * p_d] * 2.0 * ddd;
                    }
                }
                p[m + j * p_d] = pt[j];
            }
        }

        cryt = 0.0;
        for (int j = 0; j < k; ++j) {
            cryt += ef[j] / (esp[j] * 2.0);
        }

        if (std::fabs(cryt - crt) <= tol * std::fabs(cryt)) {
            converged = true;
            break;
        }
        crt = cryt;
    }

    out_iterations = converged ? it : -1;
    out_converged = converged;
    out_objective = cryt;

    double crt_pc = 0.0;
    for (int j = 0; j < k; ++j) {
        crt_pc += esp[j];
    }
    crt_pc /= static_cast<double>(nn);
    out_pc = crt_pc;
    const double xx = std::pow(static_cast<double>(k), r - 1.0);
    out_npc = (xx * crt_pc - 1.0) / (xx - 1.0);

    const double inv_r = 1.0 / r;
    for (int m = 0; m < nn; ++m) {
        for (int j = 0; j < k; ++j) {
            p[m + j * p_d] = std::pow(p[m + j * p_d], inv_r);
        }
    }
}

void caddy_core(
    int nn,
    int k,
    double* p,       // column-major in/out
    int& ktrue,
    int* ncluv_1based
) {
    std::vector<int> nfuzz(static_cast<size_t>(k), 0);

    double pbest = p[0];
    int nbest = 1;
    for (int i = 1; i < k; ++i) {
        if (pbest < p[i * nn]) {
            pbest = p[i * nn];
            nbest = i + 1;
        }
    }
    nfuzz[0] = nbest;
    ncluv_1based[0] = 1;
    ktrue = 1;

    for (int m = 1; m < nn; ++m) {
        pbest = p[m];
        nbest = 1;
        for (int i = 1; i < k; ++i) {
            if (pbest < p[m + i * nn]) {
                pbest = p[m + i * nn];
                nbest = i + 1;
            }
        }
        bool stay = false;
        for (int ktry = 0; ktry < ktrue; ++ktry) {
            if (nfuzz[static_cast<size_t>(ktry)] == nbest) {
                stay = true;
                ncluv_1based[m] = ktry + 1;
                break;
            }
        }
        if (!stay) {
            nfuzz[static_cast<size_t>(ktrue)] = nbest;
            ++ktrue;
            ncluv_1based[m] = ktrue;
        }
    }

    if (ktrue < k) {
        for (int kwalk = ktrue; kwalk < k; ++kwalk) {
            for (int kleft = 1; kleft <= k; ++kleft) {
                bool stay = false;
                for (int ktry = 0; ktry < kwalk; ++ktry) {
                    if (nfuzz[static_cast<size_t>(ktry)] == kleft) {
                        stay = true;
                        break;
                    }
                }
                if (!stay) {
                    nfuzz[static_cast<size_t>(kwalk)] = kleft;
                    break;
                }
            }
        }
    }

    std::vector<double> rdraw(static_cast<size_t>(k));
    for (int m = 0; m < nn; ++m) {
        for (int i = 0; i < k; ++i) {
            rdraw[static_cast<size_t>(i)] = p[m + (nfuzz[static_cast<size_t>(i)] - 1) * nn];
        }
        for (int i = 0; i < k; ++i) {
            p[m + i * nn] = rdraw[static_cast<size_t>(i)];
        }
    }
}

}  // namespace

FannyCoreResult fanny_from_diss(
    const double* diss,
    int n,
    int k,
    double memb_exp,
    int max_iter,
    double tol,
    const double* ini_membership,
    bool reorder_columns
) {
    if (n < 1) {
        throw std::runtime_error("n must be positive");
    }
    if (k < 1) {
        throw std::runtime_error("k must be at least 1");
    }
    if (k > n / 2 - 1) {
        throw std::runtime_error("k must be at most n//2 - 1");
    }
    if (memb_exp <= 1.0 || !std::isfinite(memb_exp)) {
        throw std::runtime_error("memb_exp must be finite and > 1");
    }
    if (max_iter < 0) {
        throw std::runtime_error("max_iter must be non-negative");
    }

    std::vector<double> dss;
    square_to_condensed(diss, n, dss);

    std::vector<double> p(static_cast<size_t>(n) * static_cast<size_t>(k));
    const bool compute_p = (ini_membership == nullptr);
    if (!compute_p) {
        // ini_membership is row-major n x k -> column-major for fuzzy()
        for (int m = 0; m < n; ++m) {
            for (int j = 0; j < k; ++j) {
                p[static_cast<size_t>(m) + static_cast<size_t>(j) * static_cast<size_t>(n)] =
                    ini_membership[static_cast<size_t>(m) * static_cast<size_t>(k) + static_cast<size_t>(j)];
            }
        }
    }

    std::vector<double> dp(static_cast<size_t>(n) * static_cast<size_t>(k), 0.0);
    std::vector<double> pt(static_cast<size_t>(k));
    std::vector<double> esp(static_cast<size_t>(k));
    std::vector<double> ef(static_cast<size_t>(k));

    int iterations = 0;
    bool converged = false;
    double objective = 0.0;
    double pc = 0.0;
    double npc = 0.0;

    fuzzy_core(
        n, k, p.data(), dp.data(), pt.data(), dss.data(),
        esp.data(), ef.data(), memb_exp, tol, max_iter, compute_p,
        iterations, converged, objective, pc, npc
    );

    int k_crisp = k;
    std::vector<int> ncluv_1based(static_cast<size_t>(n), 1);
    if (reorder_columns) {
        caddy_core(n, k, p.data(), k_crisp, ncluv_1based.data());
    } else {
        for (int m = 0; m < n; ++m) {
            int best_j = 0;
            double best_p = p[static_cast<size_t>(m)];
            for (int j = 1; j < k; ++j) {
                const double val = p[static_cast<size_t>(m) + static_cast<size_t>(j) * static_cast<size_t>(n)];
                if (val > best_p) {
                    best_p = val;
                    best_j = j;
                }
            }
            ncluv_1based[static_cast<size_t>(m)] = best_j + 1;
        }
    }

    FannyCoreResult out;
    out.membership.resize(static_cast<size_t>(n) * static_cast<size_t>(k));
    for (int m = 0; m < n; ++m) {
        for (int j = 0; j < k; ++j) {
            out.membership[static_cast<size_t>(m) * static_cast<size_t>(k) + static_cast<size_t>(j)] =
                p[static_cast<size_t>(m) + static_cast<size_t>(j) * static_cast<size_t>(n)];
        }
    }
    out.clustering.resize(static_cast<size_t>(n));
    for (int m = 0; m < n; ++m) {
        out.clustering[static_cast<size_t>(m)] = ncluv_1based[static_cast<size_t>(m)] - 1;
    }
    out.objective = objective;
    out.partition_coefficient = pc;
    out.normalized_coefficient = npc;
    out.iterations = iterations;
    out.k_crisp = k_crisp;
    out.converged = converged;
    return out;
}
