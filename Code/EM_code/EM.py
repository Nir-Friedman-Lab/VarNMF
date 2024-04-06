import numpy as np

from lib_code.EM_lib import *

@njit
def _forward(prob_table, v, K, precision=np.float64):
    """
    Calculates F[s, n] = P(X_s=n) = P(sum_l(y_l, l=0,...,s+1)=n) for s=0,...,K-2, n=0,...,v
    For each s, n, p(y_s=n) = NB(n | a_j[s], c_ij[s]) if > eps[s] else 0.

    F[0, n] = p(y_0=n)
    For s=1,...,K-2:
        F[s, n] = sum_d(F[s-1, d] * p(y_s=n-d), d=0,...,n) for n=0,...,v
    :return: F
    """
    F = np.zeros((K - 1, v + 1), dtype=precision)
    F[0] = prob_table[0]
    for s in np.arange(1, K - 1):
        F[s] = [lse0(np.array([F[s - 1, d] + prob_table[s, n - d] for d in range(n + 1)])) for n in range(v + 1)]
    return F


@njit
def _backward(prob_table, v, K, precision=np.float64):
    """
    Calculates B[s, n] = P(Z_s=n) = P(sum_l(y_l, l=s+1,...,K-1)=n) for s=0,...,K-2, n=0,...,v
    For each s, n, p(y_s=n) = NB(n | a_j[s], c_ij[s]) if > eps[s] else 0.

    B[K-2, n] = p(y_K-1=n)
    For s=K-3,...,0:
        B[s, n] = sum_d(B[s+1, d] * p(y_s+1=n-d), d=0,...,n) for n=0,...,v
    :return: B
    """
    B = np.zeros((K - 1, v + 1), dtype=precision)
    B[K - 2] = prob_table[K - 2]
    for s in np.arange(K - 3, -1, -1):
        B[s] = [lse0(np.array([B[s + 1, d] + prob_table[s, n - d] for d in range(n + 1)])) for n in range(v + 1)]
    return B


@njit
def _p_mutual_v_y(F, B, prob_table, v, K, precision=np.float64):
    """
    Calculates P[k, t] = P(sum_l(y_l, l!=k)=v-t, y_k=t) for k=0,...,K-1, t=0,...,v
    For each k, n, p(y_k=n) = NB(n | a_j[k], c_ij[k]) if > eps[k] else 0.

    For t=0,...,v: P[0, t] = p(y_0=t) * B[0, v-t]
    For t=0,...,v: P[K-1, t] = p(y_K-1=t) * F[K-2, v-t]
    For k=1,...,K-2, t=0,...,v: P[k, t] = p(y_k=t) * sum(B[k, v-t-n] * F[k-1, n] for n=0,...,v-t)
    :return: P
    """
    P = np.zeros((K, v + 1), dtype=precision)
    P[-1] = F[-1][::-1]
    P[0] = B[0][::-1]
    for k in np.arange(K - 2):
        P[k+1] = [lse0(np.array([B[k + 1, v - t - n] + F[k, n] for n in range(v + 1 - t)])) for t in range(v + 1)]
    return P + prob_table


@njit(parallel=True)
def _log_posterior_y_ij(V, log_prob_table_all, max_V, N, M, K, normalize, precision=np.float64):
    ll = np.zeros((N, M)) - np.inf
    posterior = np.zeros((N, M, K + 1, max_V + 1), dtype=precision) - np.inf
    for i in prange(N):
        for j in prange(M):
            v = V[i, j]
            if v == 0:
                temp = np.sum(log_prob_table_all[i, j][:, 0])
                ll[i, j] = temp
                posterior[i, j, :, 0] = np.array([temp for k in range(K+1)]) - (ll[i, j] if normalize else 0)
            else:
                log_prob_table = log_prob_table_all[i, j, :, :v + 1]
                F = _forward(log_prob_table[:-1], v, K + 1, precision=precision)
                B = _backward(log_prob_table[1:], v, K + 1, precision=precision)
                P = _p_mutual_v_y(F, B, log_prob_table, v, K + 1, precision=precision)
                ll[i, j] = lse0(P[0])  # should be equal for all k
                posterior[i, j, :, :v + 1] = P - (ll[i, j] if normalize else 0)
    return posterior, ll


def _log_posterior_y_ij_joint2D(i, j, k1, k2, V, log_prob_table_all, K, precision=np.float64):
    v = V[i, j]
    log_prob_table_ij = log_prob_table_all[i, j]
    log_prob_table_ij_reorder = log_prob_table_ij[[k1, k2]+[l for l in range(K+1) if l not in [k1, k2]]]
    if v == 0: # TODO
        P = log_prob_table_ij_reorder[0] + log_prob_table_ij_reorder[1]
    else:
        log_prob_table = log_prob_table_ij_reorder[:, :v + 1]
        B = _backward(log_prob_table[1:], v, K + 1, precision=precision)
        xx1, xx2 = np.meshgrid(np.arange(v+1), np.arange(v+1))
        P = log_prob_table[0][xx1] + log_prob_table[1][xx2] + B[1][v-xx1-xx2]
        P[v-xx1-xx2 < 0] = -np.inf
    return P

@njit(parallel=True)
def _log_cdf(V, log_prob_table_all, log_bg_cdf_table_all, N, M, K, precision=np.float64):
    log_cdf = np.zeros((N, M), dtype=precision) - np.inf
    for i in prange(N):
        for j in prange(M):
            v = V[i, j]
            if v == 0:  # p(V<=v) = p(V=0) = p(yk=0 for k=0-K)
                log_cdf[i, j] = np.sum(log_prob_table_all[i, j][:, 0])
            else:
                # p(V<=v) = sum_t[B[t,0]*p(y0<=v-t),t=0-v]
                log_prob_table = log_prob_table_all[i, j, :, :v + 1]
                log_bg_cdf_table = log_bg_cdf_table_all[i, j, :v + 1] if log_bg_cdf_table_all.shape[0] > 1 else log_bg_cdf_table_all[0, 0, :v + 1]  # if bg is not ndarry
                B = _backward(log_prob_table[1:], v, K + 1, precision=precision)
                log_cdf[i, j] = lse0(B[0] + log_bg_cdf_table[::-1])
    return log_cdf


class EM:

    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W=None, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(bg:bg=0)
        :param dims: Dims(N:N, M:M, K:K)
        """
        self.precision = np.float32
        self.V = data
        self.params = start_params
        for p, param in start_params.__dict__.items():  # start params validity check
            if param is not None and type(param) == np.ndarray:
                start_params.__dict__[p][param < EPS] = EPS

        if np.all(self.params.lam != 1):
            self.params.W = self.params.W * self.params.lam
            self.params.lam = np.ones_like(self.params.lam)

        self.statistics = Statistics()

        self.bg = hyper_params.bg
        if type(self.bg) != np.ndarray: self.bg = self.bg + EPS
        else: self.bg[self.bg < EPS] = EPS

        self.N, self.M, self.K = dims
        self.dims = dims

        self.arrM_AB = None
        self.arrK_AB = None
        self.max_V = None
        self.arr_maxV = None
        self.arr_loggamma_maxV_p1 = None
        self.log_y_prob_table = None
        self.log_bg_prob_table = None
        self.log_bg_cdf_table = None

    def construct_em_tables(self, construct_cdf=False):
        temp = np.array([[(j, k) for k in range(self.K)] for j in range(self.M)])
        self.arrM_AB = temp.T[0].T
        self.arrK_AB = temp.T[1].T

        self.max_V = np.max(self.V)
        self.arr_maxV = np.arange(self.max_V + 1)
        self.arr_loggamma_maxV_p1 = loggamma(self.arr_maxV + 1)
        self.log_y_prob_table = np.empty((self.N, self.M, self.K + 1, self.max_V + 1), dtype=self.precision)
        self.log_bg_prob_table = poisson_logpmf(self.arr_maxV[:, None, None], self.bg).transpose((1, 2, 0))  # p(Y[i]_kj=t|R=bg[i]_j), shape = (N, M, K+1, max_V+1)

        if construct_cdf:
            self.log_bg_cdf_table = np.empty((self.N, self.M, self.max_V + 1), dtype=self.precision) if type(self.bg) == np.ndarray else np.empty((1, 1, self.max_V + 1), dtype=self.precision) # p(Y[i]_kj>=t|R=bg[i]_j), shape = (N, M, K+1, max_V+1)
            self.log_bg_cdf_table[:, :, 0] = self.log_bg_prob_table[:, :, 0]
            for t in range(1, self.max_V + 1):
                self.log_bg_cdf_table[:, :, t] = lse2axis0(np.array([self.log_bg_cdf_table[:, :, t-1], self.log_bg_prob_table[:, :, t]]))

    def _create_c(self):
        self.c = self.params.W[:, None, :] / (self.params.b + self.params.W[:, None, :])

    def nbinom_logpmf(self, r, p):
        # k! = gamma(k + 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            # out = loggamma(k + lam) - loggamma(k + 1) - loggamma(lam) + lam * np.log(1 - p) + np.where(k != 0, k * np.log(p), 0)
            k = self.arr_maxV[:, None, None, None]
            loggamma_k = self.arr_loggamma_maxV_p1[:, None, None, None]
            out = loggamma(k + r) - loggamma_k - loggamma(r) + r * np.log(1 - p) + k * np.log(p)
            # out2 = gammaln(k + lam) - gammaln(k + 1) - gammaln(lam) + lam * np.log(p) + xlog1py(k, -p)
        return out

    def _create_y_prob_table(self, construct_cdf=False):
        if self.log_bg_prob_table is None: self.construct_em_tables(construct_cdf)
        # N x M x K+1 x max(V)+1
        self.log_y_prob_table[:, :, 0] = self.log_bg_prob_table
        self.log_y_prob_table[:, :, 1:] = self.nbinom_logpmf(self.params.a, self.c).transpose((1, 2, 3, 0))
        # fix rvs with 0 probability for all values 0,...,v - give prob 1 to y=0. todo right solution?
        assert np.isnan(self.log_y_prob_table).sum() == 0
        # temp = np.isnan(self.log_y_prob_table)
        # self.log_y_prob_table[temp] = -np.inf
        temp = np.all(self.log_y_prob_table == -np.inf, axis=-1)
        self.log_y_prob_table[temp] = [0] + [-np.inf] * self.max_V

    def _log_posterior_y(self, normalize=True, return_ll=False, save_ll=True, remove_bg=True):
        log_posterior, ll = _log_posterior_y_ij(self.V, self.log_y_prob_table, self.max_V,
                                                self.N, self.M, self.K, normalize, precision=self.precision)
        if remove_bg: log_posterior = log_posterior[:, :, 1:]  # remove background posterior
        if save_ll: self.ll = np.mean(ll)
        if return_ll: return log_posterior, ll
        return log_posterior

    def _calc_e_y(self, posterior): return np.sum(self.arr_maxV * posterior, axis=-1)

    def _calc_e_h(self, e_y): return (self.params.a + e_y) / (self.params.b + self.params.W[:, None, :])

    def _calc_e_log_h(self, posterior): return np.sum([digamma(self.arr_maxV + self.params.a[None, :, k, None]) * posterior[:, :, k] for k in range(self.K)],
                                                      axis=-1).transpose((1, 2, 0)) - np.log(self.params.b + self.params.W[:, None, :])

    @timer
    def _calc_ess_expectations(self, posterior):
        e_y = self._calc_e_y(posterior)
        e_h = self._calc_e_h(e_y)
        e_log_h = self._calc_e_log_h(posterior)
        return e_y, e_h, e_log_h

    @timer
    def _calc_statistics(self, e_y, e_h, e_log_h):
        G = np.sum(e_y, axis=1)  # sum over j - genes
        T = np.sum(e_h, axis=1)  # sum over j - genes
        S_0 = self.N
        S_1 = np.sum(e_h, axis=0)  # sum over i - samples
        S_log = np.sum(e_log_h, axis=0)  # sum over i - samples

        return Statistics(G=G, T=T, S_0=S_0, S_1=S_1, S_log=S_log)

    def _mle_w(self, G, T):
        """
        Given the expected statistics G,T, return the mle w_hat for l*(w).
        :return: mle w_hat for l*(w)
        """
        if G is None: return None
        W = G / T
        W[W < EPS] = EPS
        return W

    def _mle_a_b(self, S_0, S_1, S_log, arrM_AB=None, arrK_AB=None):
        """
        Given the expected statistics S_1, S_log, return the mle a_hat, b_hat for l*(a,b).
        :return: mle a_hat, b_hat for l*(a,b)
        """
        if S_1 is None: return None, None
        if arrM_AB is None: arrM_AB = self.arrM_AB
        if arrK_AB is None: arrK_AB = self.arrK_AB
        return fit_gamma(S_0, S_1, S_log, self.K, self.M, arrM_AB, arrK_AB)

    @timer
    def _e_step(self):
        print('start e step')
        self._create_c()
        self._create_y_prob_table()
        posterior = np.exp(self._log_posterior_y())
        e_y, e_h, e_log_h = self._calc_ess_expectations(posterior)
        del posterior
        return self._calc_statistics(e_y, e_h, e_log_h)

    @timer
    def _m_step(self):
        """
        Given the expected statistics G, T, S_1, S_log, return the mle w_hat, a_hat, b_hat for l*(w,a,b).
        :return: mle w_hat, a_hat, b_hat for l*(w,a,b)
        """
        print('start m step')
        W = self._mle_w(self.statistics.G, self.statistics.T)
        try:
            a, b = self._mle_a_b(self.statistics.S_0, self.statistics.S_1, self.statistics.S_log)
        except: # change precision
            assert self.precision != np.float64, 'precision is already float64'
            self.precision = np.float64
            self.statistics = self._e_step()
            W = self._mle_w(self.statistics.G, self.statistics.T)
            a, b = self._mle_a_b(self.statistics.S_0, self.statistics.S_1, self.statistics.S_log)

        return Params(W=W, a=a, b=b)

    @timer
    def log_likelihood(self, return_mean=True):
        self._create_c()
        self._create_y_prob_table()
        _, ll = self._log_posterior_y(return_ll=True, save_ll=False)
        if return_mean: return np.mean(ll)
        return ll

    @timer
    def get_posterior(self, remove_bg=True):
        self._create_c()
        self._create_y_prob_table()
        log_posterior, ll = self._log_posterior_y(return_ll=True, save_ll=False, remove_bg=remove_bg)
        return np.exp(log_posterior)

    # this cannot run... todo
    # def _calc_posterior_H(self, posterior_Y, H_vals):
    #     # H_vals.shape = (n_vals, M)
    #     # poisson likelihood + gamma prior => gamma posterior
    #     # p(h | y=t) = Gamma(a+t,b+w)
    #     p_h_given_y = gamma_logpdf(H_vals[:, None, :, None, None], self.params.a[None, None, :, :, None] + self.arr_maxV[None, None, None, None, :], self.params.b[None, None, :, :, None] + self.params.W[None, :, None, :, None])
    #     return np.exp(logsumexp(p_h_given_y + posterior_Y, axis=-1))

    @timer
    def predict_H(self, return_var=False):
        self._create_c()
        self._create_y_prob_table()
        log_posterior, ll = self._log_posterior_y(return_ll=True, save_ll=False)
        posterior = np.exp(log_posterior)

        e_y = np.sum(self.arr_maxV * posterior, axis=-1)
        e_h = (self.params.a + e_y) / (self.params.b + self.params.W[:, None, :])
        if not return_var: return e_h

        var_y = np.sum(np.square(self.arr_maxV[None, None, None, :] - e_y[:, :, :, None]) * posterior, axis=-1)
        var_h = (self.params.a + e_y + var_y) / np.square(self.params.b + self.params.W[:, None, :])
        return e_h, var_h, ll

    @timer
    def get_cdf(self, data):
        self._create_c()
        self._create_y_prob_table(construct_cdf=True)
        return np.exp(_log_cdf(data, self.log_y_prob_table, self.log_bg_cdf_table, self.N, self.M, self.K, precision=self.precision))

    def get_p_vals(self, data, return_adjusted=True):
        # cdf(V)
        cdf = self.get_cdf(data)
        # cdf(V-1)
        temp_data = np.copy(data) - 1
        temp_data[temp_data < 0] = 0
        cdf_m1 = self.get_cdf(temp_data)
        cdf_m1[temp_data < 0] = 0
        pvals = 2 * np.min([1 - cdf_m1, cdf], axis=0)
        pvals[pvals < 0] = 0
        if return_adjusted:
            qvals = fdr(pvals)
            return pvals, qvals
        return pvals

    def run_itr(self, itr):
        self.params = self._m_step()
        self.statistics = self._e_step()
        print('finished run_itr')

    def _while_cond_run_itr(self, itr, itr_step_calc_ll, eps, last_diff, max_itr):
        return itr <= itr_step_calc_ll or (eps > 0 and last_diff > eps) or (max_itr > 0 and itr < max_itr)

    def _fit_generator(self, cur_itr, last_diff, ll, eps, max_itr=0, itr_step_calc_ll=1):
        self.ll = ll
        self.statistics = self._e_step()

        if cur_itr == 0:
            cur_itr += 1
            yield cur_itr, self.ll, last_diff, self.params

        while self._while_cond_run_itr(cur_itr, itr_step_calc_ll, eps, last_diff, max_itr):
            last_score = self.ll
            self.run_itr(cur_itr)
            last_diff = self.ll - last_score
            assert last_diff > 0 or np.abs(last_diff)**2 < EPS, 'll decrease'
            cur_itr += 1
            yield cur_itr, self.ll, last_diff, self.params

    def fit(self, eps=0, max_itr=0, itr_step_calc_ll=1):
        cur_itr = 0
        ll = -np.inf
        last_diff = 0
        ll_scores = []
        prog = ParamsProg()
        hat = None

        run_em_generator = self._fit_generator(
            cur_itr, last_diff, ll, eps, max_itr=max_itr, itr_step_calc_ll=itr_step_calc_ll
        )
        for cur_itr, ll, last_diff, params in run_em_generator:
            print_em_itr(cur_itr, ll, last_diff)

            ll_scores.append(ll)
            hat = params
            prog = prog + params

        return ll_scores, prog, hat
