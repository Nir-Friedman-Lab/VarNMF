import numpy as np

from lib_code.EM_lib import *
from EM_code.EM import EM


class EstEstepEM(EM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(bg:bg=0, inc_step=10, ll_check=10)
                :param dims: Dims(N:N, M:M, K:K)
        """
        super().__init__(data, start_params, hyper_params, dims)
        self.itr_size = np.min([50, dims.M])

        temp = np.array([[(j, k) for k in range(self.K)] for j in range(self.itr_size)])
        self.arrM_AB = temp.T[0].T
        self.arrK_AB = temp.T[1].T

        self.ll_check = hyper_params.ll_check
        self.max_V = np.max(self.V)
        self.arr_maxV = np.arange(self.max_V + 1)

    @timer
    def _e_step(self):
        pass

    def _get_batch_EM_obj(self, data, dims_batch, min_idx, max_idx):
        params_batch = Params(W=self.params.W, a=self.params.a[min_idx:max_idx], b=self.params.b[min_idx:max_idx])
        hyper_batch = HyperParams(bg=self.bg[:, min_idx:max_idx] if type(self.bg) == np.ndarray else self.bg)
        em = EM(data[:, min_idx:max_idx], params_batch, hyper_batch, dims_batch)
        em.precision = self.precision
        em.max_V = np.max(em.V)
        em.arr_maxV = np.arange(em.max_V + 1)
        em._create_c()
        em._create_y_prob_table()
        return em

    def _mle_a_b(self, S_0, S_1, S_log, arrM_AB=None, arrK_AB=None):
        if S_1 is None: return None, None

        a, b = np.empty((self.M, self.K)), np.empty((self.M, self.K))
        itr_size = self.itr_size
        for i in tqdm(np.arange(0, self.M, itr_size), desc='EstEstepEM Mstep AB', disable=disableTQDM):
            arrM_AB, arrK_AB = self.arrM_AB, self.arrK_AB
            if i+itr_size > self.M:
                l = a[i:i+itr_size].shape[0]
                arrM_AB, arrK_AB = self.arrM_AB[:l], self.arrK_AB[:l]
            a[i:i + itr_size], b[i:i + itr_size] = fit_gamma(
                S_0, S_1[i:i+itr_size], S_log[i:i+itr_size], self.K, self.itr_size, arrM_AB, arrK_AB
            )
        return a, b

    def _log_posterior_batch_helper(self, normalize=True, return_ll=False, save_ll=True, log=True, remove_bg=True, M=None):
        M = M if M is not None else self.M
        itr_size = self.itr_size
        ll = np.zeros((self.N, M)) - np.inf
        log_posterior = np.zeros((self.N, M, self.K if remove_bg else (self.K+1), self.max_V + 1), dtype=self.precision) - (np.inf if log else 0)

        for i in tqdm(np.arange(0, M, itr_size), desc='EstEstepEM Estep calc posterior', disable=disableTQDM):
            dims_batch = Dims(N=self.N, M=self.V[:, i:i+itr_size].shape[1], K=self.K)
            em = self._get_batch_EM_obj(self.V, dims_batch, min_idx=i, max_idx=i+itr_size)
            with HidePrints():
                res = em._log_posterior_y(normalize=normalize, return_ll=return_ll or save_ll, save_ll=False, remove_bg=remove_bg)
            if return_ll or save_ll:
                log_posterior[:, i:i+itr_size, :, :em.max_V+1] = res[0] if log else np.exp(res[0])
                ll[:, i:i+itr_size] = res[1]
            else: log_posterior[:, i:i+itr_size, :, :em.max_V+1] = res if log else np.exp(res)
            del res

        if save_ll: self.ll = np.mean(ll)
        if return_ll: return log_posterior, ll
        return log_posterior

    @timer
    def _posterior_y(self, normalize=True, return_ll=False, save_ll=True, remove_bg=True):
        return self._log_posterior_batch_helper(normalize, return_ll, save_ll, log=False, remove_bg=remove_bg)

    def _log_posterior_y(self, normalize=True, return_ll=False, save_ll=True, remove_bg=True):
        return self._log_posterior_batch_helper(normalize, return_ll, save_ll, log=True, remove_bg=remove_bg)

    def log_likelihood(self, return_mean=True):
        itr_size = self.itr_size
        ll = np.zeros((self.N, self.M)) - np.inf

        for i in tqdm(np.arange(0, self.M, itr_size), desc='EstEstepEM log-likelihood', disable=disableTQDM):
            dims_batch = Dims(N=self.N, M=self.V[:, i:i+itr_size].shape[1], K=self.K)
            em = self._get_batch_EM_obj(self.V, dims_batch, min_idx=i, max_idx=i+itr_size)
            with HidePrints():
                ll[:, i:i+itr_size] = em.log_likelihood(return_mean=False)

        if return_mean: return ll.mean()
        return ll

    def _calc_ess_expectations_batch_helper(self, return_ll=False, save_ll=True, remove_bg=True):
        itr_size = self.itr_size

        K = self.K if remove_bg else (self.K+1)
        e_y = np.zeros((self.N, self.M, K), dtype=self.precision)
        e_h = np.zeros((self.N, self.M, K), dtype=self.precision)
        e_log_h = np.zeros((self.N, self.M, K), dtype=self.precision)
        ll = np.zeros((self.N, self.M)) - np.inf

        for i in tqdm(np.arange(0, self.M, itr_size), desc='EstEstepEM Estep calc ESS', disable=disableTQDM):
            dims_batch = Dims(N=self.N, M=self.V[:, i:i + itr_size].shape[1], K=self.K)
            em = self._get_batch_EM_obj(self.V, dims_batch, min_idx=i, max_idx=i + itr_size)
            with HidePrints():
                # get posterior
                res = em._log_posterior_y(return_ll=(return_ll or save_ll))
                if return_ll or save_ll:
                    posterior = np.exp(res[0])
                    ll[:, i:i+itr_size] = res[1]
                else: posterior = np.exp(res[0])
                # get ESS
                res = em._calc_ess_expectations(posterior)
                e_y[:, i:i + itr_size] = res[0]
                e_h[:, i:i + itr_size] = res[1]
                e_log_h[:, i:i + itr_size] = res[2]

            del res

        if save_ll: self.ll = np.mean(ll)
        if return_ll: return (e_y, e_h, e_log_h), ll
        return e_y, e_h, e_log_h

    def predict_H(self, return_var=False):
        itr_size = self.itr_size
        ll = np.zeros((self.N, self.M)) - np.inf
        e_h, var_h = np.zeros((self.N, self.M, self.K)), np.zeros((self.N, self.M, self.K))

        for i in tqdm(np.arange(0, self.M, itr_size), desc='EstEstepEM predictH', disable=disableTQDM):
            dims_batch = Dims(N=self.N, M=self.V[:, i:i+itr_size].shape[1], K=self.K)
            em = self._get_batch_EM_obj(self.V, dims_batch, min_idx=i, max_idx=i+itr_size)
            with HidePrints():
                res_predict = em.predict_H(return_var=return_var)

            if not return_var: e_h[:, i:i + itr_size] = res_predict
            else: e_h[:, i:i + itr_size], var_h[:, i:i + itr_size], ll[:, i:i+itr_size] = res_predict

        if not return_var: return e_h
        return e_h, var_h, ll

    def get_cdf(self, data):
        itr_size = self.itr_size
        cdf = np.zeros((self.N, self.M)) - np.inf

        for i in tqdm(np.arange(0, self.M, itr_size), desc='EstEstepEM get_cdf', disable=disableTQDM):
            dims_batch = Dims(N=self.N, M=data[:, i:i+itr_size].shape[1], K=self.K)
            em = self._get_batch_EM_obj(self.V, dims_batch, min_idx=i, max_idx=i+itr_size)
            em.construct_em_tables(construct_cdf=True)
            with HidePrints():
                cdf[:, i:i+itr_size] = em.get_cdf(data[:, i:i+itr_size])

        return cdf

    def _while_cond_run_itr(self, itr, itr_step_calc_ll, eps, last_diff, max_itr):
        flag = False
        if itr_step_calc_ll < np.inf and itr <= itr_step_calc_ll + 1: flag = True
        if eps > 0 and not flag: flag = last_diff > eps
        if max_itr > 0 and not flag: flag = itr < max_itr
        if max_itr == 0 and not flag: flag = last_diff < -eps
        return flag

    def _fit_generator(self, cur_itr, last_diff, ll, eps, max_itr=0, itr_step_calc_ll=30):
        self.ll = ll
        self.statistics = self._e_step()

        if cur_itr == 0:
            cur_itr += 1
            if itr_step_calc_ll < np.inf: self.ll = self.log_likelihood()
            else: self.ll = -np.inf  # do not calculate ll
            yield cur_itr, self.ll, last_diff, self.params

        while self._while_cond_run_itr(cur_itr, itr_step_calc_ll, eps, last_diff, max_itr):
            last_score = self.ll
            self.run_itr(cur_itr)
            cur_itr += 1
            if (cur_itr % itr_step_calc_ll == 0) or not self._while_cond_run_itr(cur_itr, itr_step_calc_ll, eps, last_diff, max_itr):
                self.ll = self.log_likelihood()
                last_diff = self.ll - last_score if last_score > -np.inf else 0
                # print_em_itr(cur_itr, self.ll, last_diff)

            yield cur_itr, self.ll, last_diff, self.params

