import numpy as np

from lib_code.EM_lib import *
from EM_code.EstEstepEM import EstEstepEM
from EM_code.EMk import EMkH, EMkW, EMkAB


class batchEM(EstEstepEM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a, b:start_b)
        :param hyper_params: HyperParams(lam:lam, bg:bg=0, T1:T1=1000, T0:T0=0, ll_check:ll_check=10)
        :param dims: Dims(N:N, M:M, K:K)
        """
        super().__init__(data, start_params, hyper_params, dims)

    @timer
    def _e_step(self):
        print('start e step')
        e_y, e_h, e_log_h = self._calc_ess_expectations_batch_helper(return_ll=False, save_ll=True)
        return self._calc_statistics(e_y, e_h, e_log_h)

    def log_likelihood(self, return_mean=True):
        if return_mean: return self.ll
        else: return super().log_likelihood(return_mean)


class batchEMkAB(EMkAB, batchEM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(a:true_a, b:true_b, bg:bg=0), a, b = true a, b, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        super().__init__(data, start_params, hyper_params, dims)


class batchEMkW(EMkW, batchEM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(a:true_a, b:true_b, bg:bg=0), a, b = true a, b, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        super().__init__(data, start_params, hyper_params, dims)


class batchEMkH(EMkH, batchEM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(H:true_H, bg:bg=0), H = true H, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        super().__init__(data, start_params, hyper_params, dims)

    def _get_batch_EM_obj(self, data, dims_batch, min_idx, max_idx):
        params_batch = Params(W=self.params.W, a=self.params.H[min_idx:max_idx])
        hyper_batch = HyperParams(bg=self.bg[:, min_idx:max_idx] if type(self.bg) == np.ndarray else self.bg)
        em = EMkH(data[:, min_idx:max_idx], params_batch, hyper_batch, dims_batch)
        em.precision = self.precision
        em.max_V = np.max(em.V)
        em.arr_maxV = np.arange(em.max_V + 1)
        em._create_c()
        em._create_y_prob_table()
        return em

