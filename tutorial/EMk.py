from EM import *


class EMkAB(EM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(a:true_a, b:true_b, bg:bg=0), a, b = true a, b, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        start_params.a = hyper_params.a if hyper_params.a is not None else start_params.a
        start_params.b = hyper_params.b if hyper_params.b is not None else start_params.b
        super().__init__(data, start_params, hyper_params, dims)

    def _calc_e_log_h(self, posterior): return None

    def _calc_statistics(self, e_y, e_h, e_log_h):
        G = np.sum(e_y, axis=1)  # sum over j - genes
        T = np.sum(e_h, axis=1)  # sum over j - genes
        return Statistics(G=G, T=T)

    def _m_step(self):
        """
        Given the expected statistics G, T, return the mle w_hat for l*(w).
        :return: mle w_hat for l*(w)
        """
        params = super()._m_step()
        params.a = self.params.a
        params.b = self.params.b
        return params


class EMkH(EM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(H:true_H, bg:bg=0), H = true H, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        self.H = hyper_params.H if hyper_params.H is not None else start_params.H
        self.H[self.H < EPS] = EPS
        if len(self.H.shape) == 2: self.H = np.broadcast_to(self.H, dims)
        super().__init__(data, start_params, hyper_params, dims)
        self.statistics.T = np.sum(self.H, axis=1)  # sum over j - genes

    def _create_c(self):
        pass

    def _create_y_prob_table(self, construct_cdf=False):
        # N x M x K x max(V), adding some noise to lambda so it'll be > 0
        if self.log_bg_prob_table is None: self.construct_em_tables(construct_cdf)
        self.log_y_prob_table = np.zeros((self.N, self.M, self.K + 1, self.max_V + 1)) * 1.0
        self.log_y_prob_table[:, :, 0] = poisson_logpmf(self.arr_maxV[:, None, None],
                                                        self.bg + EPS).transpose((1, 2, 0))
        self.log_y_prob_table[:, :, 1:] = poisson_logpmf(self.arr_maxV[:, None, None, None],
                                                         self.H * self.params.W[:, None, :]).transpose((1, 2, 3, 0))
        # fix rvs with 0 probability for all values 0,...,v - give prob 1 to y=0. todo right solution?
        assert np.isnan(self.log_y_prob_table).sum() == 0
        # temp = np.isnan(self.log_y_prob_table)
        # self.log_y_prob_table[temp] = -np.inf
        temp = np.all(self.log_y_prob_table == -np.inf, axis=-1)
        self.log_y_prob_table[temp] = [0] + [-np.inf] * self.max_V

    def _calc_e_h(self, e_y): return None
    def _calc_e_log_h(self, posterior): return None

    def _calc_statistics(self, e_y, e_h, e_log_h):
        G = np.sum(e_y, axis=1)  # sum over j - genes
        T = self.statistics.T
        return Statistics(G=G, T=T)


    def _m_step(self):
        """
        Given the expected statistics G, T, return the mle w_hat for l*(w).
        :return: mle w_hat for l*(w)
        """
        params = super()._m_step()
        params.H = self.params.H
        return params


class EMkW(EM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(W:true_W, bg:bg=0), W = true W, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        start_params.W = hyper_params.W if hyper_params.W is not None else start_params.W
        super().__init__(data, start_params, hyper_params, dims)

    def _calc_statistics(self, e_y, e_h, e_log_h):
        S_1 = np.sum(e_h, axis=0)  # sum over i - samples
        S_log = np.sum(e_log_h, axis=0)  # sum over i - samples
        return Statistics(S_0=self.N, S_1=S_1, S_log=S_log)


    def _m_step(self):
        """
        Given the expected statistics G, T, S_1, S_log, return the mle w_hat, a_hat, b_hat for l*(w,a,b).
        :return: mle w_hat, a_hat, b_hat for l*(w,a,b)
        """
        params = super()._m_step()
        params.W = self.params.W
        return params


class EMkPostH(EM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        This is used only to calculate the log likelihood of the data given the true posterior H.  # TODO
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(W:true_W, posterior_H:true_posterior_H, bg:bg=0),
                                    W = true_W , posterior_H = true posterior_H, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        start_params.W = hyper_params.W if hyper_params.W is not None else start_params.W
        start_params.posterior_H = hyper_params.posterior_H if hyper_params.posterior_H is not None else start_params.posterior_H
        super().__init__(data, start_params, hyper_params, dims)

    def log_likelihood(self, return_mean=True):
        poisson_rate = (self.start_params.W[:, None] * self.start_params).posterior_H.sum(axis=-1).T
        poisson_rate[poisson_rate == 0] = EPS
        log_likelihood = poisson_logpmf(self.data, poisson_rate)
        if return_mean: return np.mean(log_likelihood)
        return log_likelihood