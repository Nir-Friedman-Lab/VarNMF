from lib_code.EM_lib import *
from NMF_code.NMF import *
from EM_code.EM import EM


class ExtendedNMFasEM(EM):
    def __init__(self, data, start_params, hyper_params, dims):
        super().__init__(data, start_params, hyper_params, dims)
        assert hyper_params.solver == 'mu'
        self.solver = hyper_params.solver
        self.beta = hyper_params.beta
        self.init = hyper_params.init_NMF
        self.NMF_model = ExtendedNMF(
            dims=dims, init=self.init,
            beta=self.beta, solver=self.solver, start_k_W=np.arange(dims.K), start_k_H=np.arange(dims.K)
        )

    @timer
    def log_likelihood(self, V=None, W=None, H=None, S=None, return_mean=True):
        V = V if V is not None else self.V
        S = S if S is not None else self.bg
        W = W if W is not None else self.params.W
        H = H if H is not None else self.params.H
        if self.beta == 'KL':
            ll = poisson_logpmf(V, W @ H.T + S)
        elif self.beta == 'F':
            ll = norm.logpdf(V, W @ H.T + S)
        else:
            assert False
        if return_mean: return np.mean(ll)
        return ll

    @timer
    def fit(self):
        W, H, ll_scores = self.NMF_model.fit(X=self.V, W=self.params.W, H=self.params.H.T, S=self.bg, return_ll=True)
        return W, H.T, ll_scores

    def predict_H(self, return_var=False):
        pass

    def get_cdf(self, data):
        if self.beta == 'KL':
            return poisson.cdf(data, self.params.W @ self.params.H.T + self.bg)
        elif self.beta == 'F':
            return norm.cdf(data, self.params.W @ self.params.H.T + self.bg)
        else:
            assert False


class ExtendedNMFasEMkH(ExtendedNMFasEM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(a:true_a, b:true_b, bg:bg=0), a, b = true a, b, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        start_params.H = hyper_params.H
        super().__init__(data, start_params, hyper_params, dims)

    def fit(self):
        W, H, ll_scores = self.NMF_model.fit_W(X=self.V, W=self.params.W, H=self.params.H.T, S=self.bg, return_ll=True)
        return W, H.T, ll_scores


class ExtendedNMFasEMkW(ExtendedNMFasEM):
    def __init__(self, data, start_params, hyper_params, dims):
        """
        :param data: np.array NxM
        :param start_params: Params(W:start_W, a:start_a=None, b:start_b=None)
        :param hyper_params: HyperParams(a:true_a, b:true_b, bg:bg=0), a, b = true a, b, will not change during the run
        :param dims: Dims(N:N, M:M, K:K)
        """
        start_params.H = hyper_params.H
        super().__init__(data, start_params, hyper_params, dims)
        # flip model dimensions  TODO replace by adding fit_H
        self.NMF_model = ExtendedNMF(
            dims=Dims(N=dims.M, M=dims.N, K=dims.K), init=self.init,
            beta=self.beta, solver=self.solver, start_k_W=np.arange(dims.K), start_k_H=np.arange(dims.K)
        )

    def fit(self):
        W, H, ll_scores = self.NMF_model.fit_W(X=self.V.T, W=self.params.H, H=self.params.W.T, S=self.bg.T, return_ll=True)
        return H.T, W, ll_scores
