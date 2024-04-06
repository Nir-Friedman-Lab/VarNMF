from sklearn.decomposition._nmf import NMF, _initialize_nmf, _special_sparse_dot, _fit_coordinate_descent
from sklearn.utils.extmath import squared_norm
from lib_code.numba_defs import *
from scipy.stats import norm

allowed_beta_loss = {'frobenius': 2,
                     'kullback-leibler': 1,
                     'itakura-saito': 0}
EPS = np.finfo(np.float32).eps

beta_losses = {'F': 'frobenius', 'KL': 'kullback-leibler'}


def trace_dot(X, Y): return np.dot(X.ravel(), Y.ravel())


def _beta_divergence2(X, W, H, beta, square_root=False, const_k_W=None, const_k_H=None, block=None, train=True, S=None):
    res = 0
    # Frobenius norm
    if beta == 2:
        if block is None: res = squared_norm(X - np.dot(W, H)) / 2.
        else:
            if train: res = squared_norm(X[~block] - np.dot(W, H)[~block]) / 2.
            else: res = squared_norm(X[block] - np.dot(W, H)[block]) / 2.

    # generalized Kullback-Leibler divergence
    if beta == 1:
        if S is None: S = 0
        WH = np.dot(W, H) + S
        WH_data = WH.ravel()
        X_data = X.ravel()

        if block is not None:
            if train:
                WH_data = WH_data[~block.ravel()]
                X_data = X_data[~block.ravel()]
            else:
                WH_data = WH_data[block.ravel()]
                X_data = X_data[block.ravel()]

        # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
        indices = X_data > EPS
        WH_data = WH_data[indices]
        X_data = X_data[indices]

        # used to avoid division by zero
        WH_data[WH_data == 0] = EPS

        # fast and memory efficient computation of np.sum(np.dot(W, H))
        # sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))  # doesn't include S
        sum_WH = WH_data.sum()
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    return res


def _multiplicative_update_w2(X, W, H, beta_loss, S=None,
                              H_sum=None, HHt=None, XHt=None, update_H=True,
                              l1_reg_W=0, l2_reg_W=0):
    """Update W in Multiplicative Update NMF."""
    if beta_loss == 2:
        # Numerator  ## TODO no background
        if XHt is None: XHt = np.dot(X, H.T)
        if not update_H: numerator = np.copy(XHt)  # preserve the XHt, which is not re-computed (update_H=False)
        else: numerator = XHt

        # Denominator
        if HHt is None: HHt = np.dot(H, H.T)
        denominator = np.dot(W, HHt)

    else:
        # Numerator
        WH_safe_X = _special_sparse_dot(W, H, X)
        WH_safe_X[WH_safe_X <= EPS] = EPS  # to avoid division by zero
        if S is not None: WH_safe_X += S
        np.divide(X, WH_safe_X, out=WH_safe_X)
        numerator = np.dot(WH_safe_X, H.T)  # here numerator = dot(X / dot(W, H), H.T)

        # Denominator
        if H_sum is None: H_sum = np.sum(H, axis=1)  # shape(n_components, )
        denominator = H_sum[np.newaxis, :]

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W

    # Divide
    denominator[denominator == 0] = EPS
    numerator /= denominator
    delta_W = numerator
    return delta_W, H_sum, HHt, XHt


def _multiplicative_update_h2(X, W, H, beta_loss, S=None,
                              l1_reg_H=0, l2_reg_H=0):
    """Update H in Multiplicative Update NMF."""
    if beta_loss == 2:
        numerator = np.dot(W.T, X)
        denominator = np.linalg.multi_dot([W.T, W, H])
    else:
        # Numerator
        WH_safe_X = _special_sparse_dot(W, H, X)
        WH_safe_X[WH_safe_X <= EPS] = EPS  # to avoid division by zero
        if S is not None: WH_safe_X += S
        np.divide(X, WH_safe_X, out=WH_safe_X)
        numerator = np.dot(W.T, WH_safe_X)  # here numerator = dot(W.T, X / dot(W, H))

        # Denominator
        W_sum = np.sum(W, axis=0)  # shape(n_components, )
        W_sum[W_sum == 0] = 1.
        denominator = W_sum[:, np.newaxis]


    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H

    denominator[denominator <= EPS] = EPS
    numerator /= denominator
    delta_H = numerator

    return delta_H


def _log_likelihood(X, W, H, beta_loss=2, S=None):
    l = W @ H + S
    if beta_loss == 1:
        l[l == 0] = EPS
        return poisson_logpmf(X, l).mean()
    elif beta_loss == 2:
        return norm.logpdf(X, l).mean()  # gaussian noise
    assert False


def _fit_multiplicative_update2(X, W, H, beta_loss='frobenius', max_iter=200, tol=1e-4,
                                const_k_W=None, const_k_H=None, block=None, S=None,
                                update_H=True, norm_H=None, return_ll=False,
                                l1_reg_W=0, l1_reg_H=0, l2_reg_W=0, l2_reg_H=0):
    beta_loss = allowed_beta_loss[beta_loss]

    if const_k_W is not None: W_copy = W.copy()
    if const_k_H is not None: H_copy = H.copy()

    # used for the convergence criterion
    error_at_init = _beta_divergence2(X, W, H, beta_loss, square_root=(beta_loss == 1),
                                      const_k_W=const_k_W, const_k_H=const_k_H, block=block, S=S)
    previous_error = error_at_init
    ll_scores = None
    if return_ll: ll_scores = [_log_likelihood(X, W, H, beta_loss=beta_loss, S=S)]

    H_sum, HHt, XHt = None, None, None
    for n_iter in tqdm(range(1, max_iter + 1), desc='NMF fit', disable=disableTQDM):
        # update W
        # delta_W2, _, _, _ = _multiplicative_update_w(X, W, H, beta_loss, 0, 0, 1)
        delta_W, H_sum, HHt, XHt = _multiplicative_update_w2(
            X, W, H, beta_loss, S=S,
            H_sum=H_sum, HHt=HHt, XHt=XHt, update_H=update_H,
            l1_reg_W=l1_reg_W, l2_reg_W=l2_reg_W
        )
        W *= delta_W
        if const_k_W is not None: W[:, const_k_W] = W_copy[:, const_k_W]


        # update H
        # delta_H2 = _multiplicative_update_h(X, W, H, beta_loss, 0, 0, 1)
        if update_H:
            delta_H = _multiplicative_update_h2(
                X, W, H, beta_loss, S=S, l1_reg_H=l1_reg_H, l2_reg_H=l2_reg_H
            )
            H *= delta_H
            if const_k_H is not None: H[const_k_H, :] = H_copy[const_k_H, :]

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

        # test convergence criterion every 10 iterations
        if not return_ll and tol > 0 and n_iter % 10 == 0:

            error = _beta_divergence2(X, W, H, beta_loss, square_root=(beta_loss == 1),
                                      const_k_W=const_k_W, const_k_H=const_k_H, block=block, S=S)
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error
            # print(error)
        if return_ll:
            ll_scores.append(_log_likelihood(X, W, H, beta_loss=beta_loss, S=S))
            if len(ll_scores) >= 2 and ll_scores[-1] - ll_scores[-2] < tol: break

    return W, H, n_iter, ll_scores


def non_negative_factorization2(X, W, H, n_components, init,
                                update_H=True, solver='mu', beta_loss='frobenius',
                                tol=1e-4, max_iter=10000, random_state=None,
                                const_k_W=None, const_k_H=None, block=None, S=None, norm_H=None,
                                start_k_W=None, start_k_H=None,
                                l1_reg_W=0, l1_reg_H=0, l2_reg_W=0, l2_reg_H=0,
                                return_ll=False):

    X = X * 1.

    init_W, init_H = _initialize_nmf(X, n_components, init=init, random_state=random_state)

    # if norm_H is not None:
    #     H = H /

    if start_k_W is not None:
        init_W[:, start_k_W] = W[:, start_k_W]
    if start_k_H is not None:
        init_H[start_k_H, :] = H[start_k_H, :]
    W = init_W
    if update_H: H = init_H

    ll_scores = None
    if solver == 'cd':
        # assert False
        W, H, n_iter = _fit_coordinate_descent(
            X, W, H, tol, max_iter, l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H,
            update_H=update_H, random_state=random_state)
        # W, H, n_iter = _fit_coordinate_descent(X, W, H, tol=tol, max_iter=max_iter, update_H=update_H)
    elif solver == 'mu':
        # W, H, n_iter = _fit_multiplicative_update(X, W, H, beta_loss=beta_loss, max_iter=max_iter, tol=tol)
        W, H, n_iter, ll_scores = _fit_multiplicative_update2(
            X, W, H, beta_loss=beta_loss, max_iter=max_iter, tol=tol, const_k_W=const_k_W, const_k_H=const_k_H,
            block=block, S=S, update_H=update_H, norm_H=norm_H, return_ll=return_ll,
            l1_reg_W=l1_reg_W, l1_reg_H=l1_reg_H, l2_reg_W=l2_reg_W, l2_reg_H=l2_reg_H)
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    return W, H, n_iter, ll_scores


class CustomNMF(NMF):
    def __init__(self, n_components, init='random', beta_loss='KL', solver='mu', max_iter=10000, tol=1e-6,
                 alpha_W=0., alpha_H=0., l1_ratio=1., random_state=None,
                 const_k_W=None, const_k_H=None, start_k_W=None, start_k_H=None, block=None):
        # assert alpha_W == 0
        # assert alpha_H == 0 ## TODO
        super().__init__(n_components=n_components, init=init, max_iter=max_iter,
                         beta_loss=beta_loss, solver=solver,
                         # alpha=0, l1_ratio=0,  ## TODO
                         random_state=random_state, tol=tol)
        self.const_k_W = [c for c in const_k_W if c < n_components] if const_k_W is not None else None
        self.const_k_H = [c for c in const_k_H if c < n_components] if const_k_H is not None else None
        self.start_k_W = [c for c in start_k_W if c < n_components] if start_k_W is not None else None
        self.start_k_H = [c for c in start_k_H if c < n_components] if start_k_H is not None else None

        self.block = block
        self.beta = allowed_beta_loss[beta_loss]
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF
        self.l1_reg_W = alpha_W * l1_ratio
        self.l1_reg_H = alpha_H * l1_ratio
        self.l2_reg_W = .5 * alpha_W * (1 - l1_ratio)
        self.l2_reg_H = .5 * alpha_H * (1 - l1_ratio)

    def fit_transform(self, X, y=None, W=None, H=None, S=None, return_ll=False):
        S = S if S is not None else 0
        if self.block is not None:
            X = X.copy()
            X[self.block] = 0
        W, H, n_iter_, ll_scores = non_negative_factorization2(
            X=X, W=W, H=H, n_components=self.n_components, init=self.init,
            update_H=True, solver=self.solver, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, random_state=self.random_state,
            const_k_W=self.const_k_W, const_k_H=self.const_k_H, block=self.block, S=S,
            start_k_W=self.start_k_W, start_k_H=self.start_k_H,
            l1_reg_W=self.l1_reg_W * X.shape[0], l1_reg_H=self.l1_reg_H * X.shape[1],
            l2_reg_W=self.l2_reg_W * X.shape[0], l2_reg_H=self.l2_reg_H * X.shape[1],
            return_ll=return_ll
        )

        self.reconstruction_err_ = _beta_divergence2(
            X, W, H, self.beta, square_root=(self.beta == 1), const_k_W=self.const_k_W, const_k_H=self.const_k_H,
            block=self.block, S=S
        )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_

        if return_ll: return W, ll_scores
        return W

    def define_H(self, H):
        self.n_components_ = H.shape[0]
        self.components_ = H
        return H

    def transform(self, X, W=None, S=None, return_ll=False):
        """Transform the data X according to the fitted NMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        S = S if S is not None else 0
        if self.block is not None:
            X = X.copy()
            X[self.block] = 0
        W, _, n_iter_, ll_scores = non_negative_factorization2(
            X=X, W=W, H=self.components_, S=S, n_components=self.n_components_, init=self.init,
            update_H=False, solver=self.solver, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, random_state=self.random_state,
            const_k_W=self.const_k_W, const_k_H=self.const_k_H, start_k_W=self.start_k_W, block=self.block,
            l1_reg_W=self.l1_reg_W * X.shape[0], l1_reg_H=self.l1_reg_H * X.shape[1],
            l2_reg_W=self.l2_reg_W * X.shape[0], l2_reg_H=self.l2_reg_H * X.shape[1],
            return_ll=return_ll
        )
        if return_ll: return W, ll_scores
        return W

    def train_loss(self, X, W, H, S=None, norm=True):
        S = S if S is not None else 0
        return _beta_divergence2(
            X, W, H, self.beta, square_root=(self.beta == 1), const_k_W=self.const_k_W, const_k_H=self.const_k_H,
            block=self.block, train=True, S=S
        ) / ((X.shape[0] * X.shape[1] - (self.block.sum() if self.block is not None else 0)) if norm else 1)

    def test_loss(self, X, W, H, S=None, norm=True):
        S = S if S is not None else 0
        return _beta_divergence2(
            X, W, H, self.beta, square_root=(self.beta == 1), const_k_W=self.const_k_W, const_k_H=self.const_k_H,
            block=self.block, train=False, S=S
        ) / ((self.block.sum() if self.block is not None else 0) if norm else 1)

    def reconstruction_loss(self, X, W, H, S=None, norm=True):
        S = S if S is not None else 0
        return _beta_divergence2(
            X, W, H, self.beta, square_root=(self.beta == 1), const_k_W=self.const_k_W, const_k_H=self.const_k_H,
            block=None, S=S
        ) / (X.shape[0] * X.shape[1] if norm else 1)


class ExtendedNMF:
    def __init__(self, dims, max_iter=10000, init='random', tol=1e-6,
                 beta='KL', solver='mu', alpha_W=0., alpha_H=0., l1_ratio=1.,
                 const_k_W=None, const_k_H=None, block=None, start_k_W=None, start_k_H=None):
        self.dims = dims
        self.NMF_model = CustomNMF(n_components=dims.K, init=init, max_iter=max_iter, tol=tol,
                             beta_loss=beta_losses[beta], solver=solver,
                             alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio,
                             const_k_W=const_k_W, const_k_H=const_k_H, block=block,
                             start_k_W=start_k_W, start_k_H=start_k_H)

    def fit(self, X, W=None, H=None, S=None, normalize=False, H_norm=1000, return_ll=False):
        res = self.NMF_model.fit_transform(X, W=W, H=H, S=S, return_ll=return_ll)
        if return_ll: W, ll_scores = res
        else: W = res

        # W[W < EPSILON] = EPSILON
        H = self.NMF_model.components_
        if normalize and H_norm > 0:
            assert False
            # return normalize_H(W, H, H_norm) # TODO
        if return_ll: return W, H, ll_scores
        return W, H

    def fit_W(self, X, W=None, H=None, S=None, normalize=False, H_norm=1000, return_ll=False):
        if H is not None: H = self.NMF_model.define_H(H)
        res = self.NMF_model.transform(X, W=W, S=S, return_ll=return_ll)
        if return_ll: W, ll_scores = res
        else: W = res

        if normalize and H_norm > 0:
            assert False
            # return normalize_H(W, H, H_norm) # TODO
        if return_ll: return W, H, ll_scores
        return W, H

    def get_H(self):
        return self.NMF_model.components_

    def train_loss(self, X, W, H, S=None, norm=True):
        return self.NMF_model.train_loss(X, W, H, S, norm=norm)

    def test_loss(self, X, W, H, S=None, norm=True):
        return self.NMF_model.test_loss(X, W, H, S, norm=norm)

    def reconstruction_loss(self, X, W, H, S=None, norm=True):
        return self.NMF_model.reconstruction_loss(X, W, H, S, norm=norm)

