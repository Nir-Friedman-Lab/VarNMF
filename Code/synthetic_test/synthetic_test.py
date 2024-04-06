import os
import sys
sys.path.insert(1, os.getcwd())

from lib_code.gen_plot_lib import *
from EM_code.EM import *
from NMF_code.NMF import ExtendedNMF as NMF


def get_gamma_params(plots_path_i, M, K, cv):
    if (plots_path_i/f'true_gamma_params.pkl').exists():
        true_gamma_params = Params.load_params(plots_path_i, f'true_gamma_params')
        a, b, mu, var = true_gamma_params.a, true_gamma_params.b, true_gamma_params.H, true_gamma_params.var
    else:
        print('create new mu', flush=True)
        idx = np.random.choice(a=18000*2, size=M*K, replace=False)
        mu = pd.read_csv(root_path/'data/synthetic_test/H.csv', index_col=0).values.flatten()[idx].reshape(M, K)
        mu[mu < 1e-4] = 1e-4

        var = mu**2 * cv**2
        var[var < 1e-4] = 1e-4
        a, b = get_gamma_ab(mu.T, var.T)
        Params(a=a, b=b, var=var, H=mu).dump_params(plots_path_i, f'true_gamma_params')

    return a, b, mu, var


def sample_w(N, K, m_lam=2):
    lam = np.random.uniform(m_lam//2, m_lam, N)[:, None]
    w = np.random.dirichlet([1]*K, N)
    return lam, w


def sample_data(N, M, K, a, b, w, lam):
    h = gamma.rvs(a=a, scale=1/b, size=(N, K, M))
    r = (lam * w)[:, :, None] * h
    Y = np.random.poisson(r)
    x = np.sum(Y, axis=1)
    r = r.transpose((0, 2, 1))
    return h, Y, r, x


def calc_r(params): return params.W * params.H
# def calc_r(params): return params.W[:, None] * params.H.T


def calc_post_r(params):
    return params.W[:, None, :] * params.posterior_H


def nmf_ll(r, x): return poisson(r.sum(axis=-1)).logpmf(x)


def nmf_ll_mean(r, x): return nmf_ll(r, x).mean()


def min_EPS(x):
    return np.max([x, (np.zeros(x.shape)+EPS) if type(x) == np.ndarray else EPS], axis=0)


def add_noise(x, eps=.1):
    return min_EPS(np.random.normal(x, eps))


def get_em_start(nmf_params, N, K):
    temp = nmf_params.H
    temp[temp < EPS] = EPS
    a_em, b_em = get_gamma_ab(temp, np.square(temp))
    w_em = nmf_params.W
    w_em[w_em < EPS] = EPS

    return Params(W=w_em, a=a_em, b=b_em, H=(a_em/b_em))


def normalize(true_params, est_params):
    est_params = Params().copy(est_params)
    corr_mat = np.corrcoef(true_params.W, est_params.W, rowvar=False)[-est_params.W.shape[1]:, :true_params.W.shape[1]]
    idx = linear_sum_assignment(corr_mat.T, maximize=True)
    est_params.W = est_params.W[:, idx[1]]
    est_params.H = est_params.H[:, idx[1]]

    if est_params.a is not None:
        est_params.a = est_params.a[:, idx[1]]
        est_params.b = est_params.b[:, idx[1]]
        est_params.H = est_params.a / est_params.b

    if est_params.posterior_H is not None:
        est_params.posterior_H = est_params.posterior_H[:, :, idx[1]]

    d = est_params.H.sum(axis=0)
    norm_factor = d / true_params.H.sum(axis=0)
    est_params.H = est_params.H / norm_factor
    est_params.W = est_params.W * norm_factor

    if est_params.a is not None:
        est_params.b = est_params.b * norm_factor

    if est_params.posterior_H is not None:
        est_params.posterior_H = est_params.posterior_H / norm_factor[None, None, :]
    return est_params

# train
def train_NMF(x, dims):
    print('train NMF', flush=True)
    nmf_obj = NMF(dims, init='random', beta='KL', solver='mu', max_iter=10000)
    W, H, ll_scores = nmf_obj.fit(x, return_ll=True)
    nmf_params = Params(W=W, H=H.T)
    return nmf_obj, ll_scores, nmf_params


def train_EM(x, dims, nmf_params, plots_path_i, override=False,
             EM_class=EM, max_itr=0, eps=0, em_name='EM'):
    print(f'train {em_name}', flush=True)

    if not (plots_path_i/f'train_{em_name}_params.pkl').exists() or override:
        start_em_params = get_em_start(nmf_params, dims.N, dims.K)
        start_em_params.dump_params(plots_path_i, param_name=f'train_{em_name}_params')
    else:
        start_em_params = Params.load_params(plots_path_i, param_name=f'train_{em_name}_params')

    hyper_params = HyperParams()
    if 'Prior' in em_name:
        if not 'Params' in em_name: hyper_params.priorH_params = Params().copy(start_em_params)

    em = EM_class(data=x, start_params=start_em_params, hyper_params=hyper_params, dims=dims)
    ll_scores, prog, em_params = em.fit(max_itr=max_itr, eps=eps)
    em_params.posterior_H = em.predict_H()
    em_params.H = em_params.a / em_params.b
    return em, ll_scores, em_params


# test
def test_NMF(x, dims, train_nmf_params):
    print('test NMF', flush=True)
    nmf_obj = NMF(dims, init='random', beta='KL', solver='mu', max_iter=10000)
    W, H, ll_scores = nmf_obj.fit_W(x, H=train_nmf_params.H.T, return_ll=True)
    nmf_params = Params(W=W, H=H.T)
    return nmf_obj, ll_scores, nmf_params


def get_start_W_test(x, dims, H):
    nmf_obj = NMF(dims, init='random', beta='KL', solver='mu', max_iter=10000)
    start_W_test, _, _ = nmf_obj.fit_W(x, H=H.T, return_ll=True)
    return start_W_test


def test_EM(x, dims, train_em_params):
    print('test EM', flush=True)
    start_params = Params().copy(train_em_params)
    start_params.H = train_em_params.a / train_em_params.b
    start_params.W = get_start_W_test(x, dims, start_params.H)
    em = EMkAB(data=x, start_params=start_params, hyper_params=HyperParams(), dims=dims)
    ll_scores, prog, em_params = em.fit(eps=1e-6)
    em_params.posterior_H = em.predict_H()
    em_params.H = em_params.a / em_params.b
    return em, ll_scores, em_params

def create_data(plots_path_i, dims, prefix='train', override=False, m_lam=2, cv=1.):
    if not (plots_path_i/f'{prefix}_true_params.pkl').exists() or override:
        a, b, mu, var = get_gamma_params(plots_path_i, dims.M, dims.K, cv=cv)
        lam, w = sample_w(dims.N, dims.K, m_lam)
        h, Y, r, x = sample_data(dims.N, dims.M, dims.K, a, b, w, lam)
        true_params = Params(lam=lam, W=w, a=a.T, b=b.T, H=mu, posterior_H=h.transpose((0, 2, 1)))
        true_ll_scores = EM(x, true_params, hyper_params=HyperParams(), dims=dims).log_likelihood(return_mean=False)

        pd.DataFrame(x).to_csv(plots_path_i/f'{prefix}_data.csv')
        true_params.dump_params(plots_path_i, param_name=f'{prefix}_true_params')
        pd.DataFrame(true_ll_scores).to_csv(plots_path_i/f'{prefix}_true_ll_scores.csv')
    else:
        x = pd.read_csv(plots_path_i/f'{prefix}_data.csv', index_col=0).values
        true_params = Params.load_params(plots_path_i, param_name=f'{prefix}_true_params')
        if np.all(true_params.lam != 1):  # fix for ll calculations that didn't include lambda
            true_params.W = true_params.W * true_params.lam
            true_params.lam = np.ones_like(true_params.lam)
            true_params.dump_params(plots_path_i, param_name=f'{prefix}_true_params')
            true_ll_scores = EM(x, true_params, hyper_params=HyperParams(), dims=dims).log_likelihood(return_mean=False)
            pd.DataFrame(true_ll_scores).to_csv(plots_path_i / f'{prefix}_true_ll_scores.csv')
        true_ll_scores = pd.read_csv(plots_path_i/f'{prefix}_true_ll_scores.csv', index_col=0).values
    print('true ll', true_ll_scores.mean())
    return x, true_ll_scores, true_params  # x.shape = (N, M)


def fit_algo(plots_path_i, true_params, x, dims, algo_lambda, algo_name, prefix='train', override=False):
    if not (plots_path_i/f'{prefix}_{algo_name}_hat.pkl').exists() or override:
        print(plots_path_i/f'{prefix}_{algo_name}_hat.pkl')
        nmf_obj, est_ll_scores, est_params = algo_lambda(x)
        est_params.dump_params(plots_path_i, param_name=f'{prefix}_{algo_name}_hat')
        pd.DataFrame(est_ll_scores).to_csv(plots_path_i/f'{prefix}_{algo_name}_ll_scores.csv')
    else:
        est_ll_scores = pd.read_csv(plots_path_i/f'{prefix}_{algo_name}_ll_scores.csv', index_col=0).values
        est_params = Params.load_params(plots_path_i, param_name=f'{prefix}_{algo_name}_hat')

    if est_params.H.shape[0] != dims.M:  # fix for flipped H
        est_params.H = est_params.H.T
        est_params.dump_params(plots_path_i, param_name=f'{prefix}_{algo_name}_hat')
    if true_params.W.shape[1] == est_params.W.shape[1]:
        norm_est_params = normalize(true_params, est_params)  # this doesn't change the ll
    else: norm_est_params = None
    return est_ll_scores, est_params, norm_est_params


def train_all(plots_path_i, dims, cv, m_lam, run_algo_list, K_nmf=None, override=False, true_dims=None):

    x, true_ll_scores, true_params = create_data(plots_path_i, true_dims if true_dims is not None else dims,
                                                 prefix='train', override=False, cv=cv, m_lam=m_lam)

    results = {}  # (ll_scores, params, norm_params)
    results['true'] = {'ll': true_ll_scores.mean(), 'params': true_params, 'norm_params': true_params}

    if 'NMF' in run_algo_list:
        results['NMF'] = {'ll': None, 'params': None, 'norm_params': None}
        results['NMF']['ll'], results['NMF']['params'], results['NMF']['norm_params'] = fit_algo(
            plots_path_i, true_params, x, dims,
            algo_lambda=lambda x: train_NMF(x, dims),
            algo_name='NMF', prefix='train', override=override
        )
        nmf_params = results['NMF']['params']
        print('train', 'NMF', 'll', results['NMF']['ll'][-1])

    if r"NMF_tilde" in run_algo_list:
        results['NMF_tilde'] = {'ll': None, 'params': None, 'norm_params': None}
        results[r"NMF_tilde"]['ll'], results[r"NMF_tilde"]['params'], results[r"NMF_tilde"]['norm_params'] = fit_algo(
            plots_path_i, true_params, x, Dims(N=dims.N, M=dims.M, K=K_nmf),
            algo_lambda=lambda x: train_NMF(x, Dims(N=dims.N, M=dims.M, K=K_nmf)),
            algo_name=r"NMF_tilde", prefix='train', override=override
        )
        print('train', r"NMF_tilde", 'll', results[r"NMF_tilde"]['ll'][-1])

    for algo in ['EM']:
        EM_class = EM
        if algo in run_algo_list:
            results[algo] = {'ll': None, 'params': None, 'norm_params': None}
            results[algo]['ll'], results[algo]['params'], results[algo]['norm_params'] = fit_algo(
                plots_path_i, true_params, x, dims,
                algo_lambda=lambda x: train_EM(x, dims, nmf_params, plots_path_i, override=override,
                                               EM_class=EM_class, max_itr=100, em_name=algo),
                algo_name=algo, prefix='train', override=override
            )
            print('train', algo, 'll', results[algo]['ll'][-1])

    return x, results


def test_all(plots_path_i, dims, cv, m_lam, train_results_dict, run_algo_list, K_nmf=None, override=False, true_dims=None):

    x, true_ll_scores, true_params = create_data(plots_path_i, true_dims if true_dims is not None else dims,
                                                 prefix='test', override=False, cv=cv, m_lam=m_lam)

    results = {}  # (ll_scores, params, norm_params)
    results['true'] = {'ll': true_ll_scores.mean(), 'params': true_params, 'norm_params': true_params}

    for algo in ['NMF', 'NMF_tilde']:
        if algo in run_algo_list:
            nmf_dims = Dims(N=dims.N, M=dims.M, K=K_nmf) if algo == 'NMF_tilde' else dims
            results[algo] = {'ll': None, 'params': None, 'norm_params': None}
            results[algo]['ll'], results[algo]['params'], results[algo]['norm_params'] = fit_algo(
                plots_path_i, true_params, x, nmf_dims,
                algo_lambda=lambda x: test_NMF(x, nmf_dims, train_results_dict[algo]['params']),
                algo_name=algo, prefix='test', override=override
            )
            print('test', algo, 'll', results[algo]['ll'][-1])

    for algo in ['EM']:
        if algo in run_algo_list:
            results[algo] = {'ll': None, 'params': None, 'norm_params': None}
            results[algo]['ll'], results[algo]['params'], results[algo]['norm_params'] = fit_algo(
                plots_path_i, true_params, x, dims,
                algo_lambda=lambda x: test_EM(x, dims, train_results_dict[algo]['params']),
                algo_name=algo, prefix='test', override=override
            )
            print('test', algo, 'll', results[algo]['ll'][-1])

    return x, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", help="K", type=str)
    parser.add_argument("--cv", help="cv", type=str)
    args = parser.parse_args()

    K = int(args.K)
    cv = float(args.cv)

    T = 10

    N = 100
    M = 100
    m_lam = 2
    dims = Dims(N=N, K=K, M=M)
    true_dims = None

    plots_path = root_path / f'results/syn-data/unsupervised_dirichlet_start_repeats_M={M}_max_train=100_N={N}_mlam={m_lam}/K={K}/cv={str(cv)}'
    print(plots_path, flush=True)
    mkdir(plots_path, recursive=True)

    run_nmf = 1
    run_kt_nmf = 1
    run_em = 1

    run_algo_list = (['NMF'] if run_nmf else []) + \
                    ([r"NMF_tilde"] if run_kt_nmf else []) + \
                    (['EM'] if run_em else [])

    for i in range(T):

        print(i, flush=True)
        plots_path_i = plots_path/f'{i}'
        mkdir(plots_path_i)
        K_nmf = int(np.ceil((2*dims.M*dims.K+dims.N*dims.K)/(dims.M+dims.N)))
        x, train_results = train_all(
            plots_path_i, dims, cv=cv, m_lam=m_lam,
            run_algo_list=run_algo_list,
            K_nmf=K_nmf if run_kt_nmf else None,
            true_dims=true_dims
        )

        x, test_results = test_all(
            plots_path_i, dims, cv, m_lam, train_results,
            run_algo_list=run_algo_list, #override=True,
            K_nmf=int(np.ceil((2*dims.M*dims.K+dims.N*dims.K)/(dims.M+dims.N))) if run_kt_nmf else None,
            true_dims=true_dims
        )

