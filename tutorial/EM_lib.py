import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba_defs import *
from scipy.special import digamma, loggamma
from scipy.optimize import bisect, linear_sum_assignment
from scipy.stats import rankdata, gamma
from collections import namedtuple
import sys
import pickle
import re
from pathlib import Path
import holoviews as hv
hv.extension('bokeh')
sns.set_style("white")
CLR = plt.rcParams['axes.prop_cycle'].by_key()['color'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
beta_losses = {'F': 'frobenius', 'KL': 'kullback-leibler'}

EPS = 1e-16
global PRECISION
PRECISION = np.float64

SAVE_FIG = False
SHOW_FIG = False
DEBUG = False


cmap = 'vlag'

root_path = Path(__file__).parent  # should always give "simulations"

Dims = namedtuple('Dims', ['N', 'M', 'K'])


class Thresholds:
    def __init__(self, eps_train=0, max_itr_train=100, eps_test=1e-6, max_itr_test=0):
        self.eps_train = eps_train
        self.max_itr_train = max_itr_train
        self.eps_test = eps_test
        self.max_itr_test = max_itr_test


SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def copy_na(x):
    if type(x) == int: return x
    elif x is None: return None
    return np.copy(x)


def concat_na(x, y, axis=0):
    if type(x) == int: return x
    elif x is None: return None
    return np.concatenate([np.copy(x), np.copy(y)], axis=axis)

get_gamma_ab = lambda m, v: (m*m/v, m/v)  # TODO
# get_gamma_mv = lambda a, b: (a/b, a/(b*b))


class Params:
    def __init__(self, W=None, a=None, b=None, H=None, lam=1, var=0, posterior_H=None):
        self.W = W
        self.a, self.b = a, b
        self.H = H
        self.lam, self.var = lam, var
        self.posterior_H = posterior_H

    def save_csv(self, dir_name, prefix=''):
        for k, v in self.__dict__.items():
            if v is not None and type(v) == np.ndarray and len(v.shape) < 3:
                pd.DataFrame.to_csv(pd.DataFrame(v), dir_name/(prefix + f'{k}.csv'), index=False)

    def copy(self, other):
        self.W = copy_na(other.W)
        self.a, self.b = copy_na(other.a), copy_na(other.b)
        self.H = copy_na(other.H)
        self.lam, self.var = copy_na(other.lam), other.var
        self.posterior_H = copy_na(other.posterior_H) if 'posterior_H' in other.__dict__.keys() else None
        return self

    @staticmethod
    def copy2(other):
        self = Params()
        self.W = copy_na(other.W)
        self.a, self.b = copy_na(other.a), copy_na(other.b)
        self.H = copy_na(other.H)
        self.lam, self.var = copy_na(other.lam), other.var
        self.posterior_H = copy_na(other.posterior_H) if 'posterior_H' in other.__dict__.keys() else None
        return self

    @staticmethod
    def concat_by_i(this, other):
        new = Params()
        new.W = concat_na(this.W, other.W, axis=0)
        new.a, this.b = copy_na(other.a), copy_na(other.b)
        new.H = copy_na(other.H)
        new.lam = concat_na(this.lam, other.lam)
        new.var = other.var
        new.posterior_H = concat_na(this.posterior_H, other.posterior_H, axis=0)\
            if 'posterior_H' in other.__dict__.keys() else None
        return new

    def reindex_by_i(self, new_idx):
        new = Params.copy2(self)
        new.W = new.W[new_idx] if new.W is not None else None
        new.lam = new.lam[new_idx] if ((new.lam is not None) and (type(new.lam) == np.ndarray)) else None
        new.posterior_H = new.posterior_H[new_idx] if \
            (('posterior_H' in new.__dict__.keys()) and (new.posterior_H is not None)) else None
        return new

    def reindex_by_k(self, new_idx):
        new = Params.copy2(self)
        new.W = new.W[:, new_idx] if new.W is not None else None
        new.H = new.H[:, new_idx] if new.H is not None else None
        if new.a is not None:
            new.a = new.a[:, new_idx]
            new.b = new.b[:, new_idx]

        if new.posterior_H is not None:
            new.posterior_H = new.posterior_H[:, :, new_idx]

        return new

    @staticmethod
    def load_params(params_path, param_name='params_obj'):
        if param_name is not None: params_path = params_path/f'{param_name}.pkl'
        with params_path.open('rb') as input:
            params = pickle.load(input)
        return params

    def dump_params(self, params_path, param_name='params_obj'):
        with (params_path/f'{param_name}.pkl').open('wb') as output:
            pickle.dump(self, output)


class VariationalParams:
    def __init__(self, log_p=None, alpha=None, beta=None):
        self.log_p = log_p
        self.alpha, self.beta = alpha, beta

    def copy(self, other):
        self.log_p = copy_na(other.log_p)
        self.alpha, self.beta = copy_na(other.alpha), copy_na(other.beta)
        return self


class ParamsProg:
    def __init__(self, W=None, a=None, b=None, H=None):
        self.W = W if W is not None else []
        self.a = a if a is not None else []
        self.b = b if b is not None else []
        self.H = H if H is not None else []

    def save_csv(self, dir_name, prefix=''):
        for k, v in self.__dict__.items():
            if v and len(v) > 0: pd.DataFrame.to_csv(pd.DataFrame(v[-1]), dir_name/(prefix + f'{k}.csv'), index=False)

    def __add__(self, other):
        oisParamsProg = type(other) == ParamsProg
        if other.W is not None and len(other.W) > 0: self.W += (other.W[1:] if oisParamsProg else [other.W])
        if other.a is not None and len(other.a) > 0: self.a += (other.a[1:] if oisParamsProg else [other.a])
        if other.b is not None and len(other.b) > 0: self.b += (other.b[1:] if oisParamsProg else [other.b])
        if other.H is not None and len(other.H) > 0: self.H += (other.H[1:] if oisParamsProg else [other.H])
        return self

    def get_param_idx(self, t):
        W = self.W[t] if self.W is not None and len(self.W) > t else None
        a = self.a[t] if self.a is not None and len(self.a) > t else None
        b = self.b[t] if self.b is not None and len(self.b) > t else None
        H = self.H[t] if self.H is not None and len(self.H) > t else None
        return Params(W=W, a=a, b=b, H=H)

    def last(self):
        return self.get_param_idx(-1)

    def copy(self, other):
        self.W, self.a, self.b, self.H = np.copy(other.W), np.copy(other.a), np.copy(other.b), np.copy(other.H)
        return self


class HyperParams:
    def __init__(self,
                 W=None, a=None, b=None, bg=0,
                 H=None, nmf_W=None, nmf_lam=None,
                 T1=1000, T0=0, ll_check=20,
                 ab_lims=None,
                 data=None, init_NMF='random', beta='KL', solver='mu',
                 new_k=0, const_start=True, new_ks_W=None, new_ks_AB=None,
                 inc_step=10,
                 n_CAVI_itr=10000,
                 features=None, samples=None,
                 priorH_params=None, priorH_N0=None
                 ):
        self.W, self.a, self.b = W, a, b  # EMkAB / EMkWmu (W) / data generator (predefined W)
        self.bg = bg  # EM
        self.H, self.nmf_W, self.nmf_lam = H, nmf_W, nmf_lam  # NMF/EM baseline (EMkH)
        self.T1, self.T0, self.ll_check = T1, T0, ll_check  # MCEM
        self.ab_lims = ab_lims  # EMlims
        self.data, self.init_NMF, self.beta, self.solver = data, init_NMF, beta, solver  # start params with NMF
        self.new_k, self.const_start = new_k, const_start  # EM1Comp
        self.new_ks_W, self.new_ks_AB = new_ks_W, new_ks_AB  # PartialEM
        self.inc_step = int(inc_step)  # iEM \ bEM
        self.n_CAVI_itr = int(n_CAVI_itr)  # vEM
        self.features, self.samples = features, samples
        self.priorH_params, self.priorH_N0 = priorH_params, priorH_N0  # adding priors on H

    def copy(self, other):
        self.W, self.a, self.b = copy_na(other.W), copy_na(other.a), copy_na(other.b)
        self.bg = other.bg if type(other.bg) == int else copy_na(other.bg)
        self.H, self.nmf_W, self.nmf_lam = copy_na(other.H), copy_na(other.nmf_W), copy_na(other.nmf_lam)
        self.T1, self.T0, self.ll_check = other.T1, other.T0, other.ll_check
        self.ab_lims = other.ab_lims
        self.data, self.init_NMF, self.beta, self.solver = copy_na(other.data), other.init_NMF, other.beta, other.solver
        self.inc_step = other.inc_step
        self.n_CAVI_itr = other.n_CAVI_itr
        self.new_k = other.new_k
        self.new_ks_W, self.new_ks_AB = other.new_ks_W, other.new_ks_AB # PartialEM
        self.priorH_params, self.priorH_N0 = other.priorH_params, other.priorH_N0
        if 'features' in other.__dict__.keys(): self.features, self.samples = other.features, other.samples  # old versions doesn't have this
        return self


class Statistics:
    def __init__(self, G=None, T=None, S_0=None, S_1=None, S_log=None):
        self.G, self.T, self.S_0, self.S_1, self.S_log = G, T, S_0, S_1, S_log

    def copy(self, other):
        self.G, self.T, self.S_0, self.S_1, self.S_log = np.copy(other.G), np.copy(other.T), other.S_0, np.copy(other.S_1), np.copy(other.S_log)
        return self

def mkdir(dir_path, recursive=False):
    if not dir_path.exists():
        dir_path.mkdir(parents=recursive)


def print_em_itr(cur_itr, ll, last_diff):
    print(f"\nt: {cur_itr}\t ll: {ll:.5f}\t diff: {last_diff:.10f}", flush=True)


def fit_gamma(S_0, S_1, S_log, K, M, arrM_AB=None, arrK_AB=None):
    # S_0 = num of data points
    # S_1 = sum over samples i of the data H[i] (or its expectation)
    # S_log = sum over samples i of the log data log(H[i]) (or its expectation)
    # K, M = dimension of each data point
    # arrM_AB, arrK_AB = optional, indices for easy access

    if arrM_AB is None or arrK_AB is None:
        temp = np.array([[(j, k) for k in range(K)] for j in range(M)])
        arrM_AB = temp.T[0].T
        arrK_AB = temp.T[1].T

    def f(x, S_1_kj, S_log_kj, S_0):
        return digamma(x) - np.log(x) + np.log(S_1_kj / S_0) - (S_log_kj / S_0)

    # get starting point
    x_0 = 0.5 * (1 / (np.log(S_1 / S_0) - (S_log / S_0)))  # stirling
    x_0[x_0 < EPS] = EPS
    assert np.all(x_0 > 0)

    # get lims
    x = np.linspace(0, x_0 * 100, 1000)[1:].transpose((1, 2, 0))
    y = f(x, S_1[:, :, None], S_log[:, :, None], S_0)
    switch = np.logical_and(y[:, :, :-1] < 0, y[:, :, 1:] > 0)

    idx1 = np.argmax(switch, axis=2)
    l_lim = x[arrM_AB, arrK_AB, idx1]
    temp = y[arrM_AB, arrK_AB, idx1]
    l_lim[temp > 0] = EPS  # TODO

    idx2 = np.argmax(switch, axis=2) + 1
    r_lim = x[arrM_AB, arrK_AB, idx2]

    # if DEBUG:
    temp = f(l_lim, S_1, S_log, S_0)
    assert np.all(temp < 0)
    temp2 = f(r_lim, S_1, S_log, S_0)
    assert np.all(temp2 > 0)

    # find root
    res = np.array([[bisect(f, a=l_lim[j][k], b=r_lim[j][k], args=(S_1[j][k], S_log[j][k], S_0,),
                            full_output=True, disp=False)
                     for k in range(S_1.shape[1])] for j in range(S_1.shape[0])])

    a = np.array([[res[j][k][0] for k in range(S_1.shape[1])] for j in range(S_1.shape[0])])

    # for j in range(S_1.shape[0]):
    #     for k in range(S_1.shape[1]):
    #         if not res[j][k][1].converged:
    #             print("a didn't converge:")
    #             print(j, k)
    #             print("----")
    #             assert False

    a[a < EPS] = EPS
    b = a * S_0 / S_1
    b[b < EPS] = EPS
    return a, b


def fdr(p_vals):
    # adjusts over genes https://stackoverflow.com/questions/25185205/calculating-adjusted-p-values-in-python
    ranked_p_values = rankdata(p_vals, axis=0)
    fdr = p_vals * p_vals.shape[1] / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr
