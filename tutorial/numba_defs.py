import numpy as np
from timeit import default_timer as _timer
from numba import njit, prange
from numba.extending import get_cython_function_address
import ctypes
EPS = 1e-16

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)


from tqdm import tqdm
disableTQDM = False


@njit(fastmath=True, parallel=True)
def numba_gammaln(x):
    s = x.shape
    x = x.flatten()
    out = np.empty(x.shape[0])
    for i in prange(x.shape[0]):
        out[i] = gammaln_float64(x[i])
    return out.reshape(s)


def nbinom_logpmf(k, loggamma_k, r, p):
    # k! = gamma(k + 1)
    # loggamma_k = loggamma(k + 1)
    return numba_gammaln(k + r) - loggamma_k - numba_gammaln(r) + r * np.log(1 - p) + k * np.log(p)

def nbinom_logpmf2(k, r, p):
    # k! = gamma(k + 1)
    return numba_gammaln(k + r) - numba_gammaln(k + 1) - numba_gammaln(r) + r * np.log(1 - p) + k * np.log(p)


def poisson_logpmf(k, l):
    # log_pmf(k|l) = k * log(l) - l - log(k!) = k * log(l) - l - loggamma(k + 1)
    return k * np.log(l) - l - numba_gammaln(k + 1)


def gamma_logpdf(x, a, b):
    # log_pdf(x|a,b) = a * log(b) - loggamma(a) + (a-1) * log(x) - b * x
    return a * np.log(b) - numba_gammaln(a) + (a-1) * np.log(x) - b * x


def logsumexp2(arr, axis=-1):
    if axis == -1:
        return lse0(arr)
    elif axis == 0:
        return lse1(arr)
    elif axis == 2:
        return lse2(arr)
    else:
        assert False


@njit(fastmath=True, parallel=False)
def lse0(arr):
    # no njit (original) = 23.1 µs ± 18.5 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # no parallel + no fastmath = 509 ns ± 1.28 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # no parallel + fastmath = 506 ns ± 0.201 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # parallel + fastmath = 12 µs ± 809 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    d = np.max(arr)
    if d == -np.inf: return d
    return d + np.log(np.sum(np.exp(arr - d)))


@njit(fastmath=True, parallel=False)
def lse1(arr):
    # original = 23.2 µs ± 800 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # no njit = 81 µs ± 135 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # no parallel + no fastmath = 1.17 µs ± 4.23 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # no parallel + fastmath = 1.17 µs ± 3.78 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # parallel + fastmath = 14.5 µs ± 653 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    res = np.empty(arr.shape[1])
    for j in prange(arr.shape[1]):
        d = np.max(arr[:, j])
        res[j] = d + np.log(np.sum(np.exp(arr[:, j] - d)))
    return res


@njit(fastmath=True, parallel=False)
def lse2(arr):
    # original = 32.2 µs ± 1.28 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # no njit = 107 µs ± 56.5 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # no parallel + no fastmath = 1.44 µs ± 1.64 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # no parallel + fastmath = 1.55 µs ± 4.56 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # parallel + fastmath = 13.4 µs ± 627 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    res = np.empty((arr.shape[0], arr.shape[1]))
    for i in prange(arr.shape[0]):
        for j in prange(arr.shape[1]):
            d = np.max(arr[i, j])
            res[i, j] = d + np.log(np.sum(np.exp(arr[i, j] - d)))
    return res


@njit(fastmath=True, parallel=False)
def lse2axis0(arr):
    # original = 32.2 µs ± 1.28 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # no njit = 107 µs ± 56.5 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # no parallel + no fastmath = 1.44 µs ± 1.64 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # no parallel + fastmath = 1.55 µs ± 4.56 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # parallel + fastmath = 13.4 µs ± 627 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    res = np.empty((arr.shape[1], arr.shape[2]))
    for i in prange(arr.shape[1]):
        for j in prange(arr.shape[2]):
            d = np.max(arr[:, i, j])
            res[i, j] = d + np.log(np.sum(np.exp(arr[:, i, j] - d)))
    return res

# def logsumexp2(arr, axis=-1):
#     res = 0
#     if axis == -1:
#         d = arr.max()
#         res = d + np.log(np.exp(arr - d).sum())
#     elif axis == 0:
#         d = arr.max(axis=axis)
#         res = d + np.log(np.sum(np.exp(arr - d), axis=axis))
#     else:
#         d = np.amax(arr, axis=axis, keepdims=True)
#         res = (d + np.log(np.sum(np.exp(arr - d), axis=axis, keepdims=True)))[:, :, 0]
#
#     return res


def timer(func):
    def wrapper(*args, **kwargs):
        start = _timer()
        rv = func(*args, **kwargs)
        end = _timer()
        print(f'{func.__name__} func time:\t{end - start:.2f}')
        return rv
    return wrapper


@njit
def sum_axis1(arr):
    res = np.zeros(arr.shape[0])
    for i in range(arr.shape[1]):
        res += arr[:, i]
    return res


@njit
def normalize_axis1(arr):
    res = np.zeros(arr.shape[0])
    for i in range(arr.shape[1]):
        res += arr[:, i]
    return arr / (res + EPS).reshape(*res.shape, 1)


@njit
def cumsum_axis1(arr):
    res = np.zeros(arr.shape)
    res[:, 0] = arr[:, 0]
    for i in range(1, arr.shape[1]):
        res[:, i] = res[:, i-1] + arr[:, i]
    return res



