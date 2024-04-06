from lib_code.EM_lib import *


def sample_gamma(a, b, dims):
    # for each k,j, sample N times.txt h_kj ~ Gamma(a_kj, b_kj) independent.
    return np.random.gamma(a, 1 / b, dims)


def sample_sum_poi(w, H, dims, lam=1, bg=0.0):
    # for each i, sample Y_i1,...,Y_i,K in R^M, Y_ikj ~ Po(lam*w[i]_k*H[i]_k,j) and v_i ~ sum(Y_i1,...,Y_i,K).
    Y = np.random.poisson(np.multiply(lam * w, H.transpose(1, 0, 2)).transpose(1, 0, 2))
    noise_val = np.random.poisson(bg, (dims.N, dims.M)) if type(bg) == int else np.random.poisson(bg)
    V = np.sum(Y, axis=2) + noise_val
    return V, Y


def sample_sum_nb(a, b, w, dims, r=1):
    # for each i, sample Y_i1,...,Y_i,K in R^M, Y_ikj ~ NB(A_kj, lam*w[i]_k / lam*w[i]_k + B_kj) and v_i ~ sum(Y_i1,...,Y_i,K).
    nb = np.array([np.random.negative_binomial(a[j, :], 1 - (r * w / (r * w + b[j, :]))) for j in range(dims.M)]).transpose((1, 0, 2))
    V = np.sum(nb, axis=2)
    return V, nb

