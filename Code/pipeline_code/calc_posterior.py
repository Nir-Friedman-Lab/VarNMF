import os
import sys

sys.path.insert(1, os.getcwd())

from EM_code.batchEM import *

def calc_posterior(data, params, hyper, dims, calc_ll=False, calc_posterior=False, calc_pval=False):
    e_h, var_h, ll, pval = None, None, None, None

    EM_type = batchEMkAB if params.a is not None else batchEMkH
    em = EM_type(data, params, hyper, dims)

    if calc_posterior: e_h, var_h, ll = em.predict_H(return_var=True)
    elif calc_ll: ll = em.log_likelihood(return_mean=False)

    if calc_pval: pval = em.get_p_vals(data)

    return e_h, var_h, ll, pval


def _parse_args(parser):
    parser.add_argument("--N", help="int", type=int, default=0, nargs='?')
    parser.add_argument("--M", help="int", type=int, default=0, nargs='?')
    parser.add_argument("--K", help="int", type=int, default=0, nargs='?')

    parser.add_argument("--sim_dir", help="str, dir of simulation to calc stuff for", type=str, default=None, nargs='?')

    parser.add_argument("--params_dir", help="str, dir of params", type=str, default=None, nargs='?')
    parser.add_argument("--data_dir", help="str, dir of data", type=str, default=None, nargs='?')
    parser.add_argument("--bg_dir", help="str, dir of bg", type=str, default=None, nargs='?')

    parser.add_argument("--output_dir", help="str, dir for results", type=str)

    parser.add_argument("--calc_posterior", help="bool", type=boolean_string, default=True, nargs='?')
    parser.add_argument("--calc_ll", help="bool", type=boolean_string, default=True, nargs='?')
    parser.add_argument("--calc_pval", help="bool", type=boolean_string, default=True, nargs='?')

    return parser
