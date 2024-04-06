from Simulation_code.Simulation import *
from Simulation_code.DataGenerator import *
from pipeline_code.general_run import general_run


def boolean_string(s):  # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args_general(parser):

    #%%
    parser.add_argument("--train_algo", help="str, train algorithm. Options: EM, NMF, 2KNMF, HEN, MCMC, EMcW, EMkAB, vEM, etc.", type=str)
    parser.add_argument("--test_algo", help="str, test algorithm. Options: EMkAB, NMFkH, 2KNMF, HENkH, EMkH, vEMkAB, etc.", type=str, default=None, nargs='?')
    parser.add_argument("--run_test", help="bool = True if should run test", type=boolean_string, default=True, nargs='?')
    parser.add_argument("--sim_dir_name", help="str = sim_test + suffix", type=str, default="sim_test", nargs='?')
    #%%
    parser.add_argument("--N", help="int", type=int)
    parser.add_argument("--M", help="int", type=int)
    parser.add_argument("--K", help="int", type=int)
    parser.add_argument("--bg", help="bool", type=boolean_string)

    parser.add_argument("--var", help="str", type=str, default=None, nargs='?')
    parser.add_argument("--N_test", help="int", type=int, default=100, nargs='?')

    parser.add_argument("--test_data_filename", help="str = filename of specific test data", type=str, default='test', nargs='?')
    parser.add_argument("--train_data_filename", help="str = filename of specific train data", type=str, default='train', nargs='?')
    parser.add_argument("--background_filename", help="str = filename of background", type=str, default='background-filtered', nargs='?')

    # convergence params
    parser.add_argument("--eps_train", help="float", type=float, default=0, nargs='?')
    parser.add_argument("--eps_test", help="float", type=float, default=0, nargs='?')
    parser.add_argument("--max_itr_train", help="int", type=int, default=0, nargs='?')
    parser.add_argument("--max_itr_test", help="int", type=int, default=0, nargs='?')

    parser.add_argument("--run_name_T", help="int = number of repeats", type=int, default=1, nargs='?')

    # MCMC params
    parser.add_argument("--T0", help="int = start MCMC itr", type=int, default=0, nargs='?')
    parser.add_argument("--T1", help="int = end MCMC itr", type=int, default=1000, nargs='?')

    #EM1Comp param
    parser.add_argument("--new_k", help="int = comp to update", type=int, default=None, nargs='?')

    #PartialEM param
    parser.add_argument("--new_ks_W", help="list = comps to update for W, use = 2,3,4 with no space", type=str, default=None, nargs='?')
    parser.add_argument("--new_ks_AB", help="list = comps to update for AB, use = 2,3,4 with no space", type=str, default=None, nargs='?')

    # for predefined starting point
    parser.add_argument("--start_train_params_dir", help="str, dir for starting point of batch", type=str, default=None, nargs='?')
    parser.add_argument("--train_W_name", help="str = name of train W in start_train_params_dir", type=str, default='hat_W', nargs='?')
    parser.add_argument("--train_a_name", help="str = name of train a in start_train_params_dir", type=str, default='hat_a', nargs='?')
    parser.add_argument("--train_b_name", help="str = name of train b in start_train_params_dir", type=str, default='hat_b', nargs='?')
    parser.add_argument("--train_H_name", help="str = name of train H in start_train_params_dir", type=str, default='hat_H', nargs='?')
    parser.add_argument("--train_lam_name", help="str = name of train lam in start_train_params_dir", type=str, default='hat_lam', nargs='?')

    #

    return parser


def get_args_from_parser(parser):
    args = parser.parse_args()

    # parse lists https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    if args.new_ks_W is not None: args.new_ks_W = [int(item) for item in args.new_ks_W.split(',')]
    if args.new_ks_AB is not None: args.new_ks_AB = [int(item) for item in args.new_ks_AB.split(',')]

    return args
