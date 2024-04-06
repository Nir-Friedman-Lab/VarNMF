import os
import sys
sys.path.insert(1, os.getcwd())

from pipeline_code.general_setup import *
from pipeline_code.general_run import sim_types, train_types, test_types

def _get_data(train_data_dir, test_data_dir, background_dir,
              train_data_name='train', test_data_name='test', bg_name='background-filtered',
              setup_train=True, setup_test=True, setup_bg=True):
    train_data, test_data, S_train, S_test, genes = None, None, None, None, None
    if setup_bg: S = pd.read_csv(background_dir / f"{bg_name}.csv", index_col=0)

    train_data = pd.read_csv(train_data_dir / (train_data_name + ".csv"), index_col=0)
    train_samples = train_data.columns
    genes = train_data.index
    if setup_train:
        if setup_bg: S_train = S.loc[genes][train_samples]

    if setup_test:
        test_data = pd.read_csv(test_data_dir / (test_data_name + ".csv"), index_col=0)
        test_data = test_data.loc[[g for g in genes if g in test_data.index]]
        test_samples = test_data.columns
        if setup_bg: S_test = S.loc[test_data.index][test_samples]

    return train_data, test_data, S_train, S_test


def read_csv(csv_path, K):
    res = pd.read_csv(csv_path)
    if K not in res.shape:
        res = res.set_index(res.columns[0])
    return res.values


def _get_known_params(train_data_dir, run_known_train, K):
    true_params = Params()
    if run_known_train:
        a, b, H = None, None, None
        if (train_data_dir/'a.csv').exists(): a = read_csv(train_data_dir/'a.csv', K)
        if (train_data_dir/'b.csv').exists(): b = read_csv(train_data_dir/'b.csv', K)
        if (train_data_dir/'H.csv').exists(): H = read_csv(train_data_dir/'H.csv', K)
        else:
            if a is not None:
                H = a / b
                H[H < EPS] = EPS
        true_params = Params(a=a, b=b, H=H, W=None)
    return true_params


def _get_start_train_params(start_train_params_dir, K, M, batch_t=-1, W_name='', a_name='', b_name='', H_name='', lam_name=''):
    params = Params()

    if start_train_params_dir is not None:
        if (root_path/start_train_params_dir / f"{W_name}.csv").exists():
            params.W = read_csv(root_path/start_train_params_dir / f"{W_name}.csv", K)
        if (root_path/start_train_params_dir / f"{lam_name}.csv").exists():
            params.lam = read_csv(root_path/start_train_params_dir / f"{lam_name}.csv", K)
        if (root_path/start_train_params_dir / f"{a_name}.csv").exists():
            params.a = read_csv(root_path/start_train_params_dir / f"{a_name}.csv", K)
            if batch_t >= 0: params.a = params.a[batch_t * M:(batch_t + 1) * M]
        if (root_path/start_train_params_dir / f"{b_name}.csv").exists():
            params.b = read_csv(root_path/start_train_params_dir / f"{b_name}.csv", K)
            if batch_t >= 0: params.b = params.b[batch_t * M:(batch_t + 1) * M]
        if (root_path/start_train_params_dir / f"{H_name}.csv").exists():
            params.H = read_csv(root_path/start_train_params_dir / f"{H_name}.csv", K).T
            if batch_t >= 0: params.H = params.H[:, batch_t * M:(batch_t + 1) * M]

    return params


def _create_data_obj(output_dir,
                     train_data_hyper, test_data_hyper,
                     train_dims=None, test_dims=None,
                     train_true_params=Params(), test_true_params=Params(),
                     run_name='', test_suffix='', setup_train=True, setup_test=True):
    train_data_obj, test_data_obj = None, None
    if setup_train: train_data_obj = DataInitDataGenerator(
        output_dir / run_name / 'train' / 'data', true_params=train_true_params,
        hyper_params=train_data_hyper, dims=train_dims
    ).generate()
    if setup_test: test_data_obj = DataInitDataGenerator(
        output_dir / ('test' + test_suffix) / 'data', true_params=test_true_params,
        hyper_params=test_data_hyper, dims=test_dims
    ).generate()
    return train_data_obj, test_data_obj


def _setup_data_obj(train_data, test_data, output_dir_itr, K, start_train_params,
                    S_train=None, S_test=None, run_train=True, run_test=True, run_name='', test_suffix='',
                    train_true_params=Params(),
                    start_NMF_init='random', start_beta='KL', start_solver='mu',
                    new_k=None, new_ks_W=None, new_ks_AB=None):
    train_dims, test_dims, train_data_hyper, test_data_hyper = None, None, None, None
    if run_train:
        train_dims = Dims(N=train_data.shape[1], M=train_data.shape[0], K=K)
        train_data_hyper = HyperParams(
            data=train_data.values.T, bg=S_train, new_k=new_k, new_ks_W=new_ks_W, new_ks_AB=new_ks_AB,
            init_NMF=start_NMF_init, beta=start_beta, solver=start_solver,
            W=start_train_params.W, H=start_train_params.H, a=start_train_params.a, b=start_train_params.b,
            features=train_data.index, samples=train_data.columns
        )
    if run_test:
        test_dims = Dims(N=test_data.shape[1], M=test_data.shape[0], K=K)
        test_data_hyper = HyperParams(
            data=test_data.values.T, bg=S_test, new_k=new_k, new_ks_W=new_ks_W, new_ks_AB=new_ks_AB,
            features=test_data.index, samples=test_data.columns
        )

    train_data_obj, test_data_obj = _create_data_obj(
        output_dir_itr, train_data_hyper, test_data_hyper, train_dims=train_dims, test_dims=test_dims,
        train_true_params=train_true_params,
        run_name=run_name, test_suffix=test_suffix, setup_train=run_train, setup_test=run_test
    )
    return train_data_obj, test_data_obj


def run_data(output_dir, N, M, K, N_test, train_algo, test_algo, sub_dir_name, bg,
             train_data_dir, test_data_dir, background_dir,
             train_W_name, train_a_name, train_b_name, train_H_name, train_lam_name,
             train_data_name='train', test_data_name='test', bg_name='background-filtered',
             run_train=True, run_normal_test=True, run_known_train=False, run_special_test=False,
             thresholds=None, start_train_params_dir=None, const_train_result_dir=None,
             start_NMF_init='random', start_beta='KL', start_solver='mu', new_k=None, new_ks_W=None, new_ks_AB=None,
             batch_t=-1, train_batchW=False,
             run_name='', test_suffix='', sub_dir_name_test=''):

    print('\noutput_dir =', output_dir,
          '\ndims =', Dims(N, M, K),
          '\nN_test =', N_test,
          '\ntrain_algo =', train_algo,
          '\ntest_algo =', test_algo,
          '\nsub_dir_name =', sub_dir_name,
          '\nbg =', bg,
          '\nthresholds =', thresholds,
          flush=True)
    # a\b\H.shape = (), W.shape = (N,K)

    run_test = run_normal_test or run_special_test
    output_dir_itr = output_dir / f'bg={1 if bg else 0}_K={K}'

    # Data setup
    ## get data
    train_data, test_data, S_train, S_test = _get_data(
        train_data_dir, test_data_dir, background_dir, train_data_name, test_data_name, bg_name,
        setup_train=run_train, setup_test=run_test, setup_bg=bg
    )
    # get kAB or kH if required
    train_true_params = _get_known_params(
        train_data_dir, run_known_train=run_known_train, K=K
    )
    ## create data objects
    start_train_params = _get_start_train_params(
        start_train_params_dir, K, M, batch_t=batch_t,
        W_name=train_W_name, a_name=train_a_name, b_name=train_b_name, H_name=train_H_name, lam_name=train_lam_name
    )
    ## create data objects
    train_data_obj, test_data_obj = _setup_data_obj(
        train_data, test_data, output_dir_itr, K, start_train_params,
        S_train=S_train.values.T if (bg and S_train is not None) else 0, S_test=S_test.values.T if (bg and S_test is not None) else 0,
        run_train=run_train, run_test=run_test, run_name=run_name, test_suffix=test_suffix, train_true_params=train_true_params,
        start_NMF_init=start_NMF_init, start_beta=start_beta, start_solver=start_solver,
        new_k=new_k, new_ks_W=new_ks_W, new_ks_AB=new_ks_AB
    )

    if N > 0 and run_train: print('\ntrain max:\t', train_data_obj.data.max())
    if N_test > 0 and run_test: print('\ntest max:\t', test_data_obj.data.max())

    # Run
    general_run(
        output_dir_itr, train_algo, test_algo, sub_dir_name, train_data_obj, test_data_obj, K,
        run_train=run_train, run_test=run_test,
        test_suffix=test_suffix, sub_dir_name_test=sub_dir_name_test, run_name=run_name,
        thresholds=thresholds,
        init_params=(batch_t >= 0) or train_batchW or run_special_test or (start_train_params_dir is not None),
        init_H_params=run_known_train,
        const_train_result_dir=const_train_result_dir
    )


def _parse_args(parser):
    parser = parse_args_general(parser)

    # directories
    parser.add_argument("--output_dir", help="str = output_dir for the simulation. Will be under root/EM_runs", type=str)
    parser.add_argument("--data_dir", help="str = dir of split data. default is output_dir", type=str, default=None, nargs='?')
    parser.add_argument("--bg_dir", help="str = dir of background. default is data_dir or test_data_dir", type=str, default=None, nargs='?')

    # for specific test data
    parser.add_argument("--test_data_dir", help="str = dir of specific test data. If not specified will run regular train-test", type=str, default=None, nargs='?')
    parser.add_argument("--const_train_result_dir", help="str = dir of train results. Will also be the output dir if specified", type=str, default=None, nargs='?')

    return parser


def run_data_simulation(args):

    flag_real_data = not args.test_data_dir
    flag_run_data_test = args.test_data_dir

    thresholds = Thresholds(eps_train=args.eps_train, eps_test=args.eps_test,
                            max_itr_train=args.max_itr_train, max_itr_test=args.max_itr_test)

    train_algo = args.train_algo
    run_test = args.run_test
    test_algo = args.test_algo
    if test_algo is None:
        test_algo = f'{train_algo}kAB' if ('EM' in train_algo) else f'{train_algo}kH'
    if 'MCEM' in train_algo:
        temp = f'_T1={args.T1}_T0={args.T0}'
        sim_types[train_algo + temp] = sim_types[train_algo]
        train_types[train_algo + temp] = train_types[train_algo]
        # test_types[train_algo + temp] = test_types[train_algo]
        test_types[test_algo + temp] = test_types[test_algo]
        train_algo += temp
        test_algo += temp

    output_dir_sim = root_path / args.output_dir
    train_data_dir = output_dir_sim if args.data_dir is None else root_path / args.data_dir
    test_data_dir = train_data_dir
    if args.test_data_dir:
        test_data_dir = root_path / 'data' / args.test_data_dir
        const_train_result_dir = root_path / args.const_train_result_dir
        args.test_data_filename = 'counts-filtered'
    else: const_train_result_dir = None

    bg_dir = train_data_dir if args.bg_dir is None else root_path / args.bg_dir
    if args.test_data_dir: bg_dir = test_data_dir

    #######
    # # run data
    if flag_real_data:
        for run_name in range(args.run_name_T):
            run_data(
                output_dir_sim, args.N, args.M, args.K, args.N_test,
                train_algo, test_algo, args.sim_dir_name, args.bg,
                train_data_dir, test_data_dir, bg_dir,
                args.train_W_name, args.train_a_name, args.train_b_name, args.train_H_name, args.train_lam_name,
                train_data_name=args.train_data_filename,
                test_data_name=args.test_data_filename,
                bg_name=args.background_filename,
                start_train_params_dir=args.start_train_params_dir,
                thresholds=thresholds, #batch_t=args.batch_t, train_batchW=args.trainW,
                run_normal_test=run_test, run_known_train=('kH' in train_algo) or ('kAB' in train_algo),
                run_name=str(run_name), new_k=args.new_k, new_ks_W=args.new_ks_W, new_ks_AB=args.new_ks_AB
            )
            print(f'end {run_name}', flush=True)

    if flag_run_data_test:
        T = 1
        for run_name in range(T):
            run_data(
                output_dir_sim, args.N, args.M, args.K, args.N_test,
                train_algo, test_algo, args.sim_dir_name, args.bg,
                args.train_W_name, args.train_a_name, args.train_b_name, args.train_H_name, args.train_lam_name,
                train_data_dir, test_data_dir, bg_dir,
                test_data_name=args.test_data_filename,
                bg_name=args.background_filename,
                start_test_params_dir=args.start_test_params_dir,
                thresholds=thresholds, run_train=False, run_normal_test=False, run_special_test=True,
                const_train_result_dir=const_train_result_dir, run_name=train_algo  # TODO
            )

            print(f'end {run_name}', flush=True)


if __name__ == '__main__':

    print(sys.argv, flush=True)
    parser = argparse.ArgumentParser()
    parser = _parse_args(parser)
    args = parser.parse_args()
    run_data_simulation(args)

