import os
import sys
sys.path.insert(1, os.getcwd())

from pipeline_code.general_setup import *
from pipeline_code.batch_simulation import run_data_simulation
from pipeline_code.setup_data_run import _select_genes, split_train_test
from Simulation_code.Simulation import load_simulation
import copy


def create_NMF_start(N, M, K, bg, split_data_dir, start_train_params_dir, bg_dir):
    # do NMF in itr_name/start to get staring point
    print('create NMF start')
    import data_simulation

    data_simulation.run_data(
        root_path/start_train_params_dir, N, M, K, N_test=0,
        train_algo='NMF', test_algo=None, sub_dir_name='sim_test', bg=bg,
        train_data_dir=root_path/split_data_dir, test_data_dir=None, background_dir=root_path/bg_dir,
        run_train=True, run_normal_test=False, run_special_test=False, run_name='', thresholds=Thresholds()
    )


def _select_batch_genes(data, M, batch_t):
    # get M batch of genes
    genes = data.index[batch_t * M:(batch_t + 1) * M]
    return genes


def _read_split_data(split_data_dir, train_data_name='train', test_data_name='test', read_train=True, read_test=True):
    train_data, test_data = None, None
    if read_train: train_data = pd.read_csv(root_path/split_data_dir/f"{train_data_name}.csv", index_col=0)
    if read_test: test_data = pd.read_csv(root_path/split_data_dir/f"{test_data_name}.csv", index_col=0)
    return train_data, test_data


def _split_to_batches(train_data, test_data, data_dir_saveto, M, n_batches,
                      train_data_name='train', test_data_name='test', create_test=False):
    for batch_t in range(n_batches):
        batch_dir = root_path / data_dir_saveto / f'{batch_t}/'
        if not batch_dir.exists():
            batch_dir.mkdir(parents=True)
            batch_genes = _select_batch_genes(train_data, M, batch_t)
            train_data.loc[batch_genes].to_csv(batch_dir/f"{train_data_name}.csv")
            if create_test: test_data.loc[batch_genes].to_csv(batch_dir/f"{test_data_name}.csv")


def split_data_trainAB(data_dir_saveto, split_data_dir, M, n_batches, create_test=False):
    # split train, test and start to batches in itr_name/trainAB/batch_t
    train_data, test_data = _read_split_data(split_data_dir, read_test=create_test)
    _split_to_batches(train_data, test_data, data_dir_saveto, M, n_batches, create_test=create_test)


def trainAB(args):
    args.train_algo += 'kW'
    args.run_test = False
    run_data_simulation(args)


def _concat_batch_params(target, other):
    # add other to target
    for k in ['a', 'b', 'H']:
        if other.__dict__[k] is not None and target.__dict__[k] is not None:
            target.__dict__[k] = np.concatenate([target.__dict__[k], other.__dict__[k]])
    return target


def arrange_trainAB_results(batch_results_dir, final_results_dir, n_batches, run_name=0, train_algo='EMkW'):
    print('arrange trainAB results')
    # arrange results in itr_name/trainAB/res (output_dir=itr_name/trainAB)
    if not (final_results_dir/'hat.pkl').exists():
        hat = None
        for batch_t in range(n_batches):
            batch_dir = root_path/batch_results_dir/f'{batch_t}/'
            batch_sim_dir = batch_dir/f'bg={1 if args.bg else 0}_K={args.K}/{run_name}/train/sim_test/{train_algo}/'
            assert (batch_sim_dir / 'simulation_obj.pkl').exists(), f'{batch_t} simulation does not exists'
            simulator = load_simulation(batch_sim_dir)
            batch_hat = simulator.hat
            if hat is None: hat = batch_hat
            else: hat = _concat_batch_params(hat, batch_hat)

        if not final_results_dir.exists(): final_results_dir.mkdir(parents=True)
        hat.H = hat.a / hat.b
        hat.H[hat.a == EPS] = EPS
        hat.save_csv(final_results_dir, prefix='hat_')
        hat.dump_params(final_results_dir, param_name='hat')
    else:
        hat = Params.load_params(final_results_dir, param_name='hat')
    return hat


def run_AB(args, start_train_params_dir_AB):
    print(f'run AB {args.batch_t}')
    if args.create_NMF_start:
        # do NMF in itr_name/start to get staring point
        if not (root_path/start_train_params_dir_AB).exists():
            create_NMF_start(args.N, args.M * args.n_batches, args.K, args.bg, split_data_dir=args.start_train_params_dir,
                             start_train_params_dir=args.start_train_params_dir, bg_dir=f'data/{args.dir_name}')


    output_dir_AB = args.output_dir + f'trainAB/'
    if args.split_data_trainAB:
        # split train, test and start to batches in itr_name/trainAB/batch_t
        if not (root_path/output_dir_AB/f'{args.batch_t}/').exists():
            split_data_trainAB(data_dir_saveto=output_dir_AB, split_data_dir=args.data_dir,  #  TODO was args.start_train_params_dir
                               M=args.M, n_batches=args.n_batches)

    if args.trainAB and args.batch_t >= 0:
        # run batches (trainAB) in itr_name/trainAB/batch_t
        args_copy = copy.deepcopy(args)
        args_copy.output_dir = output_dir_AB + f'{args.batch_t}/'
        args_copy.data_dir = args_copy.output_dir
        args_copy.start_train_params_dir = start_train_params_dir_AB
        trainAB(args_copy)


def _get_params_trainW(data_dir_saveto, idx_chosen_genes, hat):
    trainW_params = Params().copy(hat)
    for k in ['H', 'a', 'b']:
        if trainW_params.__dict__[k] is not None:
            trainW_params.__dict__[k] = trainW_params.__dict__[k][idx_chosen_genes]
    trainW_params.save_csv(data_dir_saveto, prefix='hat_')
    trainW_params.dump_params(data_dir_saveto, param_name='hat')


# def _choose_genes(hat, genes, n_genes, K):
#     def second(E_gamma):
#         return np.array([(E_gamma[:, k] - E_gamma[:, [c for c in range(K) if c != k]].max(axis=-1)) for k in range(K)])
#     scores = second(hat.a / (hat.b + EPS)).max(axis=0)
#     idx_chosen_genes = scores.argsort()[::-1][:n_genes]
#     genes = genes[idx_chosen_genes]
#     return idx_chosen_genes, genes


def _choose_genes(hat, genes, n_genes, K, trh=2, val_trh=10):
    # logfc > 2 and val > 10
    n_genes = n_genes // K

    H = hat.a / (hat.b + EPS)
    H = H / (H.sum(axis=0) / H.sum(axis=0)[0])
    median_K = np.median(H, axis=1)[:, None]

    if K == 1:
        diff_genes_idx_n_genes = np.argsort(median_K[:, 0])[::-1][:n_genes]
        diff_genes_n_genes = genes[diff_genes_idx_n_genes]
        return diff_genes_idx_n_genes, diff_genes_n_genes

    if K > 2: logfc = pd.DataFrame(np.log2(1 + H) - np.log2(1 + median_K), index=genes)
    else: logfc = pd.DataFrame(np.log2(1 + H) - np.log2(1 + H[:, ::-1]), index=genes)

    diff_genes_n_genes = []
    diff_genes_idx_n_genes = []
    H_df = pd.DataFrame(H, index=genes)
    for k in range(K):
        cond = (logfc[k].values > trh) & (H_df[k].values > val_trh)
        if cond.sum() <= 5: cond = logfc[k].values > trh
        if cond.sum() <= 5: cond = logfc[k].values > 1
        diff_genes_idx_n_genes.append(np.arange(logfc.shape[0])[cond][np.argsort(logfc[k].loc[cond].values)[::-1][:n_genes]])
        diff_genes_n_genes.append(logfc.loc[cond].index[np.argsort(logfc[k].loc[cond].values)[::-1][:n_genes]])
        assert len(diff_genes_idx_n_genes[-1]) > 0

    diff_genes_idx_n_genes = np.unique(np.concatenate(diff_genes_idx_n_genes))
    diff_genes_n_genes = genes[diff_genes_idx_n_genes]

    return diff_genes_idx_n_genes, diff_genes_n_genes

def get_data_trainW(data_dir_saveto, split_data_dir, AB_hat_dir, n_genes, K,
                      train_data_name='train', test_data_name='test', get_test=True):
    print('split data trainW')
    train_data, test_data = _read_split_data(split_data_dir, read_test=get_test)
    genes = train_data.index
    hat = Params.load_params(AB_hat_dir, param_name='hat')

    idx_chosen_genes, genes = _choose_genes(hat, genes, n_genes, K)

    # plot chosen genes of k=0, k=1 for debug  TODO
    from plots_code.general_plot_run import scatter, show
    H = hat.a / (hat.b + EPS)
    H = H / (H.sum(axis=0) / H.sum(axis=0)[0])
    for t1 in range(K):
        for t2 in range(K):
            if t1 < t2:
                scatter(H[idx_chosen_genes, t1], H[idx_chosen_genes, t2], log=True)
                show(aspect=True, title=f't1={t1},t2={t2}')


    if not (root_path / data_dir_saveto).exists():
        (root_path / data_dir_saveto).mkdir(parents=True)
        train_data.loc[genes].to_csv(root_path / data_dir_saveto/f"{train_data_name}.csv")
        if get_test: test_data.loc[genes].to_csv(root_path / data_dir_saveto/f"{test_data_name}.csv")

    _get_params_trainW(root_path / data_dir_saveto, idx_chosen_genes, hat)


def trainW(args):
    print('trainW')
    args.train_algo += 'kAB'
    # args.run_test = False
    args.batch_t = -1
    run_data_simulation(args)


def _get_trainW_hat(trainW_results_dir, run_name, bg, K, trainW_algo_name='EM'):
    sim_dir = root_path / trainW_results_dir / f'bg={1 if bg else 0}_K={K}/{run_name}/train/sim_test/{trainW_algo_name}kAB/'
    simulator = load_simulation(sim_dir)
    W = simulator.hat.W
    return W


def _get_testW_hat(testW_results_dir, run_name, bg, K, trainW_algo_name='EM', testW_algo_name='EM'):
    sim_dir = root_path / testW_results_dir / f'bg={1 if bg else 0}_K={K}/{run_name}/test/sim_test/{trainW_algo_name}_{testW_algo_name}kAB/'
    simulator = load_simulation(sim_dir)
    W = simulator.hat.W
    return W

def arrange_trainW_results(trainW_simulation_dir, final_results_dir, AB_hat_dir, split_data_dir, bg, K, run_name=0,
                      train_data_name='train', test_data_name='test', trainW_algo_name='EM', get_test=True):
    print('arrange trainW results')
    hat, test_hat = None, None
    # arrange results in itr_name/trainW/algo_name/res (output_dir=itr_name/trainW/algo_name)
    if not (root_path/final_results_dir/'hat.pkl').exists():
        if not (root_path/final_results_dir).exists(): (root_path/final_results_dir).mkdir(parents=True)
        train_data, test_data = _read_split_data(split_data_dir, read_test=get_test)
        train_data.to_csv(root_path/final_results_dir/f"{train_data_name}.csv")
        if get_test: test_data.to_csv(root_path/final_results_dir/f"{test_data_name}.csv")

        hat = Params()
        hat.W = _get_trainW_hat(trainW_simulation_dir, run_name, bg, K, trainW_algo_name)

        AB_hat = Params.load_params(AB_hat_dir, param_name='hat')
        hat.H, hat.a, hat.b = AB_hat.H, AB_hat.a, AB_hat.b

        if hat.H is None:
            hat.H = hat.a / hat.b
            hat.H[hat.a == EPS] = EPS
        hat.save_csv(root_path/final_results_dir, prefix='hat_')
        hat.dump_params(root_path/final_results_dir, param_name='hat')
    else:
        hat = Params.load_params(root_path/final_results_dir, param_name='hat')

    if get_test:
        if not (root_path/final_results_dir/'test_hat.pkl').exists():
            test_hat = Params()
            test_hat.W = _get_testW_hat(trainW_simulation_dir, run_name, bg, K, trainW_algo_name+'kAB', trainW_algo_name)
            AB_hat = Params.load_params(AB_hat_dir, param_name='hat')
            test_hat.H, test_hat.a, test_hat.b = AB_hat.H, AB_hat.a, AB_hat.b

            if test_hat.H is None:
                test_hat.H = test_hat.a / test_hat.b
                test_hat.H[test_hat.a == EPS] = EPS
            test_hat.save_csv(root_path/final_results_dir, prefix='test_hat_')
            test_hat.dump_params(root_path/final_results_dir, param_name='test_hat')
        else:
            test_hat = Params.load_params(root_path/final_results_dir, param_name='test_hat')
    return hat, test_hat


def run_W(args, trainAB_results_dir, get_test=True):
    output_dir_W = args.output_dir + f'trainW/{args.train_algo}/'
    if args.split_data_trainW and not (root_path/(output_dir_W + f'bg={1 if args.bg else 0}_K={args.K}/train.csv')).exists():
        # get specific genes and start params to itr_name/trainW/algo_name/bg=1_K={K}
        get_data_trainW(output_dir_W + f'bg={1 if args.bg else 0}_K={args.K}/',
                          split_data_dir=args.data_dir, # TODO was args.start_train_params_dir,
                          AB_hat_dir=trainAB_results_dir, n_genes=args.trainW_n_genes, K=args.K, get_test=get_test)

    if args.trainW:
        # run trainW in itr_name/trainW/algo_name
        args_copy = copy.deepcopy(args)
        args_copy.output_dir = output_dir_W
        args_copy.data_dir = args_copy.output_dir + f'bg={1 if args.bg else 0}_K={args.K}/'
        args_copy.start_train_params_dir = args_copy.output_dir + f'bg={1 if args.bg else 0}_K={args.K}/'
        args_copy.M = args.trainW_n_genes
        trainW(args_copy)

    trainW_results_dir = (root_path / output_dir_W).parent.parent / f'res_trainW/bg={1 if args.bg else 0}_K={args.K}/'
    if args.arrange_trainW_results:
        # arrange results in itr_name/trainW/algo_name/res -> will be the start_train_params_dir of the next iteration
        arrange_trainW_results(trainW_simulation_dir=output_dir_W, final_results_dir=trainW_results_dir,
                               AB_hat_dir=trainAB_results_dir, split_data_dir= args.data_dir, # TODO was args.start_train_params_dir,
                               bg=args.bg, K=args.K, trainW_algo_name=args.train_algo, get_test=get_test)
    return output_dir_W, trainW_results_dir


def get_data_test(data_dir_saveto, test_data_dir, AB_hat_dir, n_genes, K,
                  train_data_name='train', test_data_name='test'):
    print('gat data test')
    _, test_data = _read_split_data(test_data_dir, test_data_name=test_data_name, read_train=False)
    train_data, _ = _read_split_data(test_data_dir, train_data_name=train_data_name, read_test=False)
    genes = train_data.index
    hat = Params.load_params(AB_hat_dir, param_name='hat')

    idx_chosen_genes, genes = _choose_genes(hat, genes, n_genes, K)

    if not (root_path / data_dir_saveto).exists():
        (root_path / data_dir_saveto).mkdir(parents=True)
        train_data.loc[genes].to_csv(root_path / data_dir_saveto/f"{train_data_name}.csv")
        test_data.loc[genes].to_csv(root_path / data_dir_saveto/f"{test_data_name}.csv")

    _get_params_trainW(root_path / data_dir_saveto, idx_chosen_genes, hat)


def testW(args):
    print('testW')
    args.run_test = True
    args.run_train = False
    args.batch_t = -1
    run_data_simulation(args)


def run_test(args, trainW_simulation_dir, trainW_results_dir):
    if args.test_algo is None: args.test_algo = f'{args.train_algo}kAB' if ('EM' in args.train_algo) else f'{args.train_algo}kH'
    output_dir_test = args.output_dir + f'test-{args.test_data_dir}/'
    args.test_data_dir = args.test_data_dir

    # get specific genes and start params to itr_name/test_name/algo_name
    get_data_test(output_dir_test + f'bg={1 if args.bg else 0}_K={args.K}', test_data_dir='data/' + args.test_data_dir,
                  AB_hat_dir=trainW_results_dir, n_genes=args.trainW_n_genes,
                  test_data_name=args.test_data_filename, K=args.K)

    # run test in itr_name/test_name/algo_name
    args_copy = copy.deepcopy(args)
    args_copy.output_dir = root_path/output_dir_test
    args_copy.data_dir = args_copy.output_dir/f'bg={1 if args.bg else 0}_K={args.K}'
    args_copy.M = args_copy.trainW_n_genes
    args_copy.const_train_result_dir = root_path/trainW_simulation_dir / f'bg={1 if args_copy.bg else 0}_K={args_copy.K}/0/train/sim_test/EMkAB/'
    testW(args_copy)

    testW_results_dir = root_path / output_dir_test / f'res/bg={1 if args.bg else 0}_K={args.K}/'
    if args.arrange_trainW_results:
        # arrange results in itr_name/trainW/algo_name/res -> will be the start_train_params_dir of the next iteration
        arrange_trainW_results(trainW_simulation_dir=output_dir_test, final_results_dir=testW_results_dir,
                               AB_hat_dir=trainW_results_dir, split_data_dir=args.start_train_params_dir,
                               bg=args.bg, K=args.K, trainW_algo_name=args.train_algo)

def run_test_after_trainAB(args, trainAB_simulation_dir, trainAB_results_dir):
    if args.test_algo is None: args.test_algo = f'{args.train_algo}kAB' if ('EM' in args.train_algo) else f'{args.train_algo}kH'
    output_dir_test = args.output_dir + f'test-{args.test_data_dir}/'
    args.test_data_dir = args.test_data_dir

    # get specific genes and start params to itr_name/test_name/algo_name
    get_data_test(output_dir_test + f'bg={1 if args.bg else 0}_K={args.K}', test_data_dir=args.data_dir,
                  AB_hat_dir=trainAB_results_dir, n_genes=args.test_n_genes,
                  test_data_name=args.test_data_filename, K=args.K)

    # run test in itr_name/test_name/algo_name
    args_copy = copy.deepcopy(args)
    args_copy.output_dir = root_path/output_dir_test
    args_copy.data_dir = args_copy.output_dir/f'bg={1 if args.bg else 0}_K={args.K}'
    args_copy.M = args_copy.test_n_genes
    args_copy.const_train_result_dir = trainAB_results_dir
    testW(args_copy)

    testAB_results_dir = root_path / output_dir_test / f'res/bg={1 if args.bg else 0}_K={args.K}/'
    if args.arrange_trainW_results:
        # arrange results in itr_name/trainW/algo_name/res -> will be the start_train_params_dir of the next iteration
        arrange_trainW_results(trainW_simulation_dir=output_dir_test, final_results_dir=testAB_results_dir,
                               AB_hat_dir=trainAB_simulation_dir, split_data_dir=args.start_train_params_dir,
                               bg=args.bg, K=args.K, trainW_algo_name=args.train_algo)


def _parse_args(parser):
    parser = parse_args_general(parser)

    # directories
    parser.add_argument("--output_dir", help="str = output_dir for the simulation ", type=str, default=None, nargs='?')
    parser.add_argument("--data_dir", help="str = dir of split data. default is output_dir", type=str, default=None, nargs='?')
    parser.add_argument("--bg_dir", help="str = dir of background. default is data_dir or test_data_dir", type=str, default=None, nargs='?')

    # for specific test data
    parser.add_argument("--test_data_dir", help="str = dir of specific test data. If not specified will run regular train-test", type=str, default=None, nargs='?')

    # directories helper variables
    parser.add_argument("--dir_name", help="str = data_dir for real data. Will also be the run dir name under EM_runs", type=str, default=None, nargs='?')
    parser.add_argument("--dataset", help="str = dataset for real data. e.g. H-C", type=str, default='', nargs='?')
    parser.add_argument("--suffix_result_dir", help="str, extra suffix for result dir (directly under EM_runs\data_dir)", type=str, default='', nargs='?')
    parser.add_argument("--const_train_result_dir", help="str, dir for train simulation which will be used for test", type=str, default=None, nargs='?')
    parser.add_argument("--create_split_data", help="bool = True if should create split data for batches (if doesn't already exists in output)", type=boolean_string, default=False, nargs='?')

    # for batch EM:
    parser.add_argument("--batch_t", help="int = run batch t", type=int, default=-1, nargs='?')
    parser.add_argument("--n_batches", help="int = number of batches", type=int, default=1, nargs='?')
    parser.add_argument("--create_NMF_start", help="bool = True if should create NMF starting point for batches (if doesn't already exists in start_train_params_dir)", type=boolean_string, default=False, nargs='?')
    parser.add_argument("--split_data_trainAB", help="bool = True if should split the train and test data to batches (if doesn't already exists in output)", type=boolean_string, default=False, nargs='?')
    parser.add_argument("--trainAB", help="bool = True if train A,B of batchEM", type=boolean_string, default=False, nargs='?')
    parser.add_argument("--arrange_trainAB_results", help="bool = True if should arrange results of the batches into res dir (if doesn't already exists in output)", type=boolean_string, default=False, nargs='?')
    parser.add_argument("--split_data_trainW", help="bool = True if should arrange the train and test data to trainW (if doesn't already exists in start_train_params_dir)", type=boolean_string, default=False, nargs='?')
    parser.add_argument("--trainW", help="bool = True if train W of batchEM", type=boolean_string, default=False, nargs='?')
    parser.add_argument("--trainW_n_genes", help="int = number of genes for trainW", type=int, default=200, nargs='?')
    parser.add_argument("--arrange_trainW_results", help="bool = True if should arrange results of trainW into res dir (if doesn't already exists in output)", type=boolean_string, default=False, nargs='?')
    parser.add_argument("--test_n_genes", help="int = number of genes for test", type=int, default=1000, nargs='?')

    parser.add_argument("--batch_itr_name", help="str, dir for batch iteration", type=str, default='itr1', nargs='?')

    return parser


if __name__ == '__main__':
    print(sys.argv, flush=True)
    parser = argparse.ArgumentParser()
    parser = _parse_args(parser)
    args = get_args_from_parser(parser)

    if args.output_dir is None:
        temp = f'{args.dataset}_{args.suffix_result_dir}' if args.dataset != '' else args.suffix_result_dir
        args.output_dir = f'EM_runs/{args.dir_name}/{temp}/{args.batch_itr_name}/'
    if args.data_dir is None:
        args.data_dir = args.output_dir
    if args.bg_dir is None:
        args.bg_dir = f'data/{args.dir_name}/'
    if args.start_train_params_dir is None:
        args.start_train_params_dir = args.output_dir + 'start/'

    if args.create_split_data:
        # create split data in itr_name/start
        split_train_test(args.N, args.M * args.n_batches, args.N_test,
                         samples_dir_readfrom=f'data/{args.dir_name}',
                         data_dir_saveto=args.start_train_params_dir, data_name='', dataset=args.dataset,
                         sample_split_dict=None, score_genes=False, take_first=True)


    # Run this for all batch_t in range(n_batches):
    if args.batch_itr_name == 'itr1':
        start_train_params_dir_AB = args.start_train_params_dir + f'bg={1 if args.bg else 0}_K={args.K}/train/sim_test/NMF/'
    elif args.start_train_params_dir is None:  # When itr1 is all done run the same with different start: # todo if more than 2 itr need to change to itr{l-1}
        temp = f'{args.dataset}_{args.suffix_result_dir}' if args.dataset != '' else args.suffix_result_dir
        args.start_train_params_dir = f'EM_runs/{args.dir_name}/{temp}/itr1/trainW/EM/res/bg={1 if args.bg else 0}_K={args.K}/'
        args.data_dir = args.start_train_params_dir
        start_train_params_dir_AB = args.start_train_params_dir
    else:
        # args.data_dir = args.start_train_params_dir  # TODO changed to commented
        start_train_params_dir_AB = args.start_train_params_dir
    run_AB(args, start_train_params_dir_AB)

    # When done run this:
    # arrange results in itr_name/trainAB/res
    output_dir_AB = args.output_dir + f'trainAB/'
    trainAB_results_dir = (root_path / output_dir_AB).parent / f'res_{(root_path / output_dir_AB).name}/bg={1 if args.bg else 0}_K={args.K}/'
    if args.arrange_trainAB_results:
        arrange_trainAB_results(output_dir_AB, trainAB_results_dir, args.n_batches, train_algo=f'{args.train_algo}kW')

    trainW_simulation_dir, trainW_results_dir = run_W(args, trainAB_results_dir, get_test=args.run_test)

    if args.test_data_dir:
        # run_test(args, trainW_simulation_dir, trainW_results_dir)
        run_test_after_trainAB(args, output_dir_AB, trainAB_results_dir)
