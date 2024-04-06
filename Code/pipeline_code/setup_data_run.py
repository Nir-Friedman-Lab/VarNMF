import os
import sys
sys.path.insert(1, os.getcwd())

from pipeline_code.general_setup import *
from pipeline_code.data_simulation import run_data_simulation
from lib_code.NMF_lib import load_datasets


def _get_data(samples_dir_readfrom, dataset, prefix=''):
    # get data
    _, _, datasets, _, _, _, _ = load_datasets(
        samples_path=root_path/samples_dir_readfrom, dataset_list=[dataset],
        counts=True, background=False, atlas_path=None, prefix=prefix
    )
    return datasets[dataset]


def _split_sample_names(all_samples, N, N_test, samples_dir_readfrom, sample_split_dict=None):
    # split N+N_test samples to train and test
    if (root_path/samples_dir_readfrom / f"train.csv").exists():
        train_samples = pd.read_csv(root_path/samples_dir_readfrom / f"train.csv", index_col=0).columns
        test_samples = pd.read_csv(root_path/samples_dir_readfrom / f"test.csv", index_col=0).columns
    elif sample_split_dict is not None:
        train_samples, test_samples = sample_split_dict['train'], sample_split_dict['test']
    else:
        idx = all_samples[np.random.choice(len(all_samples), size=N + N_test, replace=False)]
        train_samples, test_samples = idx[:N], idx[N:]
    return train_samples, test_samples


def _select_genes(data, M, score_genes=False, take_first=True):
    # get M genes
    if take_first:  # this is a run with (almost) all genes
        genes = data.index[:M]
    elif score_genes:
        scores = (data.var(axis=1) / (data.mean(axis=1) + EPS)).sort_values()
        genes = scores.index[::-1][:M]
    else:  # take all
        genes = data.index
    return genes


def split_train_test(N, M, N_test, samples_dir_readfrom, data_dir_saveto, data_name, dataset,
                     sample_split_dict=None, score_genes=False, take_first=True, prefix='', get_test=True):

    # no split exists
    if not (root_path/data_dir_saveto / (data_name + f"train.csv")).exists():
        print('create split data', flush=True)
        # get data
        data = _get_data(samples_dir_readfrom, dataset, prefix=prefix)

        # split N+N_test samples to train and test
        train_samples, test_samples = _split_sample_names(data.columns, N, N_test, samples_dir_readfrom, sample_split_dict)

        # get M genes
        genes = _select_genes(data, M, score_genes, take_first)

        # split data and save
        train_data = data[train_samples].loc[genes]
        if get_test: test_data = data[test_samples].loc[genes]
        if not (root_path/data_dir_saveto).exists(): (root_path/data_dir_saveto).mkdir(parents=True)
        train_data.to_csv(root_path/data_dir_saveto / (data_name + f"train.csv"))
        if get_test: test_data.to_csv(root_path/data_dir_saveto / (data_name + f"test.csv"))


def get_split_data(split_data_dir, data_name, get_test=True):
    train_data, test_data = None, None
    train_data = pd.read_csv(root_path/split_data_dir / (data_name + f"train.csv"), index_col=0)
    if get_test: test_data = pd.read_csv(root_path/split_data_dir / (data_name + f"test.csv"), index_col=0)
    return train_data, test_data


def _parse_args(parser):
    parser = parse_args_general(parser)

    # directories
    parser.add_argument("--output_dir", help="str = output_dir for the simulation", type=str, default=None, nargs='?')
    parser.add_argument("--data_dir", help="str = dir of split data. default is output_dir", type=str, default=None, nargs='?')
    parser.add_argument("--bg_dir", help="str = dir of background. default is data_dir or test_data_dir", type=str, default=None, nargs='?')

    # for specific test data
    parser.add_argument("--test_data_dir", help="str = dir of specific test data. If not specified will run regular train-test", type=str, default=None, nargs='?')
    parser.add_argument("--const_train_result_dir", help="str = dir of train results. Will also be the output dir if specified", type=str, default=None, nargs='?')

    # directories helper variables
    parser.add_argument("--dir_name", help="str = data_dir for real data. Will also be the run dir name under EM_runs", type=str, default=None, nargs='?')
    parser.add_argument("--dataset", help="str = dataset for real data. e.g. H-C", type=str, default='', nargs='?')
    parser.add_argument("--suffix_result_dir", help="str, extra suffix for result dir (directly under EM_runs\data_dir)", type=str, default='', nargs='?')
    parser.add_argument("--run_name_constant_results", help="str, the run_name of the train run", type=str, default='', nargs='?')
    parser.add_argument("--create_split_data", help="bool = True if should create split data (if doesn't already exists in output)", type=boolean_string, default=False, nargs='?')
    parser.add_argument("--split_data_file_prefix", help="str, extra suffix for split data", type=str, default='', nargs='?')

    return parser


if __name__ == '__main__':

    print(sys.argv, flush=True)
    parser = argparse.ArgumentParser()
    parser = _parse_args(parser)
    args = get_args_from_parser(parser)

    if args.data_dir is None:
        temp = f'{args.dataset}_{args.suffix_result_dir}' if args.dataset != '' else args.suffix_result_dir
        args.data_dir = f'EM_runs/{args.dir_name}/{temp}'
    if args.output_dir is None:
        args.output_dir = args.data_dir
    if args.bg_dir is None:
        args.bg_dir = f'data/{args.dir_name}'
    # if args.start_train_params_dir is None:  TODO
    #     args.start_train_params_dir = args.data_dir

    if args.test_data_dir:
        temp = f'{args.dataset}_{args.suffix_result_dir}' if args.dataset != '' else args.suffix_result_dir
        args.data_dir = f'EM_runs/{args.dir_name}/{temp}'
        args.output_dir = f'{args.data_dir}/bg={1 if args.bg else 0}_K={args.K}/{args.run_name_constant_results}/test_{args.test_data_dir}'
        args.const_train_result_dir = f'{args.data_dir}/bg={1 if args.bg else 0}_K={args.K}/{args.run_name_constant_results}/train/sim_test/{args.train_algo}'
        args.test_data_filename = 'counts-filtered'

    if args.create_split_data:
        prefix = args.split_data_file_prefix
        split_train_test(args.N, args.M, args.N_test, samples_dir_readfrom=f'data/{args.dir_name}',
                         data_dir_saveto=args.data_dir, data_name='', dataset=args.dataset,
                         sample_split_dict=None, score_genes=False, take_first=True, prefix=prefix)

    run_data_simulation(args)


