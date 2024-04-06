import os
import sys
sys.path.insert(1, os.getcwd())

from pipeline_code.general_setup import *
from pipeline_code.data_simulation import run_data


def run_data_simulation(args):

    flag_real_data = not args.test_data_dir
    flag_run_data_test = args.test_data_dir

    thresholds = Thresholds(eps_train=args.eps_train, eps_test=args.eps_test,
                            max_itr_train=args.max_itr_train, max_itr_test=args.max_itr_test)

    train_algo = args.train_algo
    run_test = args.run_test
    test_algo = args.test_algo
    if test_algo is None:
        test_algo = (f'{train_algo}kAB' if not train_algo.endswith('kAB') else train_algo) if ('EM' in train_algo) else \
            (f'{train_algo}kH' if not train_algo.endswith('kH') else train_algo)

    output_dir_sim = root_path / args.output_dir
    train_data_dir = output_dir_sim if args.data_dir is None else root_path / args.data_dir
    test_data_dir = train_data_dir
    if args.test_data_dir:
        test_data_dir = root_path / 'data' / args.test_data_dir
        const_train_result_dir = root_path / args.const_train_result_dir
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
                train_data_name=args.train_data_filename, test_data_name=args.test_data_filename, bg_name=args.background_filename,
                run_normal_test=run_test, run_name=str(run_name), batch_t=args.batch_t, start_train_params_dir=args.start_train_params_dir,
                thresholds=thresholds, train_batchW=args.trainW,
                new_k=args.new_k, new_ks_W=args.new_ks_W, new_ks_AB=args.new_ks_AB
            )
            print(f'end {run_name}', flush=True)

    if flag_run_data_test:
        T = 1
        for run_name in range(T):
            run_data(
                output_dir_sim, args.N, args.M, args.K, args.N_test,
                train_algo, test_algo, args.sim_dir_name, args.bg,
                train_data_dir, test_data_dir, bg_dir,
                args.train_W_name, args.train_a_name, args.train_b_name, args.train_H_name, args.train_lam_name,
                test_data_name=args.test_data_filename, bg_name=args.background_filename, thresholds=thresholds,
                run_train=False, run_normal_test=False, run_special_test=True,
                const_train_result_dir=const_train_result_dir, run_name=train_algo,  # TODO
                batch_t=args.batch_t
            )

            print(f'end {run_name}', flush=True)

