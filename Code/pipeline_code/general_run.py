import os
import sys
sys.path.insert(1, os.getcwd())

from Simulation_code.Simulation import *
from Simulation_code.DataGenerator import *


sim_types = {'known_H': KnownHSimulation, 'EM': RandomParamsSimulation,
             'EMkAB': KnownABSimulation, 'EMkH': KnownHSimulation,
             'NMF': NMFRandomParamsSimulation, 'NMFkH': NMFKnownHSimulation,
             '2KNMF': NMFRandomParamsSimulation, '2KNMFkH': NMFKnownHSimulation,
             'batchEM': BatchRandomParamsSimulation, 'batchEMkAB': BatchKnownABSimulation, 'batchEMkW': BatchKnownWSimulation}

train_types = {'EM': 'EM', 'EMkH': 'EM', 'EMkAB': 'EM',
               'EMkW': 'EM',
               'batchEM': 'EM', 'batchEMkAB': 'EM', 'batchEMkABkAB': 'EM',
               'NMF': 'NMF', 'NMFkH': 'NMF', '2KNMF': 'NMF', '2KNMFkH': 'NMF'}
test_types = {'EM': 'kAB', 'EMkAB': 'kAB',
              'batchEM': 'kAB', 'batchEMkAB': 'kAB', 'batchEMkABkAB': 'kAB',
             'NMF': 'kH', 'NMFkH': 'kH', 'NMFkHkH': 'kH', 'EMkH': 'kH',
              'true': 'kH', '2KNMF': 'kH', '2KNMFkH': 'kH'}  # TODO


def setup_start(output_dir, sub_dir_name, test_suffix='', data_obj=None,
                generate_start_train=True, init_H_params=False, init_params=False):
    if generate_start_train:
        param_dir = output_dir/'train'/sub_dir_name
        param_generator = NMFStartParamsGenerator if not init_H_params else NNRStartParamsGenerator
    else: # generate_start_test
        param_dir = output_dir/('test' + test_suffix)/sub_dir_name
        param_generator = NNRStartParamsGenerator
    if init_params:
        param_generator = InitStartParamsGenerator

    return param_generator(param_dir, hyper_params=data_obj.hyper, dims=data_obj.dims, data=data_obj.data).generate()


def _run_train(output_dir, sim_name, sub_dir_name, train_sim_type, train_data_obj, train_start_obj,
              true_H, test_data_obj, eps_train=1e-6, max_itr_train=2000):
    if true_H:
        hat = test_data_obj.true  # set true as train results

    else:  # run train
        simulator = run_simulation(train_sim_type, output_dir / 'train' / sub_dir_name,
                                   data_obj=train_data_obj, param_obj=train_start_obj,
                                   sub_dir_name=sim_name, eps=eps_train, max_itr=max_itr_train)
        # set train results:
        hat = simulator.hat
    return hat


def _setup_kAB(data_obj, params):
    # set a, b from train as constant a, b
    assert params.a is not None
    data_obj.hyper.a, data_obj.hyper.b = params.a, params.b
    # test_data_obj.hyper.H = EMkAB(test_data_obj.data, hat, HyperParams(a=hat.a, b=hat.b), test_data_obj.dims).predict_H().mean(axis=0)
    data_obj.hyper.H = params.a / params.b
    data_obj.hyper.H[data_obj.hyper.H < EPS] = EPS
    return data_obj


def _setup_EMkH(data_obj, params):
    # set H from train as constant a, b
    assert params.a is not None
    data_obj.hyper.H = params.a / params.b
    data_obj.hyper.H[data_obj.hyper.H < EPS] = EPS
    return data_obj


def _setup_NMFkH(data_obj, params):
    # set H as constant H
    assert params.H is not None
    data_obj.hyper.H = params.H
    data_obj.hyper.H[data_obj.hyper.H < EPS] = EPS
    return data_obj


def _setup_H_params(data_obj, params, algo_type='EM', known_type='kAB'):
    data_obj = DataGenerator(data_obj.data_dir).copy(data_obj)
    if known_type == 'kAB': data_obj = _setup_kAB(data_obj, params)
    elif algo_type == 'NMF' and known_type == 'kH': data_obj = _setup_NMFkH(data_obj, params)
    elif algo_type == 'EM' and known_type == 'kH': data_obj = _setup_EMkH(data_obj, params)
    return data_obj


def _load_train_simulation(const_train_result_dir):
    if (const_train_result_dir / 'simulation_obj.pkl').exists():
        simulator = load_simulation(const_train_result_dir)
        hat = simulator.hat
    elif (const_train_result_dir / 'hat.pkl').exists():
        hat = Params.load_params(const_train_result_dir, 'hat')
    else:
        assert False, "train simulation doesn't exists"
    return hat


def _run_train_test(output_dir, sub_dir_name, train_algo, test_algo, train_data_obj, test_data_obj,
                    run_train=True, run_test=True, test_suffix='', sub_dir_name_test='', true_H=False,
                    init_H_params=False, init_params=False,
                    thresholds=None, K=0, const_train_result_dir=None):
    if K > 0:
        train_data_obj.dims = Dims(N=train_data_obj.N, M=train_data_obj.M, K=K)
        test_data_obj.dims = Dims(N=test_data_obj.N, M=test_data_obj.M, K=K)

    # create start params for train
    train_start_obj = None
    if init_H_params:
        known_params = train_data_obj.true
        train_data_obj = _setup_H_params(train_data_obj, known_params,
                                         algo_type=train_types[train_algo], known_type=test_types[train_algo])
    if run_train:
        train_start_obj = setup_start(
            output_dir, sub_dir_name, data_obj=train_data_obj,
            generate_start_train=True, init_params=init_params, init_H_params=init_H_params
        )

    # run train if test doesn't already exist
    test_simulation_path = output_dir/('test' + test_suffix)/sub_dir_name_test/f'{train_algo}_{test_algo}'/'simulation_obj.pkl'
    if not test_simulation_path.exists():
        if run_train:
            hat = _run_train(
                output_dir, train_algo, sub_dir_name, sim_types[train_algo], train_data_obj, train_start_obj,
                true_H, test_data_obj, thresholds.eps_train, thresholds.max_itr_train
            )
        else:
            hat = _load_train_simulation(const_train_result_dir)

        if run_test:
            test_data_obj = _setup_H_params(test_data_obj, hat, algo_type=train_types[train_algo], known_type=test_types[test_algo])
            # create start params for test
            test_start_obj = setup_start(
                output_dir, sub_dir_name_test, test_suffix=test_suffix,
                data_obj=test_data_obj, generate_start_train=False,
            )

    else:  # resume existing test
        test_data_obj, test_start_obj = None, None
        print(str(output_dir) + ' already tested', flush=True)

    # run test
    if run_test:
        run_simulation(sim_types[test_algo], output_dir/('test' + test_suffix)/sub_dir_name_test,
                       data_obj=test_data_obj, param_obj=test_start_obj,
                       sub_dir_name=f'{train_algo}_{test_algo}', eps=thresholds.eps_test, max_itr=thresholds.max_itr_test)


def general_run(output_dir, train_algo, test_algo, sub_dir_name, train_data_obj, test_data_obj, K,
                run_train=True, run_test=True, test_suffix='', sub_dir_name_test='', run_name='',
                init_params=False, init_H_params=False, true_H=False,
                thresholds=None, const_train_result_dir=None):
    if '2KNMF' in train_algo:  # 2KNMF needs new start parameters
        K = 2*K
        sub_dir_name += '_2KNMF'
    else:
        K=-1
    args = {'output_dir': output_dir/run_name, 'train_algo': train_algo, 'test_algo': test_algo,
            'sub_dir_name': sub_dir_name,
            'train_data_obj': train_data_obj, 'test_data_obj': test_data_obj,
            'test_suffix': test_suffix, 'true_H': true_H,
            'thresholds': thresholds,
            'init_params': init_params, 'init_H_params': init_H_params,
            'K': K, 'const_train_result_dir': const_train_result_dir}

    sub_dir_name_test = sub_dir_name_test if len(sub_dir_name_test) > 0 else sub_dir_name
    _run_train_test(run_train=run_train, run_test=run_test, sub_dir_name_test=sub_dir_name_test, **args)

