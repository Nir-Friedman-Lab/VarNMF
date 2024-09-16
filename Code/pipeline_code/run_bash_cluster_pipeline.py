import numpy as np
import pandas as pd
import shutil
import os
from pathlib import Path


def mkdir(dir_path_wo_root, parents=True):
    if not (root_path/dir_path_wo_root).exists(): (root_path/dir_path_wo_root).mkdir(parents=parents)


root_path = Path(__file__).parent.parent.parent  # should always give "simulations"
python_path = '/opt/homebrew/bin/python3.10'
max_processes = 4


def create_bash_file(job_file, job_name, job_python_name, job_args_str, job_output_path, array_name=None):
    # for regular use
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")

        if array_name is not None:  # write loop with max_processes (https://stackoverflow.com/questions/75695266/run-bash-command-in-parallel-for-n-processes-in-batch)
            fh.writelines(f'max_processes={max_processes}\n')
            fh.writelines('counter=0\n')
            fh.writelines('for a in {' + array_name.replace('-', '..') + '}; do\n')
            fh.writelines(f"\t{python_path} {job_python_name} {job_args_str} > {job_output_path}/$a.txt &\n")
            fh.writelines('\tcounter=$((counter + 1))\n')
            # if the counter equal to processes_max: wait for all processes to finish
            fh.writelines('\tif [[ $counter -eq max_processes ]]; then\n')
            fh.writelines('\t\twait\n')
            fh.writelines('\t\tcounter=0\n')
            fh.writelines('\tfi\n')
            fh.writelines('done')
        else:
            fh.writelines(f"{python_path} {job_python_name} {job_args_str} > {job_output_path}/{job_name}.txt")


def create_bash_args_str(N, M, K, dir_name, dataset, sub_dir1, output_dir,
                         start_train_params_dir,
                         n_batches=70, batch_t='0',
                         trainAB=False, split_data_trainAB=False, arrange_trainAB_results=False,
                         trainW=False, split_data_trainW=False, arrange_trainW_results=False,
                         train_a_name='hat_a', train_b_name='hat_b', train_W_name='hat_W',
                         eps_train='1e-6', max_itr_train=250,
                         run_test=False, N_test=0, eps_test='1e-6', max_itr_test=0):
    job_args_str = f"--train_algo=batchEM " \
                   f"--N={N} " \
                   f"--M={M} " \
                   f"--K={K} " \
                   f"--bg=True " \
                   f"--dir_name={dir_name}/ " \
                   f"--output_dir=results/{dir_name}/{output_dir}/ " \
                   f"--bg_dir=data/{dataset}/{sub_dir1}/ " \
                   f"--data_dir=data/{dir_name}/ " \
                   f"--run_name_T=1 " \
                   f"--eps_train={eps_train} " \
                   f"--max_itr_train={max_itr_train} " \
                   f"--run_test={run_test} " \
                   f"--N_test={N_test} " \
                   f"--eps_test={eps_test} " \
                   f"--max_itr_test={max_itr_test} " \
                   f"--n_batches={n_batches} " \
                   f"--batch_t={batch_t} " \
                   f"--batch_itr_name=batch " \
                   f"--start_train_params_dir={start_train_params_dir} " \
                   f"--train_a_name={train_a_name} " \
                   f"--train_b_name={train_b_name} " \
                   f"--train_W_name={train_W_name} " \
                   f"--trainAB={trainAB} " \
                   f"--split_data_trainAB={split_data_trainAB} " \
                   f"--arrange_trainAB_results={arrange_trainAB_results} " \
                   f"--trainW={trainW} " \
                   f"--split_data_trainW={split_data_trainW} " \
                   f"--arrange_trainW_results={arrange_trainW_results}"
    return job_args_str


def create_bash_file_NMF(N, M, K, dataset, sub_dir1, sub_dir2):
    # for trainW
    # Example: K=10, N=185, M=7000, N_test=120,
    # dataset="ICLR-CRC+SCLC", sub_dir1="unsupervised", sub_dir2="medium-data"

    itr_name = 'NMF'
    job_output_dir = f'c_outputs/{dataset}/{sub_dir1}/{sub_dir2}/{itr_name}'
    mkdir(job_output_dir, parents=True)

    job_output_path = f'{root_path}/{job_output_dir}'
    job_file = root_path/f'Code/sh_files/{dataset}/{sub_dir1}-{sub_dir2}-batch-{itr_name}-K={K}.sh'
    job_name = f'K={K}-{itr_name}-{sub_dir1}-{sub_dir2}'
    dir_name = f"{dataset}/{sub_dir1}/{sub_dir2}"

    job_args_str = f"--train_algo=NMF " \
                   f"--N={N} " \
                   f"--M={M} " \
                   f"--K={K} " \
                   f"--bg=True " \
                   f"--dir_name={dir_name}/ " \
                   f"--output_dir=results/{dir_name}/ " \
                   f"--eps_train=0 " \
                   f"--max_itr_train=250 " \
                   f"--run_test=True " \
                   f"--run_name_T=1 " \
                   f"--bg_dir=data/{dataset}/{sub_dir1}/ " \
                   f"--data_dir=data/{dir_name}/ "

    job_python_name = f'{root_path}/Code/pipeline_code/setup_data_run.py'
    create_bash_file(job_file, job_name, job_python_name, job_args_str, job_output_path)
    return job_file


def create_bash_file_trainAB(N, M, K, dataset, sub_dir1, sub_dir2, n_batches, itr=1, run_first=False):
    # for trainAB
    # Example: K=10, n_batches=70, N=185, M=100, dataset="ICLR-CRC+SCLC", sub_dir1="unsupervised", sub_dir2="medium-data"
    itr_name = 'trainAB' + ('' if itr == 1 else f'{itr}') + ('splitAB' if run_first else '')
    job_output_dir = f'c_outputs/{dataset}/{sub_dir1}/{sub_dir2}/{itr_name}/K={K}'
    mkdir(job_output_dir, parents=True)

    job_output_path = f'{root_path}/{job_output_dir}'
    job_file = root_path/f'Code/sh_files/{dataset}/{sub_dir1}-{sub_dir2}-batch-{itr_name}-K={K}.sh'
    job_name = f'K={K}-splitAB' if run_first else f'K={K}-{itr_name}-{sub_dir1}-{sub_dir2}'
    dir_name = f"{dataset}/{sub_dir1}/{sub_dir2}"
    output_dir = f'batch{itr}'
    if itr == 1:  # start is NMF
        start_train_params_dir = f'results/{dir_name}/bg=1_K={K}/0/train/sim_test/'
        train_a_name = 'start_a'
        train_b_name = 'start_b'
        train_W_name = 'start_W'
    else: # start is last itr
        start_train_params_dir = f'results/{dir_name}/batch{itr-1}/res_trainW/bg=1_K={K}/'
        train_a_name = 'hat_a'
        train_b_name = 'hat_b'
        train_W_name = 'hat_W'

    job_args_str = create_bash_args_str(
        N, M, K, dir_name, dataset, sub_dir1, output_dir, start_train_params_dir,
        n_batches=n_batches, batch_t='0' if run_first else '$a',
        trainAB=not run_first, split_data_trainAB=run_first, arrange_trainAB_results=False,
        train_a_name=train_a_name, train_b_name=train_b_name, train_W_name=train_W_name,
        eps_train='0', max_itr_train=250,
        run_test=False)

    array_name = None if run_first else f'0-{n_batches-1}'
    job_python_name = f'{root_path}/Code/pipeline_code/setup_batch_run.py'
    create_bash_file(job_file, job_name, job_python_name, job_args_str, job_output_path, array_name=array_name)
    return job_file


def create_bash_file_trainW(N, M, K, dataset, sub_dir1, sub_dir2, n_batches, itr=1, run_test=False, N_test=0):
    # for trainW
    # Example: K=10, n_batches=70, N=185, M=1000, dataset="ICLR-CRC+SCLC", sub_dir1="unsupervised", sub_dir2="medium-data"
    itr_name = 'trainW' + ('' if itr == 1 else f'{itr}') + ('-testW' if run_test else '')
    job_output_dir = f'c_outputs/{dataset}/{sub_dir1}/{sub_dir2}/{itr_name}'
    mkdir(job_output_dir, parents=True)

    job_output_path = f'{root_path}/{job_output_dir}'
    job_file = root_path/f'Code/sh_files/{dataset}/{sub_dir1}-{sub_dir2}-batch-{itr_name}-K={K}.sh'
    job_name = f'K={K}-{itr_name}-{sub_dir1}-{sub_dir2}'
    dir_name = f"{dataset}/{sub_dir1}/{sub_dir2}"
    output_dir = f'batch{itr}'
    start_train_params_dir = f'results/{dir_name}/batch{itr}/res_trainAB/bg=1_K={K}/'

    job_args_str = create_bash_args_str(
        N, M, K, dir_name, dataset, sub_dir1, output_dir, start_train_params_dir,
        n_batches=n_batches, batch_t='0',
        trainAB=False, split_data_trainAB=False, arrange_trainAB_results=True,
        trainW=True, split_data_trainW=True, arrange_trainW_results=True,
        eps_train='1e-6',
        max_itr_train=0,
        run_test=run_test, N_test=N_test,
        eps_test='1e-6',
        max_itr_test=0)

    array_name = None
    job_python_name = f'{root_path}/Code/pipeline_code/setup_batch_run.py'
    create_bash_file(job_file, job_name, job_python_name, job_args_str, job_output_path, array_name=array_name)
    return job_file


def create_bash_file_posteriors(N, M, K, dataset, sub_dir1, sub_dir2, run_test=False, N_test=0):
    # for post and post-test
    # Example: K=10, N=185, M=7000, dataset="ICLR-CRC+SCLC", sub_dir1="unsupervised", sub_dir2="medium-data"
    itr_name = 'post' + ('-test' if run_test else '')
    job_output_dir = f'c_outputs/{dataset}/{sub_dir1}/{sub_dir2}/{itr_name}'
    mkdir(job_output_dir, parents=True)

    job_output_path = f'{root_path}/{job_output_dir}'
    job_file = root_path/f'Code/sh_files/{dataset}/{sub_dir1}-{sub_dir2}-batch-{itr_name}-K={K}.sh'
    job_name = f'K={K}-{itr_name}-{sub_dir1}-{sub_dir2}'
    dir_name = f"{dataset}/{sub_dir1}/{sub_dir2}"

    job_args_str = f"--N={N_test if run_test else N} " \
                   f"--M={M} " \
                   f"--K={K}  " \
                   f"--bg_dir=data/{dataset}/{sub_dir1}/background-filtered " \
                   f"--data_dir=data/{dir_name}/{'test' if run_test else 'train'} " \
                   f"--params_dir=results/{dir_name}/batch2/res_trainW/bg=1_K={K}/{'test_' if run_test else ''}hat " \
                   f"--output_dir=results/{dir_name}/batch2/res_trainW/bg=1_K={K}/{'test' if run_test else 'train'}/"

    array_name = None
    job_python_name = f'{root_path}/Code/pipeline_code/calc_posterior.py'
    create_bash_file(job_file, job_name, job_python_name, job_args_str, job_output_path, array_name=array_name)
    return job_file


def run_bash_pipeline_func(N, M, K, dataset, sub_dir1, sub_dir2, param, n_batches=70, itr=1,
                            N_test=0, dependencies=None, killable=True):
    # Example:
    # K=10, n_batches=70, N=185, M=100, dataset="ICLR-CRC+SCLC", sub_dir1="unsupervised", sub_dir2="medium-data",
    # param='AB', itr=1
    job_file = None
    if param == 'NMF-start':
        job_file = create_bash_file_NMF(N, M, K, dataset, sub_dir1, sub_dir2)
    elif param == 'AB-first':
        job_file = create_bash_file_trainAB(N, M, K, dataset, sub_dir1, sub_dir2, n_batches, itr=itr, run_first=True)
    elif param == 'AB':
        job_file = create_bash_file_trainAB(N, M, K, dataset, sub_dir1, sub_dir2, n_batches, itr=itr)
    elif param == 'W':
        job_file = create_bash_file_trainW(N, M, K, dataset, sub_dir1, sub_dir2, n_batches, itr=itr,
                                           run_test=itr == 2, N_test=N_test)
    elif param == 'post':
        job_file = create_bash_file_posteriors(N, M, K, dataset, sub_dir1, sub_dir2, run_test=False)
    elif param == 'post-test':
        job_file = create_bash_file_posteriors(N, M, K, dataset, sub_dir1, sub_dir2, run_test=True, N_test=N_test)

    dependencies_str = ('\ndependencies: ' + ', '.join(dependencies) + ' ') if dependencies is not None else ''

    print(f"bash {job_file}", dependencies_str)
    return [job_file.name]


def run_pipeline(N, N_test, M, K, dataset, sub_dir1, sub_dir2, n_batches,
                 run_start_NMF=True, run_itr1_AB=True, run_itr1_W=True,
                 run_itr2_AB=True, run_itr2_W=True, run_post=True,
                 jobid=None):
    # Example:
    # N=185, N_test=120, M=100, K=10, dataset="ICLR-CRC+SCLC", sub_dir1="unsupervised", sub_dir2="medium-data", n_batches=70
    M_trainAB = 100
    M_trainW = K*100

    # create NMF start
    if run_start_NMF:
        jobid = run_bash_pipeline_func(N, M, K, dataset, sub_dir1, sub_dir2, param='NMF-start',
                                        dependencies=None, killable=True)
        Kt_nmf = int(np.ceil((2 * M * K + N * K) / (M + N)))
        if Kt_nmf > 10:  # run K_tilde-NMF
            run_bash_pipeline_func(N, M, Kt_nmf, dataset, sub_dir1, sub_dir2, param='NMF-start',
                                            dependencies=None, killable=True)

    # trainAB
    if run_itr1_AB:
        jobid = run_bash_pipeline_func(N, M_trainAB, K, dataset, sub_dir1, sub_dir2, param='AB-first',
                                        n_batches=n_batches, itr=1, dependencies=jobid, killable=True)
        jobid = run_bash_pipeline_func(N, M_trainAB, K, dataset, sub_dir1, sub_dir2, param='AB',
                                        n_batches=n_batches, itr=1, dependencies=jobid, killable=True)

    # trainW
    if run_itr1_W:
        jobid = run_bash_pipeline_func(N, M_trainW, K, dataset, sub_dir1, sub_dir2, param='W',
                                        itr=1, dependencies=jobid, killable=True)


    # trainAB2
    if run_itr2_AB:
        jobid = run_bash_pipeline_func(N, M_trainAB, K, dataset, sub_dir1, sub_dir2, param='AB-first',
                                        n_batches=n_batches, itr=2, dependencies=jobid, killable=True)
        jobid = run_bash_pipeline_func(N, M_trainAB, K, dataset, sub_dir1, sub_dir2, param='AB',
                                        n_batches=n_batches, itr=2, dependencies=jobid, killable=True)

    # trainW2+testW
    if run_itr2_W:
        jobid = run_bash_pipeline_func(N, M_trainW, K, dataset, sub_dir1, sub_dir2, param='W',
                                        itr=2, N_test=N_test, dependencies=jobid, killable=True)


    if run_post:
        # post
        run_bash_pipeline_func(N, M, K, dataset, sub_dir1, sub_dir2, param='post', dependencies=jobid, killable=True)

        # post-test
        run_bash_pipeline_func(N, M, K, dataset, sub_dir1, sub_dir2, param='post-test',
                                N_test=N_test, dependencies=jobid, killable=True)



if __name__ == '__main__':
    N = 185
    N_test = 120
    M = 7000
    dataset = "CRC+SCLC"

    sub_dir1 = "unsupervised1"
    sub_dir2 = "selected-genes"
    n_batches = 70
    # for K in [5,6,7,8,9,10]:
    for K in [4]:
        # jobid = ['17225026'] #None  # change if there is something currently running to start after it finishes
        jobid = None
        run_pipeline(N, N_test, M, K, dataset, sub_dir1, sub_dir2, n_batches,
                     run_start_NMF=True, run_itr1_AB=True, run_itr1_W=True,
                     run_itr2_AB=True, run_itr2_W=True, run_post=True, jobid=jobid)

