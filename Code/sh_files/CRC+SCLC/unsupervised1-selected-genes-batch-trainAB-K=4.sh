#!/bin/bash
max_processes=4
counter=0
for a in {0..69}; do
	/opt/homebrew/bin/python3.10 /Users/elafallik/Documents/submission-github/Code/pipeline_code/setup_batch_run.py --train_algo=batchEM --N=185 --M=100 --K=4 --bg=True --dir_name=CRC+SCLC/unsupervised1/selected-genes/ --output_dir=results/CRC+SCLC/unsupervised1/selected-genes/batch1/ --bg_dir=data/CRC+SCLC/unsupervised1/ --data_dir=data/CRC+SCLC/unsupervised1/selected-genes/ --run_name_T=1 --eps_train=0 --max_itr_train=250 --run_test=False --N_test=0 --eps_test=1e-6 --max_itr_test=0 --n_batches=70 --batch_t=$a --batch_itr_name=batch --start_train_params_dir=results/CRC+SCLC/unsupervised1/selected-genes/bg=1_K=4/0/train/sim_test/ --train_a_name=start_a --train_b_name=start_b --train_W_name=start_W --trainAB=True --split_data_trainAB=False --arrange_trainAB_results=False --trainW=False --split_data_trainW=False --arrange_trainW_results=False > /Users/elafallik/Documents/submission-github/c_outputs/CRC+SCLC/unsupervised1/selected-genes/trainAB/K=4/$a.txt &
	counter=$((counter + 1))
	if [[ $counter -eq max_processes ]]; then
		wait
		counter=0
	fi
done