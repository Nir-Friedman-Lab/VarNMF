#!/bin/bash
/opt/homebrew/bin/python3.10 /Users/elafallik/Documents/submission-github/Code/pipeline_code/calc_posterior.py --N=185 --M=7000 --K=4  --bg_dir=data/CRC+SCLC/unsupervised1/background-filtered --data_dir=data/CRC+SCLC/unsupervised1/selected-genes/train --params_dir=results/CRC+SCLC/unsupervised1/selected-genes/batch2/res_trainW/bg=1_K=4/hat --output_dir=results/CRC+SCLC/unsupervised1/selected-genes/batch2/res_trainW/bg=1_K=4/train/ > /Users/elafallik/Documents/submission-github/c_outputs/CRC+SCLC/unsupervised1/selected-genes/post/K=4-post-unsupervised1-selected-genes.txt