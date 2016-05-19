#!/bin/bash -l

#SBATCH -A snic2016-7-54
#SBATCH -p core 
#SBATCH -n 8
#SBATCH -t 3-00:00:00
#SBATCH -J emotion_train

python /home/danielll/deep/cifar10_train_distributed.py
