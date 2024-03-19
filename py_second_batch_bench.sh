#!/bin/bash
#SBATCH -J bench
#! -cwd
#SBATCH -e errp.txt
#SBATCH -o outp.txt
#SBATCH -p serial
#SBATCH --mem=10G

source /etc/profile
module load anaconda3

time python CUE_benchmark.py $1 $2 $3 $4 $5
