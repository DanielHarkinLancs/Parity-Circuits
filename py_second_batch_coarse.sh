#!/bin/bash
#SBATCH -J coarse
#! -cwd
#SBATCH -e errp.txt
#SBATCH -o outp.txt
#SBATCH -p serial
#SBATCH --mem=5G

source /etc/profile
module load anaconda3

time python global_CUE_coarse.py $1 $2 $3 $4 $5
