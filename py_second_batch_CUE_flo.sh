#!/bin/bash
#SBATCH -J parity
#! -cwd
#SBATCH -e errp.txt
#SBATCH -o outp.txt
#SBATCH -p serial
#SBATCH --mem=10G

source /etc/profile
module load anaconda3

time python global_CUE_parity_symmetric_floquet.py $1 $2 $3 $4 $5
