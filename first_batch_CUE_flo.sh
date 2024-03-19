#!/bin/bash
q=3
ep=0.5
tr=0

mkdir /mmfs1/storage/users/harkind/Parity_sym_cue_temp/
mkdir /mmfs1/storage/users/harkind/Parity_sym_cue_temp/q$q
mkdir /mmfs1/storage/users/harkind/Parity_sym_cue_temp/q$q/tr$tr
for LL in {2..12..2}       
do
   mkdir /mmfs1/storage/users/harkind/Parity_sym_cue_temp/q$q/tr$tr/L$LL
for qt in {1..20000..1}                                  
do  
    qq=$(squeue|grep harkind|grep r|wc -l)
    while [ $qq -ge 350 ]
    do
    echo Sleep 200 secs load is $qq
    qq=$(squeue|grep harkind|grep r|wc -l)
    sleep 200
    done
    sbatch -J parity py_second_batch_CUE_flo.sh $q $ep $LL $qt $tr
done
done
