#!/bin/bash
q=32
LL=2
tr=0
eps=0.5

mkdir /mmfs1/storage/users/harkind/coarse_temp/
mkdir /mmfs1/storage/users/harkind/coarse_temp/q$q
mkdir /mmfs1/storage/users/harkind/coarse_temp/q$q/tr$tr

for qt in {1..20000..1}                                  
do  
    qq=$(squeue|grep harkind|grep r|wc -l)
    while [ $qq -ge 350 ]
    do
    echo Sleep 200 secs load is $qq
    qq=$(squeue|grep harkind|grep r|wc -l)
    sleep 200
    done
    sbatch -J coarse py_second_batch_coarse.sh $q $LL $qt $tr $eps
done

