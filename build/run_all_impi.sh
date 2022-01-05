#!/bin/bash

for i in 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 
do
    JOBID=`sbatch -p multiple -N 15 -n 225 --time=48:00:00 --mem=80000 --ntasks-per-node=15 ./job_ompi_$i.sh`
    echo "For $i got id $JOBID" 
done