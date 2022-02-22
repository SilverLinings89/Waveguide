#!/bin/bash

for i in 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 
do
    JOBID=`sbatch -p multiple -n 360 --time=61:00:00 --mem=85000 --ntasks-per-node=27 ./job_hump_$i.sh`
    echo "For $i got id $JOBID" 
done