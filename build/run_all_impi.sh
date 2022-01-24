#!/bin/bash

for i in 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 
do
    JOBID=`sbatch -p multiple -N 8 -n 216 --time=61:00:00 --mem=85000 --ntasks-per-node=27 ./job_ompi_$i.sh`
    echo "For $i got id $JOBID" 
done