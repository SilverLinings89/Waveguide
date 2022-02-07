#!/bin/bash

JOBID=`sbatch -p multiple -n 20 --ntasks-per-node=10 --time=6:00:00 --mem=85000 ./pml_scaling_20.sh`
echo "For 20 got id $JOBID" 

JOBID=`sbatch -p multiple -n 24 --ntasks-per-node=12 --time=7:00:00 --mem=85000 ./pml_scaling_24.sh`
echo "For 24 got id $JOBID" 

JOBID=`sbatch -p multiple -n 28 --ntasks-per-node=14 --time=8:00:00 --mem=85000 ./pml_scaling_28.sh`
echo "For 28 got id $JOBID" 

JOBID=`sbatch -p multiple -n 32 --ntasks-per-node=16 --time=9:00:00 --mem=85000 ./pml_scaling_32.sh`
echo "For 32 got id $JOBID" 

JOBID=`sbatch -p multiple -n 64 --ntasks-per-node=16 --time=9:00:00 --mem=85000 ./pml_scaling_64.sh`
echo "For 64 got id $JOBID" 
