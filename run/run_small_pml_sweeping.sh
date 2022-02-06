#!/bin/bash

JOBID=`sbatch -p single -n 1 --time=3:00:00 --mem=5000 ./pml_scaling_1.sh`
echo "For 1 got id $JOBID" 

JOBID=`sbatch -p single -n 2 --time=3:00:00 --mem=10000 ./pml_scaling_2.sh`
echo "For 2 got id $JOBID" 

JOBID=`sbatch -p single -n 3 --time=3:00:00 --mem=15000 ./pml_scaling_3.sh`
echo "For 3 got id $JOBID" 

JOBID=`sbatch -p single -n 4 --time=3:00:00 --mem=20000 ./pml_scaling_4.sh`
echo "For 4 got id $JOBID" 

JOBID=`sbatch -p single -n 5 --time=3:00:00 --mem=25000 ./pml_scaling_5.sh`
echo "For 5 got id $JOBID" 

JOBID=`sbatch -p single -n 6 --time=3:00:00 --mem=30000 ./pml_scaling_6.sh`
echo "For 6 got id $JOBID" 

JOBID=`sbatch -p single -n 7 --time=3:00:00 --mem=35000 ./pml_scaling_7.sh`
echo "For 7 got id $JOBID" 

JOBID=`sbatch -p single -n 8 --time=3:00:00 --mem=40000 ./pml_scaling_8.sh`
echo "For 8 got id $JOBID" 

JOBID=`sbatch -p single -n 9 --time=3:00:00 --mem=45000 ./pml_scaling_9.sh`
echo "For 9 got id $JOBID" 

JOBID=`sbatch -p single -n 10 --time=3:00:00 --mem=50000 ./pml_scaling_10.sh`
echo "For 10 got id $JOBID" 

JOBID=`sbatch -p single -n 11 --time=3:00:00 --mem=55000 ./pml_scaling_11.sh`
echo "For 11 got id $JOBID" 

JOBID=`sbatch -p single -n 12 --time=3:00:00 --mem=60000 ./pml_scaling_12.sh`
echo "For 12 got id $JOBID" 

JOBID=`sbatch -p single -n 13 --time=3:00:00 --mem=65000 ./pml_scaling_13.sh`
echo "For 13 got id $JOBID" 

JOBID=`sbatch -p single -n 14 --time=3:00:00 --mem=70000 ./pml_scaling_14.sh`
echo "For 14 got id $JOBID" 

JOBID=`sbatch -p single -n 15 --time=3:00:00 --mem=75000 ./pml_scaling_15.sh`
echo "For 15 got id $JOBID" 

JOBID=`sbatch -p single -n 16 --time=3:00:00 --mem=80000 ./pml_scaling_16.sh`
echo "For 16 got id $JOBID" 





