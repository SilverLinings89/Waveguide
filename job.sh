#!/bin/bash
#MSUB -l nodes=4:ppn=16
#MSUB -l walltime=05:00:00
#MSUB -l pmem=3000mb
#MSUB -v MPI_MODULE=mpi/ompi
#MSUB -v MPIRUN_OPTIONS="--bind-on core --map-by core -report-bindings"
#MSUB -v EXECUTABLE=./main
#MSUB -N Waveguide 64p
 
module load mpi/openmpi/2.0-intel-16.0
module load lib/hdf5/1.8-openmpi-2.0-intel-16.0
TASK_COUNT=$((${MOAB_PROCCOUNT}))
echo "${EXECUTABLE} running on ${MOAB_PROCCOUNT} cores with ${TASK_COUNT} MPI-tasks "
startexe="mpirun -n ${TASK_COUNT} ${MPIRUN_OPTIONS} ${EXECUTABLE}"
echo $startexe
exec $startexe
