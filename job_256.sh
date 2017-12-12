#!/bin/bash
module load mpi/openmpi/2.0-intel-16.0
module load lib/hdf5/1.8-openmpi-2.0-intel-16.0
mpirun --bind-to core --map-by core -report-bindings ./main Parameters/Parameters_256mm.xml
