#!/bin/bash
module load lib/boost/1.56.0
module load compiler/gnu/5.2
module load mpi/openmpi/2.0-gnu-5.2
module load lib/hdf5/1.8-openmpi-2.0-gnu-5.2
module load numlib/mkl/11.3.4
mpirun --bind-to core --map-by core -report-bindings ../../../../Main/main Parameters/Rectangular/PWB/hom/PWB_Case_31.xml
