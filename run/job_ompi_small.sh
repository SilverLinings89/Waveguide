#!/bin/bash
# Use when a defined module environment related to OpenMPI is wished
module load mpi/openmpi/4.0
mpirun --bind-to core --map-by core -report-bindings ../build/Main/main --run ../Parameters/Run/2-2-2.prm --case ../Parameters/Case/straight_waveguide/lowest_order_with_pml.prm
