#!/bin/bash
# Use when a defined module environment related to OpenMPI is wished
module load mpi/openmpi/4.0
mpirun --bind-to core --map-by core -report-bindings Main/main --run ../Parameters/Run/3-3-25.prm --case ../Parameters/Case/hump_examples/predefined_case_19_with_pml.prm
