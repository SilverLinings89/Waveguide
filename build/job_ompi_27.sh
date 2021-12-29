#!/bin/bash
# Use when a defined module environment related to OpenMPI is wished
module restore
mpirun --bind-to core --map-by core -report-bindings Main/main --run ../Parameters/Run/3-3-25.prm --case ../Parameters/Case/hump_examples/predefined_case_27_with_pml.prm
