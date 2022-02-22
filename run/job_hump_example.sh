#!/bin/bash
# Use when a defined module environment related to OpenMPI is wished
case=$1

module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/3-3-39.prm --case ../Parameters/Case/hump_examples/predefined_case_with_pml.prm --override \"predefined_case_number=$case\""
