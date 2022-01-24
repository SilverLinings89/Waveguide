#!/bin/bash
# Use when a defined module environment related to OpenMPI is wished
module restore
mpiexec.hydra -bootstrap slurm Main/main --run ../Parameters/Run/3-3-24.prm --case ../Parameters/Case/hump_examples/predefined_case_13_with_pml.prm
