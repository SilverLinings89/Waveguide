#!/bin/bash

module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/6-6-35.prm --case ../Parameters/Case/hump_examples/predefined_case_with_pml.prm

