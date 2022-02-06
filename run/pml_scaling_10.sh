#!/bin/bash

module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/sweeping_scaling/10.prm --case ../Parameters/Case/sweeping_scaling/10.prm
