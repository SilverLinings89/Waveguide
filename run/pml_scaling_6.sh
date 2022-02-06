#!/bin/bash

module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/sweeping_scaling/6.prm --case ../Parameters/Case/sweeping_scaling/6.prm
