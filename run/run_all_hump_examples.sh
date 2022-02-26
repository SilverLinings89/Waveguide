#!/bin/bash

for i in 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 
do
    sbatch -p multiple -n 351 --time=70:00:00 --mem=85000 --ntasks-per-node=27 << EOT
    #!/bin/bash
    module restore
    mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/3-3-39.prm --case ../Parameters/Case/hump_examples/predefined_case_with_pml.prm --override "predefined_case_number=$i"
    EOT

done