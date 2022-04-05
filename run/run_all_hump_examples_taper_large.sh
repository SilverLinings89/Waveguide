#!/bin/bash

# for i in 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 
for i in 1 5 11 17 23 29 33
do
    sbatch -p multiple -n 1920 --time=60:00:00 --mem=89000 --ntasks-per-node=32 <<EOT
#!/bin/bash
module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/8-8-30.prm --case ../Parameters/Case/hump_examples/predefined_case_with_pmlmax.prm --override "predefined_case_number=$i"
exit 0
EOT

done
