#!/bin/bash

for i in 8 10 12 14 16 18 20 22 24 26 
do
    sbatch -p single -n 8 --time=6:00:00 --mem=80000 <<EOT
#!/bin/bash
module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/8-1-1.prm --case ../Parameters/Case/sweeping_scaling/stretching_example.prm --override "system_length=$i"
exit 0
EOT

done