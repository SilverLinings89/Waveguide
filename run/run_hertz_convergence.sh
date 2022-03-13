#!/bin/bash

for i in 0 1 2
do
    sbatch -p single -n 9 --time=30:00:00 --mem=89000 --ntasks-per-node=9 <<EOT
#!/bin/bash
module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/3-3-3.prm --case ../Parameters/Case/convergence/order$i.prm
exit 0
EOT

done