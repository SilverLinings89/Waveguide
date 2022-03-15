#!/bin/bash

for i in 0 1 2
do
    sbatch -p single -n 8 --mem=120000 --time=20:00:00 <<EOT
#!/bin/bash
module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/2-2-2.prm --case ../Parameters/Case/convergence/order$i.prm
exit 0
EOT

done
