#!/bin/bash

sbatch -p single -n 10 --time=30:00:00 --mem=89000 <<EOT
#!/bin/bash
module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/10-1-1.prm --case ../Parameters/Case/optimization/base.prm
exit 0
EOT

sbatch -p single -n 10 --time=30:00:00 --mem=89000 <<EOT
#!/bin/bash
module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/10-1-1.prm --case ../Parameters/Case/optimization/base.prm
exit 0
EOT

