#!/bin/bash

sbatch -p single -n 8 --time=24:00:00 --mem=60000 <<EOT
#!/bin/bash
module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/8-1-1.prm --case ../Parameters/Case/optimization/straight_geometry_taper.prm
exit 0
EOT
