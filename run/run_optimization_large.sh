#!/bin/bash

sbatch -p multiple -n 640 --time=71:00:00 --mem=90000 --ntasks-per-node=32<<EOT
#!/bin/bash
module restore
mpiexec.hydra -bootstrap slurm ../build/Main/main --run ../Parameters/Run/8-8-10.prm --case ../Parameters/Case/optimization/larger_geometry_taper.prm
exit 0
EOT


