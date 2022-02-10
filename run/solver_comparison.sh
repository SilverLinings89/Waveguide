#!/bin/bash

declare -a arr=("GMRES" "MINRES" "TFQMR" "BICGS" "CG" "PCONLY");

truncate_output=true;
hide_errors=false;
domainlength=$((2*$1));
echo "Running for $1 processes.";
echo "Using domain length $domainlength.";

for i in "${arr[@]}"
do
    command="mpirun -np $1 ../build/Main/main --case ../Parameters/Case/solver_alternatives/base.prm --run ../Parameters/Run/solver_alternatives/base.prm --override \"solver_type=$i;geometry_size_z=$domainlength;processes_in_z=$1\"";
    echo "Using solver $i:";
    if $truncate_output ; then
        command="$command | grep 'Residual in step\|Solving took' ";
    fi
    if $hide_errors ; then
        command="$command 2> /dev/null";
    fi
    echo "Running the command: $command ";
    eval $command;
done