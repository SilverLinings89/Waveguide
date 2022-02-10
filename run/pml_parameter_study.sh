#!/bin/bash

declare -A l2errors
declare -A linftyerrors

declare -a sigmas=(5 10 20 35 50 70 90)
declare -a ncells=(5 8 11 13 16)
order=$1
num_sigmas=${#sigmas[@]}
num_cell_opts=${#ncells[@]}
f2=" %9s"
pattern='Errors: L2 = ([0-9.e-]*) and Linfty  = ([0-9.e-]*)'

echo "Using PML scaling order $1. $num_sigmas options for sigma and $num_cell_opts options for cell count.";

for ((i=0;i<num_sigmas;i++)) do
    for ((j=0;j<num_cell_opts;j++)) do
        sigma=${sigmas[i]};
        cell_count=${ncells[j]};
        echo "Using sigma max $sigma and n cells $cell_count ";
        command="mpirun -np 4 ../build/Main/main --case ../Parameters/Case/PMLParameterTest/base.prm --run ../Parameters/Run/PMLParameterTest/base.prm --override \"pml_sigma_max=$sigma;n_pml_cells=$cell_count;pml_order=$order\"";
        #command="../Solutions/run53/main0.log"
        output_of_run=$($command)
        [[ $output_of_run =~ $pattern ]]
        l2errors[$i,$j]=${BASH_REMATCH[1]}
        linftyerrors[$i,$j]=${BASH_REMATCH[2]}
        echo "Found L2 error ${BASH_REMATCH[1]} and L_infty error ${BASH_REMATCH[2]}"
    done
done

echo "L2 errors:"
for ((i=0;i<num_sigmas;i++)) do
    for ((j=0;j<num_cell_opts;j++)) do
        printf "$f2" ${l2errors[$i,$j]}
    done
    echo
done
echo

echo "L_infty errors:"
for ((i=0;i<num_sigmas;i++)) do
    for ((j=0;j<num_cell_opts;j++)) do
        printf "$f2" ${linftyerrors[$i,$j]}
    done
    echo
done
echo
