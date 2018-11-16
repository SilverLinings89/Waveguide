#!/bin/bash
val=3
for value in {1..16}
do 
	val=$(expr $val + 2)
	rm ./case_${val}.sh
	echo "#!/bin/bash" >> ./case_${val}.sh
	echo "module load lib/boost/1.56.0" >> ./case_${val}.sh
	echo "module load compiler/gnu/5.2" >> ./case_${val}.sh
	echo "module load mpi/openmpi/2.0-gnu-5.2" >> ./case_${val}.sh
	echo "module load lib/hdf5/1.8-openmpi-2.0-gnu-5.2" >> ./case_${val}.sh
	echo "module load numlib/mkl/11.3.4" >> ./case_${val}.sh
	echo "mpirun --bind-to core --map-by core -report-bindings Main/main Parameters/Rectangular/PWB/hom/PWB_Case_${val}.xml" >> ./case_${val}.sh
	chmod a+x ./case_${val}.sh
done

val=3
for value in {1..16}
do 
	val=$(expr $val + 2)
	echo msub -q multinode -l nodes=4:ppn=8,pmem=6000mb,walltime=04:00:00 case_${val}.sh
done
