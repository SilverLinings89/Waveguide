#!/bin/bash
StepWidth=1
val=35 #The first computed Case is val + StepWidth and Zerobased so the line computed from the shapelist is linenumber = val+stepwidth-1
SaveVal=${val}
NumberOfCases=21
for value in $(seq 1 $NumberOfCases)
do 
	val=$(expr $val + $StepWidth)
	rm ./case_${val}.sh
	rm Parameters/Rectangular/PWB/hom/PWB_Case_${val}.xml
	touch Parameters/Rectangular/PWB/hom/PWB_Case_${val}.xml
	sed "s/#CASENUMBER#/${val}/g" Parameters/Rectangular/PWB/hom/PWB_Case_${val}.xml > Parameters/Rectangular/PWB/hom/PWB_Case_${val}.xml
	echo "#!/bin/bash" >> ./case_${val}.sh
	echo "module load lib/boost/1.56.0" >> ./case_${val}.sh
	echo "module load compiler/gnu/5.2" >> ./case_${val}.sh
	echo "module load mpi/openmpi/2.0-gnu-5.2" >> ./case_${val}.sh
	echo "module load lib/hdf5/1.8-openmpi-2.0-gnu-5.2" >> ./case_${val}.sh
	echo "module load numlib/mkl/11.3.4" >> ./case_${val}.sh
	echo "mpirun --bind-to core --map-by core -report-bindings Main/main Parameters/Rectangular/PWB/hom/PWB_Case_${val}.xml" >> ./case_${val}.sh
	chmod a+x ./case_${val}.sh
done

val=${SaveVal}
for value in $(seq 1 $NumberOfCases)
do 
	val=$(expr $val + $StepWidth)
	echo msub -q multinode -l nodes=4:ppn=8,pmem=6000mb,walltime=04:00:00 case_${val}.sh
done
