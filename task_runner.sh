 #!/bin/bash
 
 counter=1
 while [ $counter -lt 1025 ]; do\
	temp="0$(bc<<<"scale=5;$counter/10000")"
	mpirun -np 4 ./main Parameters/Parameters.xml $temp
	counter=$((counter *2)) 
done
