reset
set xlabel 'Number of Processes [1]'
set ylabel 'Time [s]'
plot 'data.dat' using 1:3 with linespoints title "Assemble", \
'' using 1:4 with linespoints title "Preconditioner Initialization", \
'' using 1:5 with linespoints title "Solve"
