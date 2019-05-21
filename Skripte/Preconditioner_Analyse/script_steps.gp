reset
set xtics nomirror
set x2tics
set autoscale xfix
set autoscale x2fix
set xlabel 'Number of Processes'
set ylabel 'Number of Preconditioner Steps'
set x2label 'Cells per wavelength'
plot 'data.dat' using 1:2 with lines ps 2 lw 2 lt rgb "blue" notitle, \
'' using ($1*32)/100:2 axes x2y1 with points ps 2 lw 2 lt rgb "blue" notitle
