reset
array series[5]
series[1] = 1
series[2] = 0.1
series[3] = 0.01
series[4] = 0.001
series[5] = 0.0001
set logscale y
set autoscale xfix
set xlabel 'Step'
set ylabel 'Residual'
plot series with linespoints ps 2 lw 2 lt rgb "blue" title "run1"
