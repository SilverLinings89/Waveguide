reset
set term png
set output "../Solutions/run186/ConvergenceHistoryLevel10.png"
array series0[7]
series0[1] = 20587.3
series0[2] = 7.33914
series0[3] = 0.862946
series0[4] = 0.0525358
series0[5] = 0.00178838
series0[6] = 5.50631e-05
series0[7] = 1.85628e-06
set title "Convergence History on level 1"
set logscale y
set xlabel 'Step'
set ylabel 'Residual'
set format y "%.1e"
plot series0 with linespoints title"Run 1"
