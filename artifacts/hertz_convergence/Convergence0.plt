reset
set term png
set output "../Solutions/run186/Convergence0.png"
set title "Convergence History"
show title
array xvals[7]
xvals[1] = 11844
xvals[2] = 17404
xvals[3] = 31584
xvals[4] = 74200
xvals[5] = 103788
xvals[6] = 182404
xvals[7] = 232584
array yvalsnum[7]
yvalsnum[1] = 0.443371
yvalsnum[2] = 0.168291
yvalsnum[3] = 0.0894338
yvalsnum[4] = 0.0423629
yvalsnum[5] = 0.0381638
yvalsnum[6] = 0.0188396
yvalsnum[7] = 0
array yvalstheo[7]
yvalstheo[1] = 0.430107
yvalstheo[2] = 0.160704
yvalstheo[3] = 0.101312
yvalstheo[4] = 0.0598878
yvalstheo[5] = 0.0579333
yvalstheo[6] = 0.0399267
yvalstheo[7] = 0.0313593
set logscale y
set logscale x
set xlabel ''
set ylabel ''
set format y "%.1e"
plot sample [i=1:8] '+' using (xvals[i]):(yvalsnum[i]) with linespoints title "numerical error", [j=1:8] '+' using (xvals[j]):(yvalstheo[j]) with linespoints title "theoretical error"
