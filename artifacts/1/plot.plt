set terminal svg
set output "plot.svg"
set key autotitle columnhead 
f(x) = a*x**2 + b*x + c
g(x) = a2*x + b2
fit f(x) "./data.dat" using 1:3 via a,b,c
fit g(x) "./data.dat" using 1:2 via a2, b2
set xlabel "Number of Subdomains"
set y2tics
set y2label "Seconds"
set ylabel "Steps"
set ytics nomirror
set grid
set xrange [1:300]
set yrange [0:40]
set y2range [0:20000] 
set key left top
plot './data.dat' using 1:2 axis x1y1, './data.dat' using 1:3 axis x1y2, f(x) with l axis x1y2 title 'f(x)=0.25x^2+9x+800', g(x) with l axis x1y1 title 'g(x)=0.09x+5.34'