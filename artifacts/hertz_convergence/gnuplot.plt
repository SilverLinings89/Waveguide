set xrange [0.005:0.5]
set logscale x
set logscale y
f(x)=5*x**1.2
plot 'data' u 5:2 w l title "Numerical", 'data' u 5:3 w l title "Theoretical" , f(x) w l title "h"
set xlabel "h"
