#!/bin/bash

echo 'set terminal pngcairo
filter(x)=(x<=1000.0)?(x):(0)
set output '\''powerplot.png'\''
set key autotitle columnheader
set style data lines
set xlabel '\''z in \mu m'\''
plot '\'$1\'' using 1:(filter($2)), '\'''\'' using 1:(filter($3)), '\'''\'' using 1:(filter($4))' | gnuplot
