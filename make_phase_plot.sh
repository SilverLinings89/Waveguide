#!/bin/bash

echo "set terminal pngcairo
set output 'powerplot.png'
set key autotitle columnheader
set style data lines
set xlabel 'z in \mu m'
plot 'complex_qualities.dat' using 1:2, '' using 1:3, '' using 1:4" | gnuplot
