#!/bin/bash
doxygen "./Doxygen/Doxyfile"
cd Documentation/latex
make pdf
mv refman.pdf ../WaveguideSolverDocumentation.pdf
