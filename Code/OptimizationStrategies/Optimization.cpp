#ifndef Optimization_CPP
#define Optimization_CPP
#include "../OptimizationStrategies/Optimization.h"

#include "../Core/Waveguide.h"
#include "../OutputGenerators/Console/GradientTable.h"

using namespace dealii;

Optimization::Optimization():
    pout(std::cout, GlobalParams.MPI_Rank==0) {

}

Optimization::~Optimization() {

}

#endif

