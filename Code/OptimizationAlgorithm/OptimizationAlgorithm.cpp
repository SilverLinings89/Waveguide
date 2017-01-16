#ifndef OPTIMIZATION_ALGORITHM_CPP
#define OPTIMIZATION_ALGORITHM_CPP

#include "OptimizationAlgorithm.h"

OptimizationAlgorithm::OptimizationAlgorithm () {

}

OptimizationAlgorithm::~OptimizationAlgorithm() {

}

void OptimizationAlgorithm::pass_gradient(std::vector<double> in_gradient) {
  states.push_back(in_gradient);
  return;
}

void OptimizationAlgorithm::pass_residual(double in_residual) {
  residuals.push_back(in_residual);
  return;
}

#endif
