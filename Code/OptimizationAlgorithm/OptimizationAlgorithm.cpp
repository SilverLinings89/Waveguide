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

void OptimizationAlgorithm::pass_full_step(double in_residual, std::vector<double> in_configuration) {
  states.push_back(in_configuration);
  residuals.push_back(in_residual);
}

void OptimizationAlgorithm::pass_residual(double in_residual) {
  residuals.push_back(in_residual);
  return;
}

#endif
