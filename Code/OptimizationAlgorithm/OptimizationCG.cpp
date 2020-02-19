#ifndef OPTIMIZATION_CG_CPP
#define OPTIMIZATION_CG_CPP

#include "OptimizationCG.h"

OptimizationCG::OptimizationCG() {}

OptimizationCG::~OptimizationCG() {}

void OptimizationCG::pass_result_small_step(std::vector<double>) {
  // TODO: This implementation is still missing - uncertain and unimportant.
  return;
}

void OptimizationCG::pass_result_big_step(double) {
  // TODO: implement this function as core functionality of CG-based stepping
  // scheme.
  return;
}

bool OptimizationCG::perform_small_step_next(int) {
  // TODO: implement this function as core functionality of CG-based stepping
  // scheme.
  return false;
}

double OptimizationCG::get_small_step_step_width(int) {
  // TODO: implement this function as core functionality of CG-based stepping
  // scheme.
  return 0.0;
}

bool OptimizationCG::perform_big_step_next(int) {
  // TODO: implement this function as core functionality of CG-based stepping
  // scheme.
  return false;
}

std::vector<double> OptimizationCG::get_big_step_configuration() {
  std::vector<double> ret;
  // TODO: implement this function as core functionality of CG-based stepping
  // scheme.
  return ret;
}

#endif
