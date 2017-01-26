#ifndef OPTIMIZATION_STEEPEST_DESCENT_CPP
#define OPTIMIZATION_STEEPEST_DESCENT_CPP

#include "OptimizationSteepestDescent.h"

OptimizationSteepestDescent::OptimizationSteepestDescent() {

}

OptimizationSteepestDescent::~OptimizationSteepestDescent() {

}

void OptimizationSteepestDescent::pass_result_small_step(std::vector<double>) {
  // TODO: This implementation is still missing - uncertain and unimportant.
  return;
}

void OptimizationSteepestDescent::pass_result_big_step(double) {
  // TODO: implement this function as core functionality of CG-based stepping scheme.
  return;
}


bool OptimizationSteepestDescent::perform_small_step_next( int small_steps_before ) {
  // TODO: implement this function as core functionality of CG-based stepping scheme.
  return false;
}

double OptimizationSteepestDescent::get_small_step_step_width( int small_steps_before ) {
  // TODO: implement this function as core functionality of CG-based stepping scheme.
  return 0.0;
}

bool OptimizationSteepestDescent::perform_big_step_next( int small_steps_before )  {
  // TODO: implement this function as core functionality of CG-based stepping scheme.
  return false;
}

std::vector<double> OptimizationSteepestDescent::get_big_step_configuration() {
  std::vector<double> ret;
  // TODO: implement this function as core functionality of CG-based stepping scheme.
  return ret;
}
#endif
