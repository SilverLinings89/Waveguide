#ifndef OPTIMIZATION_CPP
#define OPTIMIZATION_CPP

#include "Optimization1D.h"

Optimization1D::Optimization1D( ) {


}

void Optimization1D::pass_gradient(std::vector<double> in_gradient) {
  states.push_back(in_gradient);
  return;
}

std::vector<double> Optimization1D::get_configuration() {
  std::vector<double> ret(0);
  // TODO: This requires more complex implementation because it does not depend on old values but only on the current state.
  return ret;
}

void Optimization1D::pass_residual(double in_residual) {
  residuals.push_back(in_residual);
  return;
}
#endif
