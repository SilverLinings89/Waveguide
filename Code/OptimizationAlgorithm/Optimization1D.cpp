#ifndef OPTIMIZATION_CPP
#define OPTIMIZATION_CPP

#include "Optimization1D.h"

Optimization1D::Optimization1D( ) {
  steps_widths = new double[5];
  steps_widths[0] = 0.00001;
  steps_widths[1] = 0.0001;
  steps_widths[2] = 0.001;
  steps_widths[3] = 0.01;
  steps_widths[4] = 0.1;

}

Optimization1D::~Optimization1D(){
  delete steps_widths;
}

std::vector<double> Optimization1D::get_configuration() {
  std::vector<double> ret(0);
  // TODO: This requires more complex implementation because it does not depend on old values but only on the current state.
  return ret;
}

bool Optimization1D::perform_small_step_next(int small_steps_before ){
  if (small_steps_before < 5) {
    return true;
  } else {
    return false;
  }
}

double Optimization1D::get_small_step_step_width(int small_steps_before ){
  if(small_steps_before < 5 && small_steps_before >= 0) {
    return steps_widths[small_steps_before];
  } else {
    std::cout<< "Warning in Optimization1D::get_small_step_step_width(int)" <<std::endl;
    return -1.0;
  }
}

bool Optimization1D::perform_small_big_next(int small_steps_before ) {
  return (small_steps_before >=5);
}

std::vector<double> Optimization1D::get_big_step_configuration(){
  std::vector<double> ret;
  int small_step_count = states.size() ;
  int big_step_count = residuals.size() ;
  if( big_step_count == 0 || (small_step_count != 5* big_step_count ) ) {
    std::cout << "Warning in Optimization1D::get_big_step_configuration()" <<std::endl;
  } else {
    std::complex<double> residual = residuals[big_step_counter-1];
    int ndofs = states[small_step_count-1].size();
    ret.reserve(ndofs);
    for ( unsigned int i = 0; i < ndofs; i++ ) {
      double max = 0;
      int index =-1;
      for(unsigned int j = 0; j<5; j++) {
        double magn = std::abs(residual + states[small_step_count - 5 + j][i]);
        if(magn > max ){
          max = magn;
          index = j;
        }
      }
      ret[i] = states[small_step_count - 5 + index][i];
    }
  }
  return ret;
}

#endif
