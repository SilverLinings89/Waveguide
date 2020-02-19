#ifndef Optimization1D_H_
#define Optimization1D_H_

#include "../Core/NumericProblem.h"
#include "OptimizationAlgorithm.h"

/**
 * \class Optimization1D
 * \brief This class implements the computation of an optimization step by doing
 * 1D optimization based on an adjoint scheme.
 *
 * Objects of the Type OptimizationAlgorithm are used by the class
 * OptimizationStrategy to compute the next viable configuration based on former
 * results. Its is encapsulated in it's own class to offer comparison and easy
 * changing between differenct schemes. \author Pascal Kraft \date 9.1.2017
 */
class Optimization1D : public OptimizationAlgorithm<std::complex<double>> {
 public:
  Optimization1D();

  ~Optimization1D();

  bool perform_small_step_next(int small_steps_before);

  double get_small_step_step_width(int small_steps_before);

  bool perform_big_step_next(int small_steps_before);

  std::vector<double> get_big_step_configuration();

  // double * steps_widths;
};

#endif
