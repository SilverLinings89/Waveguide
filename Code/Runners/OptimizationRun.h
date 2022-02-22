#pragma once

#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include <functional>

class OptimizationRun: public Simulation {
  static NonLocalProblem *mainProblem;
  static std::vector<std::vector<double>> shape_dofs;
  static std::vector<std::vector<double>> shape_gradients;
  static unsigned int step_counter;
  std::function< double(const dealii::Vector<double> &x, dealii::Vector<double> &g)> function_pointer;
  const unsigned int n_free_dofs;
  
 
 public:
  OptimizationRun();

  virtual ~OptimizationRun();

  void prepare() override;

  void run() override;

  void prepare_transformed_geometry() override;

  double compute_step();

  static double perform_step(const dealii::Vector<double> &x, dealii::Vector<double> &g);

  static void solve_main_problem();

  static void set_shape_dofs(const dealii::Vector<double> in_shape_dofs);
};
