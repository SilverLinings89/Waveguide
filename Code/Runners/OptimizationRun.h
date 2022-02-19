#pragma once

#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"

class OptimizationRun: public Simulation {
  NonLocalProblem *mainProblem;
  std::vector<std::vector<double>> shape_dofs;
  std::vector<std::vector<double>> shape_gradients;
  unsigned int step_counter = 0;

 public:
  OptimizationRun();

  virtual ~OptimizationRun();

  void prepare() override;

  void run() override;

  void prepare_transformed_geometry() override;

  double compute_step();

  std::pair<double, std::vector<double>> OptimizationRun::perform_step(std::vector<double> x);

  void solve_main_problem();

  void set_shape_dofs(std::vector<double> in_shape_dofs);
};
