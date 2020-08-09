#pragma once

#include "../Helpers/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"

class Simulation {
  HierarchicalProblem *mainProblem;

 public:
  Simulation();

  virtual ~Simulation();

  void prepare();

  void run();

  void overwrite_parameters_from_console();

  void prepare_transformed_geometry();

  void create_output_directory();
};
