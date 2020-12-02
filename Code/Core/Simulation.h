#pragma once

#include "../Helpers/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"

class Simulation {
  HierarchicalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  Simulation(const std::string run_file,const std::string case_file);

  virtual ~Simulation();

  void prepare();

  void run();

  void overwrite_parameters_from_console();

  void prepare_transformed_geometry();

  void create_output_directory();
};
