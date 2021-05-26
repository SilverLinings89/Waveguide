#pragma once

#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"

class Simulation {
  HierarchicalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  Simulation();

  virtual ~Simulation();

  virtual void prepare() = 0;

  virtual void run() = 0;

  virtual void prepare_transformed_geometry() = 0;

  void create_output_directory();
};
