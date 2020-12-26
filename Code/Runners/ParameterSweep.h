#pragma once

#include "./Simulation.h"
#include "../Helpers/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"

class ParameterSweep: public Simulation {
  LocalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  ParameterSweep();

  ~ParameterSweep();

  void prepare() override;

  void run() override;

  void overwrite_parameters_from_console() override;

  void prepare_transformed_geometry() override;

};
