#pragma once

#include "./Simulation.h"
#include "../GlobalObjects/GeometryManager.h"
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

  void prepare_transformed_geometry() override;

};
