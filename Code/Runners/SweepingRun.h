#pragma once

#include "../Helpers/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"

class SweepingRun: public Simulation {
  NonLocalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  SweepingRun();

  virtual ~SweepingRun();

  void prepare() override;

  void run() override;

  void overwrite_parameters_from_console() override;

  void prepare_transformed_geometry() override;

};
