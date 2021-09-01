#pragma once

#include "../GlobalObjects/GeometryManager.h"
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

  void prepare_transformed_geometry() override;

};