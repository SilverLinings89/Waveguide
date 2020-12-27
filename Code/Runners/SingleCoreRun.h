#pragma once

#include "../Helpers/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"

class SingleCoreRun: public Simulation {
  LocalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  SingleCoreRun();

  virtual ~SingleCoreRun();

  void prepare() override;

  void run() override;

  void prepare_transformed_geometry() override;

};
