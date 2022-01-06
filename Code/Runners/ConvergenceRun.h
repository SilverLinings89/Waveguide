#pragma once

#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"

class ConvergenceRun: public Simulation {
  NonLocalProblem *mainProblem;
  RectangularMode * rmProblem;
  NonLocalProblem *otherProblem;
  
 public:
  ConvergenceRun();

  virtual ~ConvergenceRun();

  void prepare() override;

  void run() override;

  void prepare_transformed_geometry() override;

};
