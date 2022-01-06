#pragma once

#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"
#include "../OutputGenerators/Images/ConvergenceOutputGenerator.h"

class ConvergenceRun: public Simulation {
  NonLocalProblem *mainProblem;
  NonLocalProblem *otherProblem;
  std::vector<Position> evaluation_positions;
  std::vector<std::vector<ComplexNumber>> evaluation_base_problem;
  ConvergenceOutputGenerator output;
  
 public:
  ConvergenceRun();

  ~ConvergenceRun();

  void prepare() override;

  void run() override;

  void prepare_transformed_geometry() override;

  double compute_error_for_run();
};
