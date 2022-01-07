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
  std::vector<std::vector<ComplexNumber>> evaluation_exact_solution;
  ConvergenceOutputGenerator output;
  std::vector<double> numerical_errors;
  std::vector<double> theoretical_errors;
  std::vector<unsigned int> total_cells;
  std::vector<double> h_values;
  std::vector<unsigned int> n_dofs_for_cases;
  unsigned int base_problem_n_dofs;
  unsigned int base_problem_n_cells;
  double base_problem_h;
  double base_problem_theoretical_error;

  double norming_factor = 1.0;
  
 public:
  ConvergenceRun();

  ~ConvergenceRun();

  void prepare() override;

  void run() override;

  void write_outputs();

  void prepare_transformed_geometry() override;

  void set_norming_factor();

  double compute_error_for_two_eval_vectors(std::vector<std::vector<ComplexNumber>> a, std::vector<std::vector<ComplexNumber>> b);
};
