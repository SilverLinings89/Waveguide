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

 /**
  * @brief Construct a new Convergence Run object
  * The constructor does nothing.
  */
  ConvergenceRun();

  ~ConvergenceRun();

  /**
   * @brief Solve the reference problem and setup the others.
   * In a convergence run we have the reference solution on the finest grid and then a set of other sizes as the actual data. This function solves the reference problem and prepares the others.
   * 
   */
  void prepare() override;

  /**
   * @brief Solves the coarser problems and computes their theoretical and numerical error.
   * Then calls write_outputs().
   * 
   */
  void run() override;

  /**
   * @brief Writes the results of the convergence study to the command line.
   * 
   */
  void write_outputs();

  /**
   * @brief Not implemented / not required here.
   * 
   */
  void prepare_transformed_geometry() override;

  /**
   * @brief Computes and stores the max vector component of the reference solutions norm.
   * 
   */
  void set_norming_factor();

  /**
   * @brief Computes the L2 difference of two solutions, i.e. the reference solution and another one.
   * As a consequence the order of the provided vectors does not matter.
   * 
   * @param a first solution vector
   * @param b other solution vector
   * @return double L2 norm of the difference.
   */
  double compute_error_for_two_eval_vectors(std::vector<std::vector<ComplexNumber>> a, std::vector<std::vector<ComplexNumber>> b);
};
