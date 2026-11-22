#pragma once

/**
 * @file ConvergenceRun.h
 * @author Pascal Kraft
 * @brief Contains the runner for convergence studies.
 */
#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"
#include "../OutputGenerators/Images/ConvergenceOutputGenerator.h"

/**
 * @brief Runner for a convergence study in the mesh size.
 *
 * The reference problem uses the parameter convergence_max_cells as the number of cells per direction and
 * subdomain. Its solution is evaluated at the cell centers of its level 0 mesh. The same problem is then solved for
 * the coarser cell counts in convergence_cell_counts. For each of them the solution at these points is compared
 * with the reference solution (numerical error) and with the exact solution (theoretical error).
 */
class ConvergenceRun: public Simulation {
  /** @brief The reference problem on the finest mesh. Deleted at the end of prepare(). */
  NonLocalProblem *mainProblem = nullptr;
  /** @brief The problem for the cell count that is currently computed in run(). */
  NonLocalProblem *otherProblem = nullptr;
  /** @brief Points at which all solutions are compared: the cell centers of the level 0 mesh of the reference problem on this process. */
  std::vector<Position> evaluation_positions;
  /** @brief The three field components of the reference solution at every evaluation position. */
  std::vector<std::vector<ComplexNumber>> evaluation_base_problem;
  /** @brief The three field components of the exact solution (the source field) at every evaluation position. */
  std::vector<std::vector<ComplexNumber>> evaluation_exact_solution;
  /** @brief Collects the errors for the convergence plot. */
  ConvergenceOutputGenerator output;
  /** @brief Difference between each coarse solution and the reference solution, one entry per cell count. */
  std::vector<double> numerical_errors;
  /** @brief Difference between each coarse solution and the exact solution, one entry per cell count. */
  std::vector<double> theoretical_errors;
  /** @brief Total number of cells of each coarse problem. */
  std::vector<unsigned int> total_cells;
  /** @brief Mesh size of each coarse problem, see NonLocalProblem::compute_h(). */
  std::vector<double> h_values;
  /** @brief Total number of dofs of each coarse problem. */
  std::vector<unsigned int> n_dofs_for_cases;
  /** @brief Total number of dofs of the reference problem. */
  unsigned int base_problem_n_dofs;
  /** @brief Total number of cells of the reference problem. */
  unsigned int base_problem_n_cells;
  /** @brief Mesh size of the reference problem. */
  double base_problem_h;
  /** @brief Difference between the reference solution and the exact solution. */
  double base_problem_theoretical_error;

  /** @brief Largest Euclidean norm of the reference solution over the evaluation positions, see set_norming_factor(). */
  double norming_factor = 1.0;
  
 public:

 /**
  * @brief Construct a new Convergence Run object
  * The constructor does nothing.
  */
  ConvergenceRun();

  /** @brief Deletes the problems that still exist. */
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
