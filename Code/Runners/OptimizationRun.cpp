/*
 * Simulation.cpp
 *
 *  Created on: Jun 24, 2019
 *      Author: kraft
 */

#include <mpi.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "Simulation.h"
#include "OptimizationRun.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/optimization/solver_bfgs.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/base/exceptions.h>
#include "../Helpers/staticfunctions.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"
#include <functional>

NonLocalProblem * OptimizationRun::mainProblem;
std::vector<std::vector<double>> OptimizationRun::shape_dofs;
std::vector<std::vector<double>> OptimizationRun::shape_gradients;
unsigned int OptimizationRun::step_counter;
bool OptimizationRun::stopping_criteria_active = false;
double OptimizationRun::initial_gradient_norm = -1;
double OptimizationRun::best_loss = 0;
std::vector<double> OptimizationRun::best_shape_dofs;
unsigned int OptimizationRun::evaluations_without_improvement = 0;
std::vector<double> OptimizationRun::loss_history;
std::vector<double> OptimizationRun::gradient_norm_history;

/**
 * Thrown by perform_step to leave the BFGS solver of deal.II once one of the stopping criteria is fulfilled.
 */
class OptimizationFinished : public std::exception {
 public:
  explicit OptimizationFinished(const std::string & in_reason) : reason(in_reason) {}
  const char * what() const noexcept override { return reason.c_str(); }
  std::string reason;
};

OptimizationRun::OptimizationRun():
  n_free_dofs(GlobalSpaceTransformation->n_free_dofs())
  {
  function_pointer = &OptimizationRun::perform_step;
  OptimizationRun::step_counter = 0; 
}

OptimizationRun::~OptimizationRun() {
  delete mainProblem;
}

void OptimizationRun::prepare() {
  print_info("OptimizationRun::prepare", "Start", LoggingLevel::DEBUG_ONE);
  
  OptimizationRun::mainProblem = new NonLocalProblem(GlobalParams.Sweeping_Level);
  OptimizationRun::mainProblem->initialize();
  
  print_info("OptimizationRun::prepare", "End", LoggingLevel::DEBUG_ONE);
}

void OptimizationRun::run() {
  print_info("OptimizationRun::run", "Start", LoggingLevel::PRODUCTION_ONE);
  const unsigned int n_shape_dofs = n_free_dofs;
  dealii::Vector<double> shape_dofs(n_shape_dofs);
  OptimizationRun::step_counter = 0;
  for(unsigned int i = 0; i < n_shape_dofs; i++) {
    shape_dofs[i] = GlobalSpaceTransformation->get_free_dof(i);
    if(GlobalParams.MPI_Rank == 0) {
      std::cout << "Shape dof " << i << ": " << shape_dofs[i] << std::endl; 
    }
  }
  if(GlobalParams.Perform_gradient_check) {
    run_gradient_check(shape_dofs);
    GlobalTimerManager.write_output();
    print_info("OptimizationRun::run", "End", LoggingLevel::PRODUCTION_ONE);
    return;
  }
  // The stopping criteria are evaluated in perform_step: a relative tolerance for the gradient norm, stagnation of
  // the loss and a maximum number of evaluations. The tolerance of the solver control of deal.II is an absolute
  // gradient norm and only serves as a fallback, its maximum number of steps limits the BFGS steps.
  dealii::SolverControl sc(GlobalParams.optimization_n_shape_steps, GlobalParams.optimization_residual_tolerance, false, false);
  dealii::SolverBFGS<dealii::Vector<double>> solver(sc);
  std::string reason = "The gradient norm is below the absolute tolerance.";
  OptimizationRun::stopping_criteria_active = true;
  try{
    solver.solve(function_pointer, shape_dofs);
  } catch(OptimizationFinished & e) {
    reason = e.what();
  } catch(dealii::SolverControl::NoConvergence & e) {
    reason = "The maximum number of BFGS steps (" + std::to_string(GlobalParams.optimization_n_shape_steps) + ") was reached.";
  } catch(dealii::StandardExceptions::ExcMessage & e) {
    reason = "The BFGS solver stopped with a message.";
  }
  OptimizationRun::stopping_criteria_active = false;
  print_info("OptimizationRun::run", "Optimization finished after " + std::to_string(OptimizationRun::loss_history.size()) + " evaluations. " + reason, LoggingLevel::PRODUCTION_ONE);
  print_info("OptimizationRun::run", "Loss: " + std::to_string(OptimizationRun::loss_history[0]) + " at the initial shape, " + std::to_string(OptimizationRun::best_loss) + " at the best shape.", LoggingLevel::PRODUCTION_ONE);
  if(GlobalParams.MPI_Rank == 0) {
    std::ofstream out(GlobalOutputManager.get_full_filename("optimization_history.txt"));
    out.precision(10);
    out << "# reason: " << reason << std::endl;
    out << "# evaluation loss gradient_norm shape_parameters..." << std::endl;
    for(unsigned int i = 0; i < OptimizationRun::loss_history.size(); i++) {
      out << i << " " << OptimizationRun::loss_history[i] << " " << OptimizationRun::gradient_norm_history[i];
      for(unsigned int j = 0; j < OptimizationRun::shape_dofs[i].size(); j++) {
        out << " " << OptimizationRun::shape_dofs[i][j];
      }
      out << std::endl;
    }
  }

  // The last evaluation is not necessarily the best one (the line search also tries worse shapes). Compute the
  // fields of the best shape again, so that the outputs belong to the result of the optimization.
  dealii::Vector<double> best(n_shape_dofs);
  for(unsigned int i = 0; i < n_shape_dofs; i++) {
    best[i] = OptimizationRun::best_shape_dofs[i];
  }
  OptimizationRun::set_shape_dofs(best);
  OptimizationRun::mainProblem->assemble();
  OptimizationRun::mainProblem->compute_solver_factorization();
  OptimizationRun::mainProblem->solve_with_timers_and_count();

  GlobalTimerManager.write_output();
  OptimizationRun::mainProblem->output_results();
  print_info("OptimizationRun::run", "End", LoggingLevel::PRODUCTION_ONE);
}

void OptimizationRun::run_gradient_check(const dealii::Vector<double> & shape_dofs) {
  const unsigned int n = shape_dofs.size();
  const double h = GlobalParams.gradient_check_step_width;
  dealii::Vector<double> gradient(n), unused(n);
  const double loss = OptimizationRun::perform_step(shape_dofs, gradient);
  std::vector<double> finite_differences(n);
  for(unsigned int i = 0; i < n; i++) {
    dealii::Vector<double> plus = shape_dofs, minus = shape_dofs;
    plus[i] += h;
    minus[i] -= h;
    const double loss_plus = OptimizationRun::perform_step(plus, unused);
    const double loss_minus = OptimizationRun::perform_step(minus, unused);
    finite_differences[i] = (loss_plus - loss_minus) / (2 * h);
  }
  if(GlobalParams.MPI_Rank == 0) {
    std::ofstream out(GlobalOutputManager.get_full_filename("gradient_check.txt"));
    out.precision(10);
    out << "loss = " << loss << std::endl;
    out << "step_width = " << h << std::endl;
    std::cout << "Gradient check at the initial shape (loss " << loss << ", step width " << h << "):" << std::endl;
    std::cout << "  parameter | value | adjoint gradient | finite difference | ratio" << std::endl;
    for(unsigned int i = 0; i < n; i++) {
      out << "adjoint_gradient_" << i << " = " << gradient[i] << std::endl;
      out << "finite_difference_" << i << " = " << finite_differences[i] << std::endl;
      std::cout << "  " << i << " | " << shape_dofs[i] << " | " << gradient[i] << " | " << finite_differences[i] << " | " << gradient[i] / finite_differences[i] << std::endl;
    }
  }
}

void OptimizationRun::prepare_transformed_geometry() {
}

void OptimizationRun::solve_main_problem() {
  OptimizationRun::mainProblem->assemble();
  OptimizationRun::mainProblem->compute_solver_factorization();
  OptimizationRun::mainProblem->solve_with_timers_and_count();
  OptimizationRun::mainProblem->set_rhs_for_adjoint_problem();
  OptimizationRun::mainProblem->solve_adjoint();
}

double OptimizationRun::perform_step(const dealii::Vector<double> & x, dealii::Vector<double> & g) {
  std::vector<double> x_vec(x.size());
  for(unsigned int i = 0; i < x.size(); i++) {
    x_vec[i] = x[i];
  }
  OptimizationRun::shape_dofs.push_back(x_vec);
  OptimizationRun::set_shape_dofs(x);
  OptimizationRun::solve_main_problem();
  double loss_functional_evaluation = -std::abs(mainProblem->compute_signal_strength_of_solution());
  print_info("OptimizationRun::perform_step", "Loss functional in step " + std::to_string(OptimizationRun::step_counter) + ": " + std::to_string(loss_functional_evaluation));
  std::vector<double> shape_grad = mainProblem->compute_shape_gradient();
  OptimizationRun::shape_gradients.push_back(shape_grad);
  std::string msg = "Shape gradient: ( ";
  for(unsigned int i = 0; i < g.size(); i++) {
    g[i] = shape_grad[i];
    msg += std::to_string(g[i]);
    if(i < g.size() -1) {
      msg += ", ";
    } else {
      msg += ")";
    }
  }
  print_info("OptimizationRun::perform_step", msg);
  OptimizationRun::step_counter += 1;

  if(OptimizationRun::stopping_criteria_active) {
    double gradient_norm = 0;
    for(unsigned int i = 0; i < g.size(); i++) {
      gradient_norm += g[i] * g[i];
    }
    gradient_norm = std::sqrt(gradient_norm);
    const bool is_first = OptimizationRun::loss_history.size() == 0;
    OptimizationRun::loss_history.push_back(loss_functional_evaluation);
    OptimizationRun::gradient_norm_history.push_back(gradient_norm);
    if(is_first) {
      OptimizationRun::initial_gradient_norm = gradient_norm;
    }
    // "Improved" means: better than the best loss so far by more than the loss tolerance.
    const bool improved = is_first || (loss_functional_evaluation < OptimizationRun::best_loss - GlobalParams.optimization_loss_tolerance * std::abs(OptimizationRun::best_loss));
    if(is_first || loss_functional_evaluation < OptimizationRun::best_loss) {
      OptimizationRun::best_loss = loss_functional_evaluation;
      OptimizationRun::best_shape_dofs = x_vec;
    }
    if(improved) {
      OptimizationRun::evaluations_without_improvement = 0;
    } else {
      OptimizationRun::evaluations_without_improvement++;
    }
    if(gradient_norm <= GlobalParams.optimization_gradient_tolerance * OptimizationRun::initial_gradient_norm) {
      throw OptimizationFinished("The norm of the shape gradient (" + std::to_string(gradient_norm) + ") is below " + std::to_string(GlobalParams.optimization_gradient_tolerance) + " times its initial value (" + std::to_string(OptimizationRun::initial_gradient_norm) + ").");
    }
    if(OptimizationRun::evaluations_without_improvement >= 4) {
      throw OptimizationFinished("The loss did not improve in four consecutive evaluations.");
    }
    if(OptimizationRun::loss_history.size() >= GlobalParams.optimization_max_evaluations) {
      throw OptimizationFinished("The maximum number of evaluations (" + std::to_string(GlobalParams.optimization_max_evaluations) + ") was reached.");
    }
  }
  return loss_functional_evaluation;
}

void OptimizationRun::set_shape_dofs(const dealii::Vector<double> in_shape_dofs) {
  std::string msg = "( ";
  for(unsigned int i = 0; i < in_shape_dofs.size(); i++) {
    msg += std::to_string(in_shape_dofs[i]);
    if(i != in_shape_dofs.size() - 1) {
      msg += ", ";
    } else {
      msg += ")";
    }
  }
  print_info("OptimizationRun::set_shape_dofs", msg);

  for(unsigned int i = 0; i < in_shape_dofs.size(); i++) {
    GlobalSpaceTransformation->set_free_dof(i, in_shape_dofs[i]);
  }

}

