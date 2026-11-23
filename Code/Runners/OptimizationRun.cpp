/*
 * Simulation.cpp
 *
 *  Created on: Jun 24, 2019
 *      Author: kraft
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <mpi.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "Simulation.h"
#include "OptimizationRun.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/optimization/solver_bfgs.h>
#include <deal.II/optimization/line_minimization.h>
#include <deal.II/numerics/history.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/base/exceptions.h>
#include "../Helpers/staticfunctions.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"
#include "../OutputGenerators/RunReport.h"
#include <functional>

NonLocalProblem * OptimizationRun::mainProblem;
std::vector<std::vector<double>> OptimizationRun::shape_dofs;
std::vector<std::vector<double>> OptimizationRun::shape_gradients;
unsigned int OptimizationRun::step_counter;
bool OptimizationRun::stopping_criteria_active = false;
double OptimizationRun::initial_gradient_norm = -1;
double OptimizationRun::best_loss = 0;
std::vector<double> OptimizationRun::best_shape_dofs;
std::vector<double> OptimizationRun::loss_history;
std::vector<double> OptimizationRun::gradient_norm_history;

// The vertical displacement f(z) of every evaluated shape at equidistant z, for the outputs.
static std::vector<std::vector<double>> sampled_shapes;
static const unsigned int n_shape_samples = 201;

static std::vector<double> sample_current_shape() {
  std::vector<double> ret(n_shape_samples);
  for(unsigned int i = 0; i < n_shape_samples; i++) {
    const double z = Geometry.global_z_range.first + (Geometry.global_z_range.second - Geometry.global_z_range.first) * i / (n_shape_samples - 1.0);
    ret[i] = GlobalSpaceTransformation->math_to_phys(Position(0, 0, z))[1];
  }
  return ret;
}

/**
 * Thrown by perform_step to leave the BFGS solver of deal.II once one of the stopping criteria is fulfilled.
 */
class OptimizationFinished : public std::exception {
 public:
  /**
   * @brief Stores the reason.
   *
   * @param in_reason Text that names the stopping criterion that was fulfilled.
   */
  explicit OptimizationFinished(const std::string & in_reason) : reason(in_reason) {}
  /**
   * @brief Access to the reason in the form required by std::exception.
   *
   * @return const char* The text passed to the constructor.
   */
  const char * what() const noexcept override { return reason.c_str(); }
  /** @brief Text that names the stopping criterion that was fulfilled. It is written to the console and the run report. */
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
  // Stopping criteria: a relative tolerance for the gradient norm and a maximum number of evaluations (both in
  // perform_step) and stagnation of the loss (in the line search below). The tolerance of the solver control of
  // deal.II is an absolute gradient norm and only serves as a fallback, its maximum number of steps limits the
  // BFGS steps.
  dealii::SolverControl sc(GlobalParams.optimization_n_shape_steps, GlobalParams.optimization_residual_tolerance, false, false);
  dealii::SolverBFGS<dealii::Vector<double>> solver(sc);
  // Two settings of deal.II's BFGS are replaced, because they made the optimization stall close to the optimum:
  // 1. The first trial step of a line search. deal.II derives it from the decrease of the loss in the previous
  //    step (min(1, 2 df / |g p|)). Close to the optimum this decrease is tiny, so the line search starts with a
  //    tiny step and needs many evaluations to enlarge it. For a quasi-Newton direction the natural trial step is
  //    1, which is exact for a quadratic loss once the curvature is known.
  // 2. The initial approximation of the inverse Hessian. deal.II uses the identity. The usual scaling
  //    (s, y) / (y, y) with the last step s and the last change of the gradient y gives directions of the right
  //    length.
  unsigned int steps_without_improvement = 0;
  solver.connect_line_search_slot([&](double & f, dealii::Vector<double> & x, dealii::Vector<double> & g, const dealii::Vector<double> & p) -> double {
    const double f0 = f;
    const double g0 = g * p;
    const dealii::Vector<double> x0 = x;
    const auto line_function = [&](const double & step) -> std::pair<double, double> {
      x = x0;
      x.add(step, p);
      f = OptimizationRun::perform_step(x, g);
      return std::make_pair(f, g * p);
    };
    const auto result = dealii::LineMinimization::line_search<double>(line_function, f0, g0, dealii::LineMinimization::poly_fit<double>, 1.0, 0.9, 0.001);
    // Stagnation is judged per BFGS step and not per evaluation, because a line search may need several
    // evaluations before it finds a good step.
    if(f0 - f <= GlobalParams.optimization_loss_tolerance * std::abs(f0)) {
      steps_without_improvement++;
    } else {
      steps_without_improvement = 0;
    }
    if(steps_without_improvement >= 2) {
      throw OptimizationFinished("Two consecutive BFGS steps improved the loss by less than " + RunReport::number(GlobalParams.optimization_loss_tolerance) + " times its value.");
    }
    return result.first;
  });
  solver.connect_preconditioner_slot([](dealii::Vector<double> & g, const dealii::FiniteSizeHistory<dealii::Vector<double>> & s, const dealii::FiniteSizeHistory<dealii::Vector<double>> & y) {
    if(s.size() > 0) {
      g *= (s[0] * y[0]) / (y[0] * y[0]);
    }
  });
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

  if(GlobalParams.MPI_Rank == 0) {
    // The shapes of all evaluations: first line z, then one line f(z) per evaluation.
    std::ofstream shapes(GlobalOutputManager.get_full_filename("optimization_shapes.txt"));
    shapes.precision(10);
    shapes << "# vertical displacement f(z) of the waveguide axis; first line: z, then one line per evaluation" << std::endl;
    for(unsigned int i = 0; i < n_shape_samples; i++) {
      shapes << (i > 0 ? " " : "") << Geometry.global_z_range.first + (Geometry.global_z_range.second - Geometry.global_z_range.first) * i / (n_shape_samples - 1.0);
    }
    shapes << std::endl;
    for(const std::vector<double> & one_shape : sampled_shapes) {
      for(unsigned int i = 0; i < one_shape.size(); i++) shapes << (i > 0 ? " " : "") << one_shape[i];
      shapes << std::endl;
    }
    unsigned int best_evaluation = 0;
    for(unsigned int i = 0; i < OptimizationRun::loss_history.size(); i++) {
      if(OptimizationRun::loss_history[i] < OptimizationRun::loss_history[best_evaluation]) best_evaluation = i;
    }
    RunReport & report = GlobalRunReport;
    report.section("Optimization");
    report.entry("Loss functional", "-|s|, the negative signal transmission");
    report.entry("Algorithm", "BFGS (deal.II SolverBFGS), gradient from one adjoint solve per evaluation");
    report.entry("Shape parameters", std::to_string(n_shape_dofs) + " (slopes of the vertical displacement at the inner nodes)");
    report.entry("Evaluations", std::to_string(OptimizationRun::loss_history.size()) + " (each one forward and one adjoint solve)");
    report.entry("Reason for stopping", reason);
    report.entry("Signal transmission at the initial shape", RunReport::number(-OptimizationRun::loss_history[0], 8));
    report.entry("Signal transmission at the best shape", RunReport::number(-OptimizationRun::best_loss, 8) + " (evaluation " + std::to_string(best_evaluation) + ")");
    std::vector<std::vector<std::string>> rows;
    for(unsigned int i = 0; i < OptimizationRun::loss_history.size(); i++) {
      std::string parameters;
      for(unsigned int j = 0; j < OptimizationRun::shape_dofs[i].size(); j++) {
        parameters += (j > 0 ? ", " : "") + RunReport::number(OptimizationRun::shape_dofs[i][j], 5);
      }
      double largest_displacement = 0;
      if(i < sampled_shapes.size()) {
        for(double value : sampled_shapes[i]) largest_displacement = std::max(largest_displacement, std::abs(value));
      }
      std::string gradient;
      if(i < OptimizationRun::shape_gradients.size()) {
        for(unsigned int j = 0; j < OptimizationRun::shape_gradients[i].size(); j++) {
          gradient += (j > 0 ? ", " : "") + RunReport::number(OptimizationRun::shape_gradients[i][j], 4);
        }
      }
      rows.push_back({std::to_string(i), RunReport::number(-OptimizationRun::loss_history[i], 8), RunReport::number(OptimizationRun::gradient_norm_history[i], 4), RunReport::number(largest_displacement, 4), parameters, gradient});
    }
    report.table("Evaluations", {"evaluation", "signal |s|", "norm of the gradient", "largest |f(z)| [um]", "shape parameters", "gradient of the loss"}, rows);
    std::ofstream gradients(GlobalOutputManager.get_full_filename("optimization_gradients.txt"));
    gradients.precision(10);
    gradients << "# evaluation gradient of the loss with respect to the shape parameters..." << std::endl;
    for(unsigned int i = 0; i < OptimizationRun::loss_history.size() && i < OptimizationRun::shape_gradients.size(); i++) {
      gradients << i;
      for(double value : OptimizationRun::shape_gradients[i]) gradients << " " << value;
      gradients << std::endl;
    }
    report.note("The outputs of the field (slices, signal along z, ParaView files) belong to the best shape, which is computed again at the end.");
    report.section("Solution process");
    report.note("In an optimization run the table of the iterative solvers covers all forward and adjoint solves of all evaluations.");
  }
  GlobalRunReport.write();
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
  if(OptimizationRun::stopping_criteria_active) sampled_shapes.push_back(sample_current_shape());
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
    if(is_first || loss_functional_evaluation < OptimizationRun::best_loss) {
      OptimizationRun::best_loss = loss_functional_evaluation;
      OptimizationRun::best_shape_dofs = x_vec;
    }
    if(gradient_norm <= GlobalParams.optimization_gradient_tolerance * OptimizationRun::initial_gradient_norm) {
      throw OptimizationFinished("The norm of the shape gradient (" + std::to_string(gradient_norm) + ") is below " + std::to_string(GlobalParams.optimization_gradient_tolerance) + " times its initial value (" + std::to_string(OptimizationRun::initial_gradient_norm) + ").");
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

