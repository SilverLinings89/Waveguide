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
#include "ConvergenceRun.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include "../Helpers/staticfunctions.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"

ConvergenceRun::ConvergenceRun() { }

ConvergenceRun::~ConvergenceRun() {
  delete mainProblem;
  delete otherProblem;
}

void ConvergenceRun::set_norming_factor() {
  double max_value_norm = 0.0;
  for(unsigned int i = 0; i < evaluation_base_problem.size(); i++) {
    double local = std::abs(evaluation_base_problem[i][0])*std::abs(evaluation_base_problem[i][0]) + std::abs(evaluation_base_problem[i][1])*std::abs(evaluation_base_problem[i][1]) + std::abs(evaluation_base_problem[i][2])*std::abs(evaluation_base_problem[i][2]);
    local = std::sqrt(local);
    if(local > max_value_norm) {
      max_value_norm = local;
    }
  }
  norming_factor = max_value_norm;
}

void ConvergenceRun::prepare() {
  print_info("ConvergenceRun::prepare", "Start", LoggingLevel::DEBUG_ONE);
  GlobalParams.Cells_in_x = GlobalParams.convergence_max_cells;
  GlobalParams.Cells_in_y = GlobalParams.convergence_max_cells;
  GlobalParams.Cells_in_z = GlobalParams.convergence_max_cells;
  Geometry.initialize();
  mainProblem = new NonLocalProblem(GlobalParams.Sweeping_Level);
  mainProblem->initialize();
  for(auto it = Geometry.levels[0].inner_domain->triangulation.begin_active(); it != Geometry.levels[0].inner_domain->triangulation.end(); it++) {
    evaluation_positions.push_back(it->center());
  }
  for(unsigned int i = 0; i < evaluation_positions.size(); i++) {
    NumericVectorLocal local_solution(3);
    GlobalParams.source_field->vector_value(evaluation_positions[i], local_solution);
    std::vector<ComplexNumber> local_solution_vector;
    for(unsigned int j = 0; j < 3; j++) {
      local_solution_vector.push_back(local_solution[j]);
    }
    evaluation_exact_solution.push_back(local_solution_vector);
  }
  mainProblem->assemble();
  mainProblem->compute_solver_factorization();
  mainProblem->solve_with_timers_and_count();
  mainProblem->output_results();
  mainProblem->empty_memory();
  base_problem_n_dofs = mainProblem->compute_total_number_of_dofs();
  base_problem_n_cells = mainProblem->n_total_cells();
  base_problem_h = mainProblem->compute_h();
  evaluation_base_problem = mainProblem->evaluate_solution_at(evaluation_positions);
  base_problem_theoretical_error = compute_error_for_two_eval_vectors(evaluation_base_problem, evaluation_exact_solution);
  delete mainProblem;
  print_info("ConvergenceRun::prepare", "End", LoggingLevel::DEBUG_ONE);
}

void ConvergenceRun::run() {
    print_info("ConvergenceRun::run", "Start", LoggingLevel::PRODUCTION_ONE);
    for(unsigned int run_index = 0; run_index < GlobalParams.convergence_cell_counts.size()-1; run_index++) {
      GlobalParams.Cells_in_x = GlobalParams.convergence_cell_counts[run_index];
      GlobalParams.Cells_in_y = GlobalParams.convergence_cell_counts[run_index];
      GlobalParams.Cells_in_z = GlobalParams.convergence_cell_counts[run_index];
      Geometry.initialize();
      otherProblem = new NonLocalProblem(GlobalParams.Sweeping_Level);
      otherProblem->initialize();
      otherProblem->assemble();
      otherProblem->compute_solver_factorization();
      otherProblem->solve_with_timers_and_count();
      std::vector<std::vector<ComplexNumber>> other_evaluations = otherProblem->evaluate_solution_at(evaluation_positions);
      double numerical_error = compute_error_for_two_eval_vectors(evaluation_base_problem, other_evaluations);
      double theoretical_error = compute_error_for_two_eval_vectors(evaluation_exact_solution, other_evaluations);
      numerical_errors.push_back(numerical_error);
      theoretical_errors.push_back(theoretical_error);
      std::string msg = "Result: " + std::to_string(GlobalParams.convergence_cell_counts[run_index]) + " found numerical error " + std::to_string(numerical_error) + "and theoretical error " + std::to_string(theoretical_error);
      print_info("ConvergenceRun::run", msg , LoggingLevel::PRODUCTION_ONE);
      unsigned int temp_ndofs = otherProblem->compute_total_number_of_dofs();
      n_dofs_for_cases.push_back(temp_ndofs);
      h_values.push_back(otherProblem->compute_h());
      total_cells.push_back(otherProblem->n_total_cells());
      output.push_values(temp_ndofs,numerical_error,theoretical_error);
      otherProblem->empty_memory();
    }
    write_outputs();

    print_info("ConvergenceRun::run", "End", LoggingLevel::PRODUCTION_ONE);
}

void ConvergenceRun::prepare_transformed_geometry() {
}

void ConvergenceRun::write_outputs() {
    std::string msg = "The base problem had " + std::to_string(base_problem_n_dofs) + " dofs, " + std::to_string(base_problem_n_cells) + " cells, h was " + std::to_string(base_problem_h) + " and a theoretical error of "+ std::to_string(base_problem_theoretical_error);
    print_info("ConvergenceRun::results", msg, LoggingLevel::PRODUCTION_ONE);
    msg =  "The computed numerical errors were: ";
    for(unsigned int i = 0; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
      msg += std::to_string(numerical_errors[i]) + " ";
    }
    print_info("ConvergenceRun::results", msg, LoggingLevel::PRODUCTION_ONE);
    msg = "The computed theoretical errors were: ";
    for(unsigned int i = 0; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
      msg += std::to_string(theoretical_errors[i]) + " ";
    }
    print_info("ConvergenceRun::results", msg, LoggingLevel::PRODUCTION_ONE);
    msg = "The cell counts were: ";
    for(unsigned int i = 0; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
      msg += std::to_string(total_cells[i]) + " ";
    }
    print_info("ConvergenceRun::results", msg, LoggingLevel::PRODUCTION_ONE);
    msg = "The number of dofs were: ";
    for(unsigned int i = 0; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
      msg += std::to_string(n_dofs_for_cases[i]) + " ";
    }
    print_info("ConvergenceRun::results", msg, LoggingLevel::PRODUCTION_ONE);
    msg = "The maximal values of h were: ";
    for(unsigned int i = 0; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
      msg += std::to_string(h_values[i]) + " ";
    }
    print_info("ConvergenceRun::results", msg, LoggingLevel::PRODUCTION_ONE);

    output.set_title("Convergence History");
    output.write_gnuplot_file();
    output.run_gnuplot();
}

double ConvergenceRun::compute_error_for_two_eval_vectors(std::vector<std::vector<ComplexNumber>> a, std::vector<std::vector<ComplexNumber>> b) {
  double local = 0.0;
  for(unsigned int i = 0; i < a.size(); i++) {
    double x = std::abs(a[i][0] - b[i][0]);
    double y = std::abs(a[i][1] - b[i][1]);
    double z = std::abs(a[i][2] - b[i][2]);
    local += std::sqrt(x*x + y*y + z*z);
  }
  local /= evaluation_positions.size();
  local *= (Geometry.local_x_range.second - Geometry.local_x_range.first) * (Geometry.local_y_range.second - Geometry.local_y_range.first) * (Geometry.local_z_range.second - Geometry.local_z_range.first);
  double ret = dealii::Utilities::MPI::sum(local, MPI_COMM_WORLD);
  ret /= norming_factor;
  return ret;
}
