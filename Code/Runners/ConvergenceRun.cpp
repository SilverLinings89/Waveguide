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

void ConvergenceRun::prepare() {
  print_info("ConvergenceRun::prepare", "Start", true, LoggingLevel::DEBUG_ONE);
  GlobalParams.Cells_in_x = GlobalParams.convergence_max_cells;
  GlobalParams.Cells_in_y = GlobalParams.convergence_max_cells;
  GlobalParams.Cells_in_z = GlobalParams.convergence_max_cells;
  
  mainProblem = new NonLocalProblem(GlobalParams.Sweeping_Level);
  mainProblem->initialize();
  
  print_info("ConvergenceRun::prepare", "End", true, LoggingLevel::DEBUG_ONE);
}

void ConvergenceRun::run() {
    print_info("ConvergenceRun::run", "Start", true, LoggingLevel::PRODUCTION_ONE);
    std::vector<double> errors;
    std::vector<unsigned int> total_cells;
    std::vector<double> h_values;
    std::vector<unsigned int> n_dofs_for_cases;

    mainProblem->assemble();
    
    mainProblem->compute_solver_factorization();
    mainProblem->solve_with_timers_and_count();
    for(auto it = Geometry.levels[0].inner_domain->triangulation.begin_active(); it != Geometry.levels[0].inner_domain->triangulation.end(); it++) {
      evaluation_positions.push_back(it->center());
    }
    mainProblem->empty_memory();
    evaluation_base_problem = mainProblem->evaluate_solution_at(evaluation_positions);
    for(unsigned int run_index = 0; run_index < GlobalParams.convergence_cell_counts.size()-1; run_index++) {
      GlobalParams.Cells_in_x = GlobalParams.convergence_cell_counts[run_index];
      GlobalParams.Cells_in_y = GlobalParams.convergence_cell_counts[run_index];
      GlobalParams.Cells_in_z = GlobalParams.convergence_cell_counts[run_index];
      otherProblem = new NonLocalProblem(GlobalParams.Sweeping_Level);
      otherProblem->initialize();
      otherProblem->assemble();
      otherProblem->compute_solver_factorization();
      otherProblem->solve_with_timers_and_count();
      double temp_error = compute_error_for_run();
      errors.push_back(temp_error);
      unsigned int temp_ndofs = otherProblem->compute_total_number_of_dofs();
      n_dofs_for_cases.push_back(temp_ndofs);
      h_values.push_back(otherProblem->compute_h());
      total_cells.push_back(otherProblem->n_total_cells());
      output.push_values(temp_ndofs,temp_error);
      delete otherProblem;
      mainProblem->empty_memory();
    }

    if(GlobalParams.MPI_Rank==0) {
      std::cout << "The computed errors were: ";
      for(unsigned int i ; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
        std::cout << errors[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "The cell counts were: ";
      for(unsigned int i ; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
        std::cout << total_cells[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "The number of dofs were: ";
      for(unsigned int i ; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
        std::cout << n_dofs_for_cases[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "The maximal values of h were: ";
      for(unsigned int i ; i < GlobalParams.convergence_cell_counts.size()-1; i++) {
        std::cout << h_values[i] << " ";
      }
      std::cout << std::endl;
    }
    output.set_title("Convergence History");
    output.write_gnuplot_file();
    output.run_gnuplot();
    print_info("ConvergenceRun::run", "End", true, LoggingLevel::PRODUCTION_ONE);
}

double ConvergenceRun::compute_error_for_run() {
  double local = 0;
  std::vector<std::vector<ComplexNumber>> other_evaluations = otherProblem->evaluate_solution_at(evaluation_positions);
  for(unsigned int i = 0; i < other_evaluations.size(); i++) {
    double a = std::abs(evaluation_base_problem[i][0] - other_evaluations[i][0]);
    double b = std::abs(evaluation_base_problem[i][1] - other_evaluations[i][1]);
    double c = std::abs(evaluation_base_problem[i][1] - other_evaluations[i][1]);

    local += std::sqrt(a*a + b*b + c*c);
  }
  double ret = dealii::Utilities::MPI::sum(local, MPI_COMM_WORLD);
  return ret;
}

void ConvergenceRun::prepare_transformed_geometry() {
}

