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
#include "../Helpers/staticfunctions.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"

OptimizationRun::OptimizationRun() { }

OptimizationRun::~OptimizationRun() {
  delete mainProblem;
}

void OptimizationRun::prepare() {
  print_info("OptimizationRun::prepare", "Start", LoggingLevel::DEBUG_ONE);
  
  mainProblem = new NonLocalProblem(GlobalParams.Sweeping_Level);
  mainProblem->initialize();
  
  print_info("OptimizationRun::prepare", "End", LoggingLevel::DEBUG_ONE);
}

void OptimizationRun::run() {
    print_info("OptimizationRun::run", "Start", LoggingLevel::PRODUCTION_ONE);
    const unsigned int n_shape_dofs = GlobalSpaceTransformation.NFreeDofs();
    dealii::Vector<double> shape_dofs(n_shape_dofs);
    for(unsigned int i = 0; i < n_shape_dofs; i++) {
      shape_dofs[i] = GlobalSpaceTransformation->get_free_dof(i);
    }
    dealii::SolverControl sc(GlobalParams.optimization_n_shape_steps, GlobalParams.optimization_residual_tolerance, true, true);
    dealii::SolverBFGS<dealii::Vector<double>> solver(sc);
    solver.solve(perform_step, shape_dofs);
    GlobalTimerManager.write_output();
    mainProblem->output_results();
    print_info("OptimizationRun::run", "End", LoggingLevel::PRODUCTION_ONE);
}

void OptimizationRun::prepare_transformed_geometry() {
}

void solve_main_problem() {
  mainProblem->assemble();
  mainProblem->compute_solver_factorization();
  mainProblem->solve_with_timers_and_count();
}

std::pair<double, std::vector<double>> OptimizationRun::perform_step(std::vector<double> x) {
  std::pair<double, std::vector<double>> ret;
  set_shape_dofs(x);
  solve_main_problem();
  ret.first = GlobalParams.Amplitude_of_input_signal - std::abs(mainProblem->compute_signal_strength_of_solution());
  ret.second = mainProblem->compute_shape_gradient();
}

void set_shape_dofs(std::vector<double> in_shape_dofs) {
  if(in_shape_dofs.size() == GlobalSpaceTransformation->NFreeDofs()) {
    for(unsigned int i = 0; i < in_shape_dofs.size(); i++) {
      GlobalSpaceTransformation->set_free_dof(i, in_shape_dofs[i]);
    }
  } else {
    std::cout << "There was an error setting the dofs. Size mismatch of shape update."<< std::endl; 
  }
}

