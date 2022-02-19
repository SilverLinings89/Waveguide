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
#include <functional>

NonLocalProblem * OptimizationRun::mainProblem;
std::vector<std::vector<double>> OptimizationRun::shape_dofs;
std::vector<std::vector<double>> OptimizationRun::shape_gradients;
unsigned int OptimizationRun::step_counter;

OptimizationRun::OptimizationRun() {
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
    const unsigned int n_shape_dofs = GlobalSpaceTransformation->NFreeDofs();
    dealii::Vector<double> shape_dofs(n_shape_dofs);
    OptimizationRun::step_counter = 0;
    for(unsigned int i = 0; i < n_shape_dofs; i++) {
      shape_dofs[i] = GlobalSpaceTransformation->get_free_dof(i);
    }
    dealii::SolverControl sc(GlobalParams.optimization_n_shape_steps, GlobalParams.optimization_residual_tolerance, true, true);
    dealii::SolverBFGS<dealii::Vector<double>> solver(sc);
    solver.solve(function_pointer, shape_dofs);
    GlobalTimerManager.write_output();
    OptimizationRun::mainProblem->output_results();
    print_info("OptimizationRun::run", "End", LoggingLevel::PRODUCTION_ONE);
}

void OptimizationRun::prepare_transformed_geometry() {
}

void OptimizationRun::solve_main_problem() {
  OptimizationRun::mainProblem->assemble();
  OptimizationRun::mainProblem->compute_solver_factorization();
  OptimizationRun::mainProblem->solve_with_timers_and_count();
}

double OptimizationRun::perform_step(const dealii::Vector<double> & x, dealii::Vector<double> & g) {
  std::pair<double, std::vector<double>> ret;
  OptimizationRun::set_shape_dofs(x);
  OptimizationRun::solve_main_problem();
  ret.first = GlobalParams.Amplitude_of_input_signal - std::abs(mainProblem->compute_signal_strength_of_solution());
  ret.second = mainProblem->compute_shape_gradient();
  for(unsigned int i = 0; i < g.size(); i++) {
    g[i] = ret.second[i];
  }
  return ret.first;
}

void OptimizationRun::set_shape_dofs(const dealii::Vector<double> in_shape_dofs) {
  if(in_shape_dofs.size() == GlobalSpaceTransformation->NFreeDofs()) {
    for(unsigned int i = 0; i < in_shape_dofs.size(); i++) {
      GlobalSpaceTransformation->set_free_dof(i, in_shape_dofs[i]);
    }
  } else {
    std::cout << "There was an error setting the dofs. Size mismatch of shape update."<< std::endl; 
  }
}

