/*
 * Simulation.cpp
 *
 *  Created on: Jun 24, 2019
 *      Author: kraft
 */

#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "Simulation.h"
#include "SingleCoreRun.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include "../Helpers/staticfunctions.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"

SingleCoreRun::SingleCoreRun() { }

SingleCoreRun::~SingleCoreRun() {
  delete mainProblem;
}

void SingleCoreRun::prepare() {
  print_info("SingleCoreRun::prepare", "Start", LoggingLevel::DEBUG_ONE);
  
  mainProblem = new LocalProblem();

  mainProblem->initialize();

  print_info("SingleCoreRun::prepare", "End", LoggingLevel::DEBUG_ONE);
}

void SingleCoreRun::run() {
  print_info("SingleCoreRun::run", "Start", LoggingLevel::PRODUCTION_ONE);

  mainProblem->assemble();

  print_info("SingleCoreRun::run", "Assembling completed", LoggingLevel::PRODUCTION_ONE);
  
  // mainProblem->compute_solver_factorization();
  
  mainProblem->solve(); 
  const double l2_error = mainProblem->compute_error();
  const double l2_norm_exact = mainProblem->compute_L2_norm_of_exact_solution();
  print_info("SingleCoreRun::run", "L2 error against the input signal: " + std::to_string(l2_error) + " (relative: " + std::to_string(l2_error / l2_norm_exact) + ")", LoggingLevel::PRODUCTION_ONE);
  {
    // Machine readable summary, analogous to the one of sweeping runs.
    std::ofstream results(GlobalOutputManager.get_full_filename("results.txt"));
    results.precision(12);
    results << "error_l2 = " << l2_error << std::endl;
    results << "error_l2_relative = " << l2_error / l2_norm_exact << std::endl;
    results << "n_dofs = " << Geometry.levels[0].n_total_level_dofs - mainProblem->n_unused_dof_indices << std::endl;
    results << "n_dofs_interior = " << Geometry.levels[0].inner_domain->n_locally_active_dofs << std::endl;
    results << "matrix_nonzeros = " << mainProblem->matrix->n_nonzero_elements() << std::endl;
    results << "local_factorization_time_max = " << mainProblem->time_of_factorization << std::endl;
    results << "local_factor_entries_full_rank = " << mainProblem->factor_entries_full_rank << std::endl;
    results << "local_factor_entries_stored = " << mainProblem->factor_entries_stored << std::endl;
    results << "local_factor_memory_mb_max = " << mainProblem->factor_memory_mb << std::endl;
  }
  mainProblem->write_multifile_output("solution");
  GlobalTimerManager.write_output();
  print_info("SingleCoreRun::run", "End", LoggingLevel::PRODUCTION_ONE);
}

void SingleCoreRun::prepare_transformed_geometry() {
}

