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
  print_info("SingleCoreRun::run", "L2 error against the input signal: " + std::to_string(l2_error), LoggingLevel::PRODUCTION_ONE);
  mainProblem->write_multifile_output("solution");
  GlobalTimerManager.write_output();
  print_info("SingleCoreRun::run", "End", LoggingLevel::PRODUCTION_ONE);
}

void SingleCoreRun::prepare_transformed_geometry() {
}

