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
#include "../Core/GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"

SingleCoreRun::SingleCoreRun() { }

SingleCoreRun::~SingleCoreRun() {
  delete mainProblem;
}

void SingleCoreRun::prepare() {
  print_info("SingleCoreRun::prepare", "Start", true, LoggingLevel::DEBUG_ONE);
  if(GlobalParams.Point_Source_Type == 0) {
    rmProblem = new RectangularMode();
  }

  mainProblem = new LocalProblem();

  mainProblem->initialize();

  print_info("SingleCoreRun::prepare", "End", true, LoggingLevel::DEBUG_ONE);
}

void SingleCoreRun::run() {
  print_info("SingleCoreRun::run", "Start", true, LoggingLevel::PRODUCTION_ONE);

  if(GlobalParams.Point_Source_Type == 0) {
    rmProblem->run();
  }
    
  mainProblem->assemble();

  mainProblem->compute_solver_factorization();
  
  mainProblem->solve();
  
  mainProblem->output_results("solution_");
  
  print_info("SingleCoreRun::run", "End", true, LoggingLevel::PRODUCTION_ONE);
}

void SingleCoreRun::prepare_transformed_geometry() {
}

