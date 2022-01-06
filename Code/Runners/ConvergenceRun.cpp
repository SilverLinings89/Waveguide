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
#include "SweepingRun.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include "../Helpers/staticfunctions.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"

SweepingRun::SweepingRun() { }

SweepingRun::~SweepingRun() {
  delete mainProblem;
}

void SweepingRun::prepare() {
  print_info("SweepingRun::prepare", "Start", true, LoggingLevel::DEBUG_ONE);
  if(GlobalParams.Point_Source_Type == 0) {
    rmProblem = new RectangularMode();
  } 
  
  mainProblem = new NonLocalProblem(GlobalParams.Sweeping_Level);
  mainProblem->initialize();
  
  print_info("SweepingRun::prepare", "End", true, LoggingLevel::DEBUG_ONE);
}

void SweepingRun::run() {
    print_info("SweepingRun::run", "Start", true, LoggingLevel::PRODUCTION_ONE);
    if(GlobalParams.Point_Source_Type == 0) {
        rmProblem->run();
    }
    mainProblem->assemble();
    
    mainProblem->compute_solver_factorization();
    mainProblem->solve_with_timers_and_count();
    GlobalTimerManager.write_output();
    mainProblem->output_results();
    print_info("Simulation::run", "End", true, LoggingLevel::PRODUCTION_ONE);
}

void SweepingRun::prepare_transformed_geometry() {
}

