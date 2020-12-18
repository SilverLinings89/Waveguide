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
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include "../Helpers/staticfunctions.h"
#include "./GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"

Simulation::Simulation(const std::string run_file,const std::string case_file) {
  initialize_global_variables(run_file, case_file);
  create_output_directory();
  prepare_transformed_geometry();
}

Simulation::~Simulation() {
  delete mainProblem;
}

void Simulation::prepare() {
  print_info("Simulation::prepare", "Start", true, LoggingLevel::DEBUG_ONE);
  if(GlobalParams.Point_Source_Type == 0) {
    rmProblem = new RectangularMode();
  } else {
    if (GlobalParams.NumberProcesses > 1) {
      mainProblem = new NonLocalProblem(GlobalParams.HSIE_SWEEPING_LEVEL);
    } else {
      mainProblem = new LocalProblem();
    }
    mainProblem->initialize();
  }
  print_info("Simulation::prepare", "End", true, LoggingLevel::DEBUG_ONE);
}

void Simulation::run() {
  print_info("Simulation::run", "Start", true, LoggingLevel::PRODUCTION_ONE);
  if(GlobalParams.Point_Source_Type == 0) {
    rmProblem->run();
  } else {
    mainProblem->assemble();

    mainProblem->compute_solver_factorization();
    
    mainProblem->solve();
    
    mainProblem->output_results("solution_");
  }
  print_info("Simulation::run", "End", true, LoggingLevel::PRODUCTION_ONE);
}

void Simulation::create_output_directory() {
  char *pPath;
  pPath = getenv("WORK");
  bool seperate_solutions = (pPath != nullptr);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    dealii::deallog.depth_console(10);
  } else {
    deallog.depth_console(0);
  }
  int i = 0;
  bool dir_exists = true;
  while (dir_exists) {
    std::stringstream out;
    if (seperate_solutions) {
      out << pPath << "/";
    }
    out << "Solutions/run";
    out << i;
    solutionpath = out.str();
    struct stat myStat;
    const char *myDir = solutionpath.c_str();
    if ((stat(myDir, &myStat) == 0) &&
        (((myStat.st_mode) & S_IFMT) == S_IFDIR)) {
      i++;
    } else {
      dir_exists = false;
    }
  }
  i = Utilities::MPI::max(i, MPI_COMM_WORLD);
  std::stringstream out;
  if (seperate_solutions) {
    out << pPath << "/";
  }
  out << "Solutions/run";

  out << i;
  solutionpath = out.str();
  mkdir(solutionpath.c_str(), ACCESSPERMS);

  log_stream.open(
      solutionpath + "/main" +
          std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
          ".log",
      std::ios::binary);

  deallog.attach(log_stream);
}

void Simulation::prepare_transformed_geometry() {
}

