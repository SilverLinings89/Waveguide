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
#include <complex>
#include "ParameterSweep.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include "../Helpers/staticfunctions.h"
#include "../Core/GlobalObjects.h"
#include "../ModalComputations/RectangularMode.h"

ParameterSweep::ParameterSweep() {}

ParameterSweep::~ParameterSweep() {
  delete mainProblem;
}

void ParameterSweep::prepare() {
  print_info("ParameterSweep::prepare", "Start", true, LoggingLevel::DEBUG_ONE);
  if(GlobalParams.Point_Source_Type == 0) {
    rmProblem = new RectangularMode();
  } 
  print_info("ParameterSweep::prepare", "End", true, LoggingLevel::DEBUG_ONE);
}

void ParameterSweep::run() {
  print_info("ParameterSweep::run", "Start", true, LoggingLevel::PRODUCTION_ONE);
  double ** errors = new double*[GlobalParams.N_Kappa_0_Steps];
  for(unsigned int i = 0; i < GlobalParams.N_Kappa_0_Steps; i++) {
    errors[i] = new double [(GlobalParams.Max_HSIE_Order - GlobalParams.Min_HSIE_Order)];
    for (unsigned int j = 0; j < (GlobalParams.Max_HSIE_Order - GlobalParams.Min_HSIE_Order); j++) {
      errors[i][j] = 0;
    }
  }
  for(unsigned int kappa_step = 0; kappa_step < GlobalParams.N_Kappa_0_Steps; kappa_step++) {
    GlobalParams.kappa_0_angle = (2*GlobalParams.Pi / GlobalParams.N_Kappa_0_Steps) * kappa_step;
    GlobalParams.kappa_0 = {std::sin(GlobalParams.kappa_0_angle), std::cos(GlobalParams.kappa_0_angle)};
    for(unsigned int order = GlobalParams.Min_HSIE_Order; order < GlobalParams.Max_HSIE_Order; order ++){
      print_info("ParameterSweep::run", "Performing parameter study for kappa (" + std::to_string(GlobalParams.kappa_0.real()) + " + " + std::to_string(GlobalParams.kappa_0.imag()) + ") and order " + std::to_string(order), false, LoggingLevel::PRODUCTION_ALL);
      GlobalParams.HSIE_polynomial_degree = order*2;
      unsigned int n_procs = GlobalParams.NumberProcesses;
      unsigned int rank = GlobalParams.MPI_Rank;
      if(order % n_procs == rank) {
        mainProblem = new LocalProblem();

        mainProblem->initialize();

        mainProblem->assemble();

        mainProblem->compute_solver_factorization();
        
        mainProblem->solve();
        
        mainProblem->output_results("kappa" + std::to_string(GlobalParams.kappa_0_angle) + "order" + std::to_string(order));

        errors[kappa_step][order] = mainProblem->compute_L2_error();
      }
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  for(unsigned int kappa_step = 0; kappa_step < GlobalParams.N_Kappa_0_Steps; kappa_step++) {
    for(unsigned int order = GlobalParams.Min_HSIE_Order; order < GlobalParams.Max_HSIE_Order; order ++){
      errors[kappa_step][order] = dealii::Utilities::MPI::sum(errors[kappa_step][order], MPI_COMM_WORLD);
    }
  }
  if(GlobalParams.MPI_Rank == 0) {
    std::ofstream myfile ("computed_errors.dat");
    myfile << "kappa_0 \\ order" << "\t";
    for(unsigned int order = GlobalParams.Min_HSIE_Order; order < GlobalParams.Max_HSIE_Order; order ++){
      myfile << order << "\t" ;
    }
    myfile << std::endl;
    for(unsigned int kappa_step = 0; kappa_step < GlobalParams.N_Kappa_0_Steps; kappa_step++) {
      myfile << (2*GlobalParams.Pi / GlobalParams.N_Kappa_0_Steps) * kappa_step << "\t";
      for(unsigned int order = GlobalParams.Min_HSIE_Order; order < GlobalParams.Max_HSIE_Order; order ++){
        myfile << errors[kappa_step][order] << "\t";
      }
      myfile << std::endl;
    } 
    myfile.close();
  }
  
  print_info("ParameterSweep::run", "End", true, LoggingLevel::PRODUCTION_ONE);
}

void ParameterSweep::prepare_transformed_geometry() { }
