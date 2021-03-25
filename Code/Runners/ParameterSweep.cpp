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
  const unsigned int n_order_steps = (GlobalParams.Max_HSIE_Order - GlobalParams.Min_HSIE_Order) / 3;
  double ** errors = new double*[GlobalParams.N_Kappa_0_Steps];
  for(unsigned int i = 0; i < GlobalParams.N_Kappa_0_Steps; i++) {
    errors[i] = new double [GlobalParams.Max_HSIE_Order - GlobalParams.Min_HSIE_Order];
    for (unsigned int j = 0; j < n_order_steps; j++) {
      errors[i][j] = 0;
    }
  }
  for(unsigned int kappa_step = 0; kappa_step < GlobalParams.N_Kappa_0_Steps; kappa_step++) {
    GlobalParams.kappa_0_angle = (2*GlobalParams.Pi / GlobalParams.N_Kappa_0_Steps) * kappa_step;
    GlobalParams.kappa_0 = {7, -1};
    for (unsigned int order_index = 0; order_index < n_order_steps; order_index++) {
      unsigned int order = GlobalParams.Min_HSIE_Order + order_index*3;
      GlobalParams.HSIE_polynomial_degree = order;
      unsigned int n_procs = GlobalParams.NumberProcesses;
      unsigned int rank = GlobalParams.MPI_Rank;
      if ((kappa_step * GlobalParams.N_Kappa_0_Steps + order_index) % n_procs == rank) {
        print_info("ParameterSweep::run",
                  "Performing parameter study for kappa (" +
                      std::to_string(GlobalParams.kappa_0.real()) + " + " +
                      std::to_string(GlobalParams.kappa_0.imag()) +
                      ") and order " + std::to_string(order),
                  false, LoggingLevel::PRODUCTION_ALL);
        mainProblem = new LocalProblem();

        mainProblem->initialize();
        mainProblem->reinit();

        mainProblem->assemble();

        mainProblem->compute_solver_factorization();
        
        mainProblem->solve();

        mainProblem->output_results();
        
        const double error_run =  mainProblem->compute_L2_error();
        print_info("ParameterSweep::run", "Found error " + std::to_string(error_run) + " for order " + std::to_string(order) + " and kappa" +
                                    std::to_string(GlobalParams.kappa_0_angle));
        errors[kappa_step][order_index] = error_run;
        delete mainProblem;
      }
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  for(unsigned int kappa_step = 0; kappa_step < GlobalParams.N_Kappa_0_Steps; kappa_step++) {
    for (unsigned int order_index = 0; order_index < n_order_steps; order_index++) {
      errors[kappa_step][order_index] = dealii::Utilities::MPI::sum(errors[kappa_step][order_index], MPI_COMM_WORLD);
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
