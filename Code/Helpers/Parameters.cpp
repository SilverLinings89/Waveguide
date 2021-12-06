#include <cmath>
#include <iostream>
#include "Parameters.h"
#include "../Helpers/staticfunctions.h"
#include "../Solutions/ExactSolution.h"
#include "../Solutions/ExactSolutionRamped.h"
#include "PointSourceField.h"

auto Parameters::complete_data() -> void {
    
    if(!Enable_Parameter_Run) {
        kappa_0 = { std::sin(kappa_0_angle), std::cos(kappa_0_angle) };
        unsigned int required_procs = Blocks_in_x_direction * Blocks_in_y_direction * Blocks_in_z_direction;
        if(required_procs != NumberProcesses) {
            print_info("Parameters::complete_data", "The number of mpi processes does not match the required processes", false, LoggingLevel::DEBUG_ALL);
            exit(0);
        }
        Index_in_z_direction = MPI_Rank / (Blocks_in_x_direction*Blocks_in_y_direction);
        Index_in_y_direction = (MPI_Rank-(Index_in_z_direction * Blocks_in_x_direction*Blocks_in_y_direction)) / Blocks_in_x_direction;
        Index_in_x_direction = MPI_Rank - Index_in_z_direction*(Blocks_in_x_direction*Blocks_in_y_direction) - Index_in_y_direction*Blocks_in_x_direction ;
        MPI_Rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    } else {
        Index_in_x_direction = 0;
        Index_in_y_direction = 0;
        Index_in_z_direction = 0;
        MPI_Rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    }
    if(Point_Source_Type == 2) {
        source_field = new PointSourceFieldHertz(0.5);
        if(MPI_Rank == 0) print_info("Parameters::complete_data", "Using Hertz-type point source.", false, LoggingLevel::PRODUCTION_ONE);
    }
    if(Point_Source_Type == 1) {
        source_field = new PointSourceFieldCosCos();
        if(MPI_Rank == 0) print_info("Parameters::complete_data", "Using Cos-Cos-type point source.", false, LoggingLevel::PRODUCTION_ONE);
    }
    if(Point_Source_Type == 0) {
        source_field = new ExactSolution(true, false);
        if(MPI_Rank == 0) print_info("Parameters::complete_data", "Using mode for rectangular waveguide.", false, LoggingLevel::PRODUCTION_ONE);
    }
    if(Point_Source_Type == 3) {
        source_field = new ExactSolution(true, false);
        if(MPI_Rank == 0) print_info("Parameters::complete_data", "Using mode for rectangular waveguide.", false, LoggingLevel::PRODUCTION_ONE);
    }
    Omega = 2.0 * Pi / Lambda;
    if(Use_Predefined_Shape) {
      deallog << "Using case " << Number_of_Predefined_Shape << std::endl;
      std::ifstream input( "Modes/test.csv" );
      std::string line;
      int counter = 0;
      bool case_found = false;
      if(Number_of_Predefined_Shape >= 0 && Number_of_Predefined_Shape < 36){
          while(std::getline( input, line ) && counter < 36){
            if(counter == Number_of_Predefined_Shape) {
              sd.SetByString(line);
              case_found = true;
            }
            counter++;
          }
          if(!case_found) {
            deallog << "There was a severe error. The case was not found therefore not initialized." << std::endl;
          }
      }
    } else {
      deallog << "Not using case." << std::endl;
    }
}

auto Parameters::check_validity() -> bool {
    bool valid = true;

    // prescribing zero on the input interface should only be used with tapered input. Otherewise there are two Dirichlet constraints on the same interface
    if(prescribe_0_on_input_side && !use_tapered_input_signal) valid = false;

    // check if the waveguide edges are resolved
    // TODO.


    return valid;
}
