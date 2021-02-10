#include <cmath>
#include <iostream>
#include "Parameters.h"
#include "../Helpers/staticfunctions.h"
#include "../Helpers/ExactSolution.h"
#include "../Helpers/ExactSolutionRamped.h"
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
    Logging_Level = DEBUG_ALL;
    MPI_Rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    } else {
        Index_in_x_direction = 0;
        Index_in_y_direction = 0;
        Index_in_z_direction = 0;
        Logging_Level = DEBUG_ALL;
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
}
