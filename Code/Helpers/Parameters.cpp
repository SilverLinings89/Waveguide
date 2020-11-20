#include <cmath>
#include <iostream>
#include "Parameters.h"
#include "../Helpers/staticfunctions.h"

auto Parameters::complete_data() -> void {
    kappa_0 = { std::sin(kappa_0_angle), std::cos(kappa_0_angle) };
    unsigned int required_procs = Blocks_in_x_direction * Blocks_in_y_direction * Blocks_in_z_direction;
    if(required_procs != NumberProcesses) {
        print_info("Parameters::complete_data", "The number of mpi processes does not match the required processes", false, LoggingLevel::DEBUG_ALL);
        exit(0);
    }
    Index_in_z_direction = MPI_Rank / (Blocks_in_x_direction*Blocks_in_y_direction);
    Index_in_y_direction = (MPI_Rank-(Index_in_z_direction * Blocks_in_x_direction*Blocks_in_y_direction)) / Blocks_in_x_direction;
    Index_in_x_direction = 0; // TODO: implement this.
    Logging_Level = DEBUG_ALL;
}
