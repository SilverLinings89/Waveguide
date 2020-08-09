#pragma once

#include <cmath>
#include <iostream>

auto Parameters::complete_data() -> void {
    kappa_0 = { std::sin(kappa_0_angle), std::cos(kappa_0_angle) };
    unsigned int required_procs = Blocks_in_x_direction * Blocks_in_y_direction * Blocks_in_z_direction;
    if(required_procs != NumberProcesses) {
        std::cout << "The number of mpi processes does not match the required processes" << std::endl;
        exit();
    }
}
