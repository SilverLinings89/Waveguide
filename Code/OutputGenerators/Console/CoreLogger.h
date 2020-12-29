#pragma once
#include <string>

/**
 * Outputs I want:
 * - Timing output for all solver runs on any level.
 * - Convergence histories for any solver run on any level (except the lowest one maybe, bc. thats direct).
 * - Convergence rates
 * - Dof Numbers on all levels
 * - Memory Consumption of the direct solver
 * 
 * So this object mainly manages run meta-information. It needs functions that register which run the code is on (which iteration on which level etc.)
 * There will only be one instance of this object and it will be available globally.
 * It should use the FileLogger global instance to create files.
 */


class CoreLogger {
    CoreLogger();

    void start_new_convergence_history(unsigned int sweeping_level);
    void add_convergence_step(double residual, unsigned int sweeping_level);
    void final_convergence_step(double residual, unsigned int sweeping_level);
    void store_direct_solver_meta_data(std::string meta_data_string);
    void store_run_meta_data(unsigned int n_local_cells, unsigned int sweeping_level, unsigned int n_hsie_dofs);

    void write_all_outputs();
}