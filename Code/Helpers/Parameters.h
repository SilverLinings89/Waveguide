#pragma once

#include <mpi.h>
#include <string>
#include "ShapeDescription.h"
#include "../Core/Types.h"
#include "Enums.h"

/**
 * \class Parameters
 * \brief This structure contains all information contained in the input file
 * and some values that can simply be computed from it.
 *
 * In the application, static Variable of this type makes the input parameters
 * available globally. \author: Pascal Kraft \date: 28.11.2016
 */

const static std::string PrecOptionNames[] = {
    "Sweeping", "FastSweeping", "HSIESweeping", "HSIEFastSweeping"};

struct Parameters {
    ShapeDescription sd;

    double Solver_Precision;

    unsigned int GMRES_Steps_before_restart;

    unsigned int GMRES_max_steps;

    unsigned int MPI_Rank;

    unsigned int NumberProcesses;

    double Amplitude_of_input_signal;

    double Width_of_waveguide; // x-direction

    double Height_of_waveguide; // y-direction

    double Horizontal_displacement_of_waveguide; // x-direction

    double Vertical_displacement_of_waveguide;

    double Epsilon_R_in_waveguide;

    double Epsilon_R_outside_waveguide;

    double Mu_R_in_waveguide;

    double Mu_R_outside_waveguide;

    unsigned int HSIE_polynomial_degree;

    bool Perform_Optimization;

    double kappa_0_angle; // For HSIE-Elements. kappa_0 is a complex number with absolute value 1. This is the angle against the real axis.

    ComplexNumber kappa_0; // For HSIE-Elements

    unsigned int Nedelec_element_order;
    
    unsigned int Blocks_in_z_direction;

    unsigned int Blocks_in_x_direction;

    unsigned int Blocks_in_y_direction;

    unsigned int Index_in_x_direction;

    unsigned int Index_in_y_direction;

    unsigned int Index_in_z_direction;

    unsigned int Cells_in_x;
    
    unsigned int Cells_in_y;
    
    unsigned int Cells_in_z;

    int current_run_number;

    double Geometry_Size_X;

    double Geometry_Size_Y;

    double Geometry_Size_Z;

    unsigned int Number_of_sectors;

    double Sector_thickness;

    double Sector_padding;

    double Pi = 3.141592653589793238462;

    double Omega = 1.0;

    double Lambda = 1.0;

    double Waveguide_value_V = 1.0;

    bool Use_Predefined_Shape = false;

    unsigned int Number_of_Predefined_Shape = 1;

    unsigned int HSIE_SWEEPING_LEVEL =
        1;  // 1 means normal sweeping, 2 means hierarchical sweeping with depth
            // 1, 3 means hierarchical sweeping with depth 2.

    auto complete_data() -> void;

    LoggingLevel Logging_Level = LoggingLevel::PRODUCTION_ONE;
};
