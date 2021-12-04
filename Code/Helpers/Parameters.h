#pragma once

#include <mpi.h>
#include <string>
#include "ShapeDescription.h"
#include "../Core/Types.h"
#include "../Core/Enums.h"

/**
 * \class Parameters
 * \brief This structure contains all information contained in the input file
 * and some values that can simply be computed from it.
 *
 * In the application, static Variable of this type makes the input parameters
 * available globally. \author: Pascal Kraft \date: 28.11.2016
 */

struct Parameters {
    ShapeDescription sd;

    double Solver_Precision = 1e-6;

    unsigned int GMRES_Steps_before_restart = 30;

    unsigned int GMRES_max_steps = 100;

    unsigned int MPI_Rank;

    unsigned int NumberProcesses;

    double Amplitude_of_input_signal = 1.0;

    double Width_of_waveguide = 2.0; // x-direction

    double Height_of_waveguide = 1.8; // y-direction

    double Horizontal_displacement_of_waveguide = 0; // x-direction

    double Vertical_displacement_of_waveguide = 0;

    double Epsilon_R_in_waveguide = 2.3409;

    double Epsilon_R_outside_waveguide = 1.8496;

    double Epsilon_R_effective = 2.1588449;

    double Mu_R_in_waveguide = 1.0;

    double Mu_R_outside_waveguide = 1.0;

    unsigned int HSIE_polynomial_degree = 5;

    bool Perform_Optimization = false;

    double kappa_0_angle = 1.0; // For HSIE-Elements. kappa_0 is a complex number with absolute value 1. This is the angle against the real axis.

    ComplexNumber kappa_0; // For HSIE-Elements

    unsigned int Nedelec_element_order = 0;
    
    unsigned int Blocks_in_z_direction = 1;

    unsigned int Blocks_in_x_direction = 1;

    unsigned int Blocks_in_y_direction = 1;

    unsigned int Index_in_x_direction;

    unsigned int Index_in_y_direction;

    unsigned int Index_in_z_direction;

    unsigned int Cells_in_x = 20;
    
    unsigned int Cells_in_y = 20;
    
    unsigned int Cells_in_z = 20;

    int current_run_number = 0;

    double Geometry_Size_X = 5;

    double Geometry_Size_Y = 5;

    double Geometry_Size_Z = 5;

    unsigned int Number_of_sectors;

    double Sector_thickness;

    double Sector_padding;

    double Pi = 3.141592653589793238462;

    double Omega = 1.0;

    double Lambda = 1.55;

    double Waveguide_value_V = 1.0;

    bool Use_Predefined_Shape = false;

    unsigned int Number_of_Predefined_Shape = 1;

    // 0: empty, 1: cos()cos(), 2: Hertz Dipole, 3: Waveguide
    unsigned int Point_Source_Type = 0;

    // 1 means normal sweeping, 2 means hierarchical sweeping with depth
    // 1, 3 means hierarchical sweeping with depth 2.
    unsigned int Sweeping_Level = 1;  

    LoggingLevel Logging_Level = LoggingLevel::DEBUG_ALL;

    dealii::Function<3, ComplexNumber> * source_field;

    bool Enable_Parameter_Run = false;

    unsigned int N_Kappa_0_Steps = 20;

    unsigned int Min_HSIE_Order = 1;

    unsigned int Max_HSIE_Order = 10;

    double PML_Sigma_Max = 5.0;

    unsigned int PML_N_Layers = 8;

    double PML_thickness = 1.0;

    unsigned int PML_skaling_order = 3;

    BoundaryConditionType BoundaryCondition = BoundaryConditionType::HSIE;

    bool use_tapered_input_signal = false;

    double tapering_min_z = 0.0;

    double tapering_max_z = 1.0;

    SignalTaperingType Signal_tapering_type = SignalTaperingType::C1;

    SignalCouplingMethod Signal_coupling_method = SignalCouplingMethod::Jump;

    bool prescribe_0_on_input_side = false;

    bool solve_directly = false;
    
    auto complete_data() -> void;
    auto check_validity() -> bool;
};
