#pragma once

/**
 * @file Parameters.h
 * @author Pascal Kraft
 * @brief A struct containing all provided parameter values and some computed values based on it (like MPI rank etc.)
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

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
    /**
     * @brief Node data (z, displacement m, slope v) of the predefined waveguide shape.
     * complete_data() reads it from Modes/test.csv if Use_Predefined_Shape is set and otherwise fills in a straight
     * waveguide.
     */
    ShapeDescription sd;

    /**
     * @brief Absolute tolerance of the residual norm at which the iterative solver stops (entry "solver precision").
     * With the relative convergence criterion it is also the lower bound of the tolerance of the lower sweeping levels.
     */
    double Solver_Precision = 1e-6;

    /**
     * @brief Value of the entry "GMRES restart after". Currently not used: the restart length of GMRES is set to
     * GMRES_max_steps.
     */
    unsigned int GMRES_Steps_before_restart = 30;

    /**
     * @brief Maximum number of iterations of the iterative solver (entry "GMRES maximum steps"). Also used as the
     * restart length of GMRES, so GMRES never restarts.
     */
    unsigned int GMRES_max_steps = 100;

    /** @brief Rank of this process in MPI_COMM_WORLD. */
    unsigned int MPI_Rank;

    /** @brief Number of processes in MPI_COMM_WORLD. Has to be equal to the product of the three Blocks_in_*_direction values. */
    unsigned int NumberProcesses;

    /** @brief Value of the entry "signal amplitude". Currently not used; the mode profile is normalized to a maximum norm of 1. */
    double Amplitude_of_input_signal = 1.0;

    /**
     * @brief If true, the solution is additionally written as "transformed_solution" at the end of the solve
     * (entry "output transformed solution").
     */
    bool Output_transformed_solution = false;

    /** @brief Width of the waveguide core (x-direction) in micrometers (entry "width of waveguide"). The core is centered at x = 0. */
    double Width_of_waveguide = 2.0; // x-direction

    /** @brief Height of the waveguide core (y-direction) in micrometers (entry "height of waveguide"). The core is centered at y = 0. */
    double Height_of_waveguide = 1.8; // y-direction

    /** @brief Not used. There is no parameter file entry for this value. */
    double Horizontal_displacement_of_waveguide = 0; // x-direction

    /**
     * @brief Vertical (y) displacement of the waveguide axis at the output interface relative to the input interface
     * in micrometers (entry "vertical waveguide displacement"). It is the boundary value f(z_max) of the shape function
     * of WaveguideTransformation; f(z_min) = 0 and both end slopes are 0.
     */
    double Vertical_displacement_of_waveguide = 0;

    /** @brief Relative permittivity of the waveguide core (entry "epsilon in"). */
    double Epsilon_R_in_waveguide = 2.3409;

    /** @brief Relative permittivity of the cladding, i.e. of everything outside of the core (entry "epsilon out"). */
    double Epsilon_R_outside_waveguide = 1.8496;

    /**
     * @brief Effective relative permittivity of the guided mode (entry "epsilon effective"). It defines the propagation
     * constant of the exact solution, beta = 2 pi sqrt(Epsilon_R_effective) / Lambda (see ExactSolution::beta()).
     */
    double Epsilon_R_effective = 2.1588449;

    /**
     * @brief The file with the profile of the mode that is used as input signal and as reference solution (entry
     * "mode file"). A relative path is searched in the working directory first and then in the repository. If the
     * header of the file contains the propagation constant, complete_data() sets Epsilon_R_effective from it.
     */
    std::string mode_file = "Modes/output_mode.dat";

    /** @brief The distribution of the permittivity over the cross-section (entry "waveguide cross section"). */
    CrossSectionType cross_section = CrossSectionType::Rectangular;

    /** @brief Relative permeability of the core (entry "mu in"). Read from the case file but not used: the code assumes mu_r = 1. */
    double Mu_R_in_waveguide = 1.0;

    /** @brief Relative permeability of the cladding (entry "mu out"). Read from the case file but not used: the code assumes mu_r = 1. */
    double Mu_R_outside_waveguide = 1.0;

    /**
     * @brief Polynomial degree of the Hardy space infinite elements in the direction normal to the boundary
     * (entry "HSIE polynomial degree"). ParameterSweep overwrites it for every run of the sweep.
     */
    unsigned int HSIE_polynomial_degree = 5;

    /** @brief If true, main() starts an OptimizationRun (shape optimization) instead of a single solve (entry "perform optimization"). */
    bool Perform_Optimization = false;

    /**
     * @brief Instead of optimizing, compare the adjoint based shape gradient with finite differences of the loss
     * functional (entry "perform gradient check"). Only has an effect if Perform_Optimization is true.
     */
    bool Perform_gradient_check = false;

    /** @brief Step width of the central finite differences of the gradient check, applied to the shape parameters (entry "gradient check step width"). */
    double gradient_check_step_width = 0.01;

    /** @brief Maximum number of BFGS steps of an optimization run (entry "optimization maximum steps"). */
    unsigned int optimization_n_shape_steps = 15;

    // Stopping criteria of the optimization (see OptimizationRun::perform_step).
    /**
     * @brief Maximum number of evaluations of the loss functional in an optimization run
     * (entry "optimization maximum evaluations"). Every evaluation requires a forward and an adjoint solve.
     */
    unsigned int optimization_max_evaluations = 40;

    /**
     * @brief The optimization stops when the norm of the shape gradient is below this factor times the norm of the
     * gradient at the initial shape (entry "optimization gradient tolerance").
     */
    double optimization_gradient_tolerance = 1e-2;

    /**
     * @brief The optimization stops when two consecutive BFGS steps each improved the loss by no more than this factor
     * times the absolute value of the loss (entry "optimization loss tolerance").
     */
    double optimization_loss_tolerance = 1e-5;

    /**
     * @brief Tolerance handed to the dealii::SolverControl of the BFGS solver in OptimizationRun. It has no parameter
     * file entry.
     */
    double optimization_residual_tolerance = 1.e-10;

    /**
     * @brief Phase angle (radians) of a kappa_0 with absolute value 1. It is only set and written to the output by
     * ParameterSweep. kappa_0 is not computed from it and the parameter file entry "kappa angle" is not read.
     */
    double kappa_0_angle = 1.0; // For HSIE-Elements. kappa_0 is a complex number with absolute value 1. This is the angle against the real axis.

    /**
     * @brief The complex parameter kappa_0 of the Hardy space infinite elements (entries "Kappa 0 Real" and
     * "Kappa 0 Imaginary"). Waves whose wavenumber normal to the boundary is close to kappa_0 are approximated with the
     * lowest polynomial degree.
     */
    ComplexNumber kappa_0; // For HSIE-Elements

    /** @brief Order of the Nedelec elements (dealii::FE_NedelecSZ) in the interior and in the PML; 0 is the lowest order (entry "fem order"). */
    unsigned int Nedelec_element_order = 0;
    
    /** @brief Number of processes (subdomains) in the z-direction (entry "processes in z"). */
    unsigned int Blocks_in_z_direction = 1;

    /** @brief Number of processes (subdomains) in the x-direction (entry "processes in x"). */
    unsigned int Blocks_in_x_direction = 1;

    /** @brief Number of processes (subdomains) in the y-direction (entry "processes in y"). */
    unsigned int Blocks_in_y_direction = 1;

    /** @brief Index of the subdomain of this process in the x-direction, in [0, Blocks_in_x_direction). Computed from the rank in complete_data(). */
    unsigned int Index_in_x_direction;

    /** @brief Index of the subdomain of this process in the y-direction, in [0, Blocks_in_y_direction). Computed from the rank in complete_data(). */
    unsigned int Index_in_y_direction;

    /** @brief Index of the subdomain of this process in the z-direction, in [0, Blocks_in_z_direction). Computed from the rank in complete_data(). */
    unsigned int Index_in_z_direction;

    /** @brief Number of cells of the interior mesh of one process in the x-direction (entry "cell count x"). */
    unsigned int Cells_in_x = 20;
    
    /** @brief Number of cells of the interior mesh of one process in the y-direction (entry "cell count y"). */
    unsigned int Cells_in_y = 20;
    
    /** @brief Number of cells of the interior mesh of one process in the z-direction (entry "cell count z"). */
    unsigned int Cells_in_z = 20;

    /** @brief Not used. */
    int current_run_number = 0;

    /**
     * @brief Size of the global computational domain (without PML) in the x-direction in micrometers
     * (entry "geometry size x"). The domain is [-Geometry_Size_X/2, Geometry_Size_X/2].
     */
    double Geometry_Size_X = 5;

    /**
     * @brief Size of the global computational domain (without PML) in the y-direction in micrometers
     * (entry "geometry size y"). The domain is [-Geometry_Size_Y/2, Geometry_Size_Y/2].
     */
    double Geometry_Size_Y = 5;

    /**
     * @brief Size of the global computational domain (without PML) in the z-direction in micrometers
     * (entry "geometry size z"). The domain is [global_z_shift, global_z_shift + Geometry_Size_Z].
     */
    double Geometry_Size_Z = 5;

    /**
     * @brief Number of sectors (intervals in z) of the shape description (entry "Number of shape sectors"). The shape
     * functions of WaveguideTransformation are piecewise polynomials on these sectors.
     */
    unsigned int Number_of_sectors = 1;

    /**
     * @brief Length of one sector in the z-direction in micrometers. Computed in complete_data():
     * Geometry_Size_Z / Number_of_sectors, or the distance of the first two nodes of the predefined shape.
     */
    double Sector_thickness;

    /** @brief Not used and never initialized. */
    double Sector_padding;

    /** @brief The constant pi. */
    double Pi = 3.141592653589793238462;

    /**
     * @brief Angular frequency in units in which the speed of light is 1, i.e. the vacuum wavenumber 2 pi / Lambda
     * in 1/micrometer. Computed in complete_data().
     */
    double Omega = 1.0;

    /** @brief Vacuum wavelength in micrometers. It has no parameter file entry; the value 1.55 is fixed. */
    double Lambda = 1.55;

    /** @brief Legacy value (V parameter of a waveguide). It has no parameter file entry and is only read by ModeManager::load(). */
    double Waveguide_value_V = 1.0;

    /**
     * @brief If true, the shape of the waveguide is one of the predefined shapes in Modes/test.csv and the
     * PredefinedShapeTransformation is used (entry "Use predefined shape").
     */
    bool Use_Predefined_Shape = false;

    /**
     * @brief Selects the predefined shape (entry "Predefined case number"). It is the zero-based index of the line of
     * Modes/test.csv that is passed to ShapeDescription::SetByString; it has to be smaller than 36.
     */
    unsigned int Number_of_Predefined_Shape = 1;

    /**
     * @brief Selects the field stored in source_field (entry "source type").
     * 0 and 3: mode of the rectangular dielectric waveguide (ExactSolution; AngledExactSolution for type 3 with the
     * angle transformation), 1: cos()cos() test field, 2: Hertz dipole, 4: TE10 mode of a waveguide with metallic
     * walls.
     */
    unsigned int Point_Source_Type = 0;

    /**
     * @brief Number of levels of the sweeping hierarchy (entry "sweeping level").
     * 1 means normal sweeping (in z), 2 means hierarchical sweeping with depth 1 (sweeping in y inside sweeping in z),
     * 3 means hierarchical sweeping with depth 2 (x inside y inside z).
     */
    unsigned int Sweeping_Level = 1;  

    /**
     * @brief Selects which messages print_info() writes (entry "Logging Level"). With the *_ONE levels only process 0
     * prints, with the *_ALL levels every process does.
     */
    LoggingLevel Logging_Level = LoggingLevel::DEBUG_ALL;

    /**
     * @brief The exact solution or source field of the run, created in complete_data() according to Point_Source_Type.
     * It is used for the input signal (boundary values or right hand side) and for the error computation. The object is
     * owned by this struct but never deleted.
     */
    dealii::Function<3, ComplexNumber> * source_field;

    /**
     * @brief If true, main() starts a ParameterSweep (a series of single-process solves) (entry "Enable Parameter Run").
     * In that case the process count is not checked against the block counts and the block indices are 0.
     */
    bool Enable_Parameter_Run = false;

    /** @brief Number of kappa_0 steps of the ParameterSweep. It has no parameter file entry. */
    unsigned int N_Kappa_0_Steps = 20;

    /** @brief Smallest HSIE polynomial degree of the ParameterSweep (entry "Min HSIE Order"). */
    unsigned int Min_HSIE_Order = 1;

    /** @brief Upper bound (exclusive) of the HSIE polynomial degree of the ParameterSweep (entry "Max HSIE Order"). */
    unsigned int Max_HSIE_Order = 10;

    /**
     * @brief Maximum of the imaginary part of the PML coordinate stretching (entry "PML sigma max"). The stretching
     * factor is 1 + i * PML_Sigma_Max * d^PML_skaling_order with the relative depth d in [0,1] in the PML.
     */
    double PML_Sigma_Max = 5.0;

    /** @brief Number of cell layers of the PML in its normal direction (entry "PML n layers"). */
    unsigned int PML_N_Layers = 8;

    /** @brief Thickness of the PML in micrometers (entry "PML thickness"). */
    double PML_thickness = 1.0;

    /**
     * @brief The factor a of the AngleWaveguideTransformation z' = z - a * y, which tilts the waveguide against the
     * boundaries to test the PML (entry "PML Test Angle").
     */
    double PML_Angle_Test = 0.2;

    /** @brief Exponent with which the imaginary part of the PML stretching grows towards the outer boundary (entry "PML skaling order"). */
    unsigned int PML_skaling_order = 3;

    /** @brief The absorbing boundary method: PML or Hardy space infinite elements (entry "Boundary Method"). */
    BoundaryConditionType BoundaryCondition = BoundaryConditionType::HSIE;

    /**
     * @brief True if the entry "Input Signal Method" is "Taper". Only used for the output; the assembly evaluates
     * Signal_coupling_method.
     */
    bool use_tapered_input_signal = false;

    /**
     * @brief Lower end (z, in micrometers) of the interval in which the input signal is tapered. It has no parameter
     * file entry.
     */
    double tapering_min_z = 0.0;

    /**
     * @brief Upper end (z, in micrometers) of the interval in which the input signal is tapered. It has no parameter
     * file entry.
     */
    double tapering_max_z = 1.0;

    /**
     * @brief The Krylov method of the iterative solvers (entry "solver type" or the override "solver_type").
     * Unknown names select GMRES.
     */
    SolverOptions solver_type = SolverOptions::GMRES;

    /**
     * @brief Shape of the taper of the input signal (entry "Signal tapering type"): C0 is a linear ramp, C1 a cubic
     * ramp with vanishing derivatives at both ends.
     */
    SignalTaperingType Signal_tapering_type = SignalTaperingType::C1;

    /**
     * @brief How the input signal enters the system (entry "Input Signal Method"): as Dirichlet values on the input
     * interface or as a right hand side built from the tapered exact solution.
     */
    SignalCouplingMethod Signal_coupling_method = SignalCouplingMethod::Tapering;

    /** @brief Not used. See tapering_min_z. */
    double tapering_z_min = 0;

    /** @brief Not used. */
    double tapering_t_max = 1;

    /**
     * @brief Value of the entry "Prescribe input zero". It is only written to the log; no part of the assembly
     * evaluates it.
     */
    bool prescribe_0_on_input_side = false;

    /**
     * @brief If true, the iterative solvers of the lower sweeping levels use a tolerance that depends on the current
     * residual of the level above (entry "use relative convergence criterion"). See relative_convergence_criterion.
     */
    bool use_relative_convergence_criterion = false;

    /**
     * @brief Factor of the relative convergence criterion (entry "relative convergence criterion"). The tolerance of a
     * lower level solver is min(1, residual of the level above) times this factor, but at least Solver_Precision.
     */
    double relative_convergence_criterion = 0.01;

    /** @brief If true, main() starts a ConvergenceRun on a sequence of meshes (entry "perform convergence test"). */
    bool Perform_Convergence_Test = false;

    /**
     * @brief Cell count per direction and process of the reference solution of the convergence run. It is the last
     * entry of convergence_cell_counts.
     */
    unsigned int convergence_max_cells = 20;

    /**
     * @brief The space transformation (entry "transformation type"). If Use_Predefined_Shape is set, it is always
     * PredefinedShapeTransformationType.
     */
    TransformationType transformation_type = TransformationType::WavegeuideTransformationType;

    /**
     * @brief Cell counts per direction and process of the meshes of the convergence run
     * (entry "convergence sequence cell count"). The last entry is used for the reference solution.
     */
    std::vector<unsigned int> convergence_cell_counts;

    /**
     * @brief Lower end of the global domain in the z-direction in micrometers (entry "global z shift"). A nonzero value
     * moves the origin, where the Hertz dipole is located, out of the domain.
     */
    double global_z_shift = 0;

    /**
     * @brief If true, the global system is solved with a direct solver instead of the sweeping preconditioned iterative
     * solver (entry "solve directly").
     */
    bool solve_directly = false;

    /**
     * @brief If true, the four side walls (x and y) are perfect electric conductors instead of absorbing boundaries
     * (entry "Lateral boundary").
     */
    bool pec_side_walls = false;

    /**
     * @brief Block low rank compression of the factorizations of the subdomain solvers (MUMPS). 0 switches it off
     * (entry "local solver compression tolerance").
     */
    double local_solver_compression_tolerance = 0.0;

    /**
     * @brief Value of the entry "Optimization Algorithm" (BFGS or Steepest). Currently not evaluated: OptimizationRun
     * always uses BFGS.
     */
    SteppingMethod optimization_stepping_method = SteppingMethod::BFGS;

    /**
     * @brief If false, the height of the waveguide core is a function of z with free shape parameters
     * (entry "constant waveguide height").
     */
    bool keep_waveguide_height_constant = true;

    /**
     * @brief If false, the width of the waveguide core is a function of z with free shape parameters
     * (entry "constant waveguide width").
     */
    bool keep_waveguide_width_constant = true;

    /** @brief If true, the free parameters of the vertical displacement are initialized with random values (entry "Initialize Shape Dofs Randomly"). */
    bool randomly_initialize_shape_dofs = false;
    /**
     * @brief Initial values of the free parameters of the vertical displacement (entry "Initial shape parameters").
     * The number of values has to be equal to the number of free parameters.
     */
    std::vector<double> initial_shape_parameters; // empty: default initial shape
    
    /**
     * @brief Computes the values that are not read from the parameter files.
     * Checks the process count, computes the block indices of this process, creates source_field, computes Omega and
     * fills the shape description sd and Sector_thickness. Throws std::runtime_error if the number of MPI processes
     * does not match the requested number of blocks.
     */
    auto complete_data() -> void;

    /**
     * @brief Intended to check if the mesh resolves the boundary of the waveguide core. It only prints a message.
     * @return bool always true.
     */
    auto check_validity() -> bool;
};
