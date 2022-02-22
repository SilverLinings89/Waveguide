#include "ParameterReader.h"
#include <deal.II/base/patterns.h>

using namespace dealii;

ParameterReader::ParameterReader() { }

void ParameterReader::declare_parameters() {
    run_prm.enter_subsection("Run parameters");
    {
        run_prm.declare_entry("solver precision" , "1e-6", Patterns::Double(), "Absolute precision for solver convergence.");
        run_prm.declare_entry("GMRES restart after" , "30", Patterns::Integer(), "Number of steps until GMRES restarts.");
        run_prm.declare_entry("GMRES maximum steps" , "30", Patterns::Integer(), "Number of maximum GMRES steps until failure.");
        run_prm.declare_entry("use relative convergence criterion", "true", Patterns::Bool(), "If this is set to false, lower level sweeping will ignore higher level current residual.");
        run_prm.declare_entry("relative convergence criterion", "1e-2", Patterns::Double(), "The factor by which a lower level convergence criterion is computed.");
        run_prm.declare_entry("solve directly", "false", Patterns::Bool(), "If this is set to true, GMRES will be replaced by a direct solver.");
        run_prm.declare_entry("kappa angle" , "1.0", Patterns::Double(), "Phase of the complex value kappa with norm 1 that is used in HSIEs.");
        run_prm.declare_entry("processes in x" , "1", Patterns::Integer(), "Number of processes in x-direction.");
        run_prm.declare_entry("processes in y" , "1", Patterns::Integer(), "Number of processes in y-direction.");
        run_prm.declare_entry("processes in z" , "1", Patterns::Integer(), "Number of processes in z-direction.");
        run_prm.declare_entry("sweeping level" , "1", Patterns::Integer(), "Hierarchy level to be used. 1: normal sweeping. 2: two level hierarchy, i.e sweeping in sweeping. 3: three level sweeping, i.e. sweeping in sweeping in swepping.");
        run_prm.declare_entry("cell count x" , "20", Patterns::Integer(), "Number of cells a single process has in x-direction.");
        run_prm.declare_entry("cell count y" , "20", Patterns::Integer(), "Number of cells a single process has in y-direction.");
        run_prm.declare_entry("cell count z" , "20", Patterns::Integer(), "Number of cells a single process has in z-direction.");
        run_prm.declare_entry("output transformed solution", "false", Patterns::Bool(), "If set to true, both the solution in mathematical and in physical coordinates will be written as outputs.");
        run_prm.declare_entry("Logging Level", "Production One", Patterns::Selection("Production One|Production All|Debug One|Debug All"), "Specifies which messages should be printed and by whom.");
        run_prm.declare_entry("solver type", "GMRES", Patterns::Selection("GMRES|MINRES|TFQMR|BICGS|CG|PCONLY"), "Choose the itterative solver to use.");
    }
    run_prm.leave_subsection();

    case_prm.enter_subsection("Case parameters");
    {
        case_prm.declare_entry("source type", "0", Patterns::Integer(), "PointSourceField is 0: empty, 1: cos()cos(), 2: Hertz Dipole, 3: Waveguide");
        case_prm.declare_entry("transformation type", "Waveguide Transformation", Patterns::Selection("Waveguide Transformation|Angle Waveguide Transformation|Bend Transformation"), "Inhomogenous Waveguide Transformation is used for straight waveguide cases and the predefined cases. Angle Waveguide Transformation is a PML test. Bend Transformation is an example for a 90 degree bend.");
        case_prm.declare_entry("geometry size x", "5.0", Patterns::Double(), "Size of the computational domain in x-direction.");
        case_prm.declare_entry("geometry size y", "5.0", Patterns::Double(), "Size of the computational domain in y-direction.");
        case_prm.declare_entry("geometry size z", "5.0", Patterns::Double(), "Size of the computational domain in z-direction.");
        case_prm.declare_entry("epsilon in", "2.3409", Patterns::Double(), "Epsilon r inside the material.");
        case_prm.declare_entry("epsilon out", "1.8496", Patterns::Double(), "Epsilon r outside the material.");
        case_prm.declare_entry("epsilon effective", "2.1588449", Patterns::Double(), "Epsilon r outside the material.");
        case_prm.declare_entry("mu in", "1.0", Patterns::Double(), "Mu r inside the material.");
        case_prm.declare_entry("mu out", "1.0", Patterns::Double(), "Mu r outside the material.");
        case_prm.declare_entry("fem order" , "0", Patterns::Integer(), "Degree of nedelec elements in the interior.");
        case_prm.declare_entry("signal amplitude", "1.0", Patterns::Double(), "Amplitude of the input signal or PointSourceField");
        case_prm.declare_entry("width of waveguide", "2.0", Patterns::Double(), "Width of the Waveguide core.");
        case_prm.declare_entry("height of waveguide", "1.8", Patterns::Double(), "Height of the Waveguide core.");
        case_prm.declare_entry("Enable Parameter Run", "false", Patterns::Bool(), "For a series of Local solves, this can be set to true");
        case_prm.declare_entry("Kappa 0 Real", "1", Patterns::Double(), "Real part of kappa_0 for HSIE.");
        case_prm.declare_entry("Kappa 0 Imaginary", "1", Patterns::Double(), "Imaginary part of kappa_0 for HSIE.");
        case_prm.declare_entry("PML sigma max", "10.0", Patterns::Double(), "Parameter Sigma Max for all PML layers.");
        case_prm.declare_entry("HSIE polynomial degree" , "4", Patterns::Integer(), "Polynomial degree of the Hardy-space polynomials for HSIE surfaces.");
        case_prm.declare_entry("Min HSIE Order", "1", Patterns::Integer(), "Minimal HSIE Element order for parameter run.");
        case_prm.declare_entry("Max HSIE Order", "21", Patterns::Integer(), "Maximal HSIE Element order for parameter run.");
        case_prm.declare_entry("Boundary Method", "HSIE", Patterns::Selection("HSIE|PML"), "Choose the boundary element method (options are PML and HSIE).");
        case_prm.declare_entry("PML thickness", "1.0", Patterns::Double(), "Thickness of PML layers.");
        case_prm.declare_entry("PML skaling order", "3", Patterns::Integer(), "PML skaling order is the exponent with wich the imaginary part grows towards the outer boundary.");
        case_prm.declare_entry("PML n layers", "8", Patterns::Integer(), "Number of cell layers used in the PML medium.");
        case_prm.declare_entry("PML Test Angle", "0.2", Patterns::Double(), "For the angeling test, this is a in z' = z - a * y.");
        case_prm.declare_entry("Input Signal Method", "Dirichlet", Patterns::Selection("Dirichlet|Taper|Jump"), "Taper uses a tapered exact solution to build a right hand side. Dirichlet applies dirichlet boundary values.");
        case_prm.declare_entry("Signal tapering type", "C1", Patterns::Selection("C0|C1"), "Tapering type for signal input");
        case_prm.declare_entry("Prescribe input zero", "false", Patterns::Bool(), "If this is set to true, there will be a dirichlet zero condition enforced on the global input interface (Process index z: 0, boundary id: 4).");
        case_prm.declare_entry("Predefined case number", "1", Patterns::Integer(), "Number in [1,35] that describes the predefined shape to use.");
        case_prm.declare_entry("Use predefined shape", "false", Patterns::Bool(), "If set to true, the geometry for the predefined case from 'Predefined case number' will be used.");
        case_prm.declare_entry("Number of shape sectors", "5", Patterns::Integer(), "Number of sectors for the shape approximation");
        case_prm.declare_entry("perform convergence test", "false", Patterns::Bool(), "If true, the code will perform a cnovergence run on a sequence of meshes.");
        case_prm.declare_entry("convergence sequence cell count", "1,2,4,8,10,14,16,20", Patterns::List(Patterns::Integer()), "The sequence of cell counts in each direction to be used for convergence analysis.");
        case_prm.declare_entry("global z shift", "0", Patterns::Double(), "Shifts the global geometry to remove the center of the dipole for convergence studies.");
        case_prm.declare_entry("Optimization Algorithm", "BFGS", Patterns::Selection("BFGS|Steepest"), "The algorithm to compute the next parametrization in an optimization run.");
        case_prm.declare_entry("Initialize Shape Dofs Randomly", "false", Patterns::Bool(), "If set to true, the shape dofs are initialized to random values.");
        case_prm.declare_entry("perform optimization", "false", Patterns::Bool(), "If true, the code will perform shape optimization.");
        case_prm.declare_entry("vertical waveguide displacement", "0", Patterns::Double(), "The delta of the waveguide core at the input and output interfaces.");
        case_prm.declare_entry("constant waveguide height", "true", Patterns::Bool(), "If false, the waveguide shape will be subject to optimization in the y direction.");
        case_prm.declare_entry("constant waveguide width", "true", Patterns::Bool(), "If false, the waveguide shape will be subject to optimization in the x direction.");
    }
    case_prm.leave_subsection();
}

Parameters ParameterReader::read_parameters(const std::string run_file, const std::string case_file) {
    std::ifstream run_file_stream(run_file, std::ifstream::in);
    std::ifstream case_file_stream(case_file, std::ifstream::in);
    struct Parameters ret;
    declare_parameters();
    run_prm.parse_input(run_file_stream);
    run_prm.enter_subsection("Run parameters");
    {
        
        ret.Solver_Precision = run_prm.get_double("solver precision");
        ret.GMRES_Steps_before_restart = run_prm.get_integer("GMRES restart after");
        ret.GMRES_max_steps = run_prm.get_integer("GMRES maximum steps");
        ret.use_relative_convergence_criterion = run_prm.get_bool("use relative convergence criterion");
        ret.relative_convergence_criterion = run_prm.get_double("relative convergence criterion");
        ret.solve_directly = run_prm.get_bool("solve directly");
        ret.Blocks_in_x_direction = run_prm.get_integer("processes in x");
        ret.Blocks_in_y_direction = run_prm.get_integer("processes in y");
        ret.Blocks_in_z_direction = run_prm.get_integer("processes in z");
        ret.Sweeping_Level = run_prm.get_integer("sweeping level");
        ret.Cells_in_x = run_prm.get_integer("cell count x");
        ret.Cells_in_y = run_prm.get_integer("cell count y");
        ret.Cells_in_z = run_prm.get_integer("cell count z");
        std::string logging = run_prm.get("Logging Level");
        ret.Output_transformed_solution = run_prm.get_bool("output transformed solution");
        if(logging == "Debug One") ret.Logging_Level = LoggingLevel::DEBUG_ONE;
        if(logging == "Debug All") ret.Logging_Level = LoggingLevel::DEBUG_ALL;
        if(logging == "Production One") ret.Logging_Level = LoggingLevel::PRODUCTION_ONE;
        if(logging == "Production All") ret.Logging_Level = LoggingLevel::PRODUCTION_ALL;
        std::string solver_t = run_prm.get("solver type");
        ret.solver_type = solver_option(solver_t);
    }
    case_prm.parse_input(case_file_stream);
    case_prm.enter_subsection("Case parameters");
    {
        ret.Point_Source_Type = case_prm.get_integer("source type");
        std::string trafo_t = case_prm.get("transformation type");
        if(ret.Use_Predefined_Shape || trafo_t == "Predefined Shape Transformation") {
            ret.transformation_type = TransformationType::PredefinedShapeTransformationType;
        } else {
            if(trafo_t == "Waveguide Transformation") {
                ret.transformation_type = TransformationType::WavegeuideTransformationType;
            }
            if(trafo_t == "Angle Waveguide Transformation") {
                ret.transformation_type = TransformationType::AngleWaveguideTransformationType;
            }
            if(trafo_t == "Bend Transformation") {
                ret.transformation_type = TransformationType::BendTransformationType;
            }
        }
        ret.Geometry_Size_X = case_prm.get_double("geometry size x");
        ret.Geometry_Size_Y = case_prm.get_double("geometry size y");
        ret.Geometry_Size_Z = case_prm.get_double("geometry size z");
        ret.Epsilon_R_in_waveguide = case_prm.get_double("epsilon in");
        ret.Epsilon_R_outside_waveguide = case_prm.get_double("epsilon out");
        ret.Epsilon_R_effective = case_prm.get_double("epsilon effective");
        ret.kappa_0 = 0;
        ret.kappa_0.real(case_prm.get_integer("Kappa 0 Real"));
        ret.kappa_0.imag(case_prm.get_integer("Kappa 0 Imaginary"));
        ret.Mu_R_in_waveguide = case_prm.get_double("mu in");
        ret.Mu_R_outside_waveguide = case_prm.get_double("mu out");
        ret.Amplitude_of_input_signal = case_prm.get_double("signal amplitude");
        ret.Nedelec_element_order = case_prm.get_integer("fem order");
        ret.Width_of_waveguide = case_prm.get_double("width of waveguide");
        ret.Height_of_waveguide = case_prm.get_double("height of waveguide");
        ret.Enable_Parameter_Run = case_prm.get_bool("Enable Parameter Run");
        ret.Min_HSIE_Order = case_prm.get_integer("Min HSIE Order");
        ret.Max_HSIE_Order = case_prm.get_integer("Max HSIE Order");
        if(case_prm.get("Boundary Method") == "PML") {
            ret.BoundaryCondition = BoundaryConditionType::PML;
        } else {
            ret.BoundaryCondition = BoundaryConditionType::HSIE;
        }
        ret.PML_N_Layers = case_prm.get_integer("PML n layers");
        ret.PML_skaling_order = case_prm.get_integer("PML skaling order");
        ret.PML_thickness = case_prm.get_double("PML thickness");
        ret.PML_Angle_Test = case_prm.get_double("PML Test Angle");
        ret.HSIE_polynomial_degree = case_prm.get_integer("HSIE polynomial degree");
        ret.Signal_coupling_method = SignalCouplingMethod::Tapering;
        std::string method = case_prm.get("Input Signal Method");
        if(method == "Dirichlet") {
            ret.Signal_coupling_method = SignalCouplingMethod::Dirichlet;
        }
        if(method == "Jump") {
            ret.Signal_coupling_method = SignalCouplingMethod::Jump;
        }
        ret.use_tapered_input_signal = case_prm.get("Input Signal Method") == "Taper";
        ret.PML_Sigma_Max = case_prm.get_double("PML sigma max");
        if(case_prm.get("Signal tapering type") == "C0") {
            ret.Signal_tapering_type = SignalTaperingType::C0;
        }
        ret.Perform_Convergence_Test = case_prm.get_bool("perform convergence test");
        if(ret.Perform_Convergence_Test) {
            std::string cell_counts = case_prm.get("convergence sequence cell count");
            std::vector<std::string> parts = split(cell_counts, ",");
            for(unsigned int i = 0; i < parts.size(); i++) {
                ret.convergence_cell_counts.push_back(std::stoi(parts[i]));
            }
            ret.convergence_max_cells = ret.convergence_cell_counts[ret.convergence_cell_counts.size() -1];
        }
        ret.prescribe_0_on_input_side = case_prm.get_bool("Prescribe input zero");
        ret.Use_Predefined_Shape = case_prm.get_bool("Use predefined shape");
        ret.Number_of_Predefined_Shape = case_prm.get_integer("Predefined case number");
        ret.Number_of_sectors = case_prm.get_integer("Number of shape sectors");
        ret.global_z_shift = case_prm.get_double("global z shift");
        std::string optimization_algorithm = case_prm.get("Optimization Algorithm");
        if(optimization_algorithm == "Steepest") {
            ret.optimization_stepping_method = SteppingMethod::Steepest;
        }
        ret.randomly_initialize_shape_dofs = case_prm.get_bool("Initialize Shape Dofs Randomly");
        ret.Perform_Optimization = case_prm.get_bool("perform optimization");
        ret.Vertical_displacement_of_waveguide = case_prm.get_double("vertical waveguide displacement");
        ret.keep_waveguide_height_constant = case_prm.get_bool("constant waveguide height");
        ret.keep_waveguide_width_constant = case_prm.get_bool("constant waveguide width");
    }
    return ret;
}
