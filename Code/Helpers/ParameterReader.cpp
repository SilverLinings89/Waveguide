#include "ParameterReader.h"
#include <deal.II/base/patterns.h>

using namespace dealii;

ParameterReader::ParameterReader() {

}

void ParameterReader::declare_parameters() {
    run_prm.enter_subsection("Run parameters");
    {
        run_prm.declare_entry("perform optimization", "false", Patterns::Bool(), "If true, the code will perform shape optimization.");
        run_prm.declare_entry("solver precision" , "1e-6", Patterns::Double(), "Absolute precision for solver convergence.");
        run_prm.declare_entry("GMRES restart after" , "30", Patterns::Integer(), "Number of steps until GMRES restarts.");
        run_prm.declare_entry("GMRES maximum steps" , "100", Patterns::Integer(), "Number of maximum GMRES steps until failure.");
        run_prm.declare_entry("HSIE polynomial degree" , "4", Patterns::Integer(), "Polynomial degree of the Hardy-space polynomials for HSIE surfaces.");
        run_prm.declare_entry("kappa angle" , "1.0", Patterns::Double(), "Phase of the complex value kappa with norm 1 that is used in HSIEs.");
        run_prm.declare_entry("fem order" , "0", Patterns::Integer(), "Degree of nedelec elements in the interior.");
        run_prm.declare_entry("processes in x" , "1", Patterns::Integer(), "Number of processes in x-direction.");
        run_prm.declare_entry("processes in y" , "1", Patterns::Integer(), "Number of processes in y-direction.");
        run_prm.declare_entry("processes in z" , "1", Patterns::Integer(), "Number of processes in z-direction.");
        run_prm.declare_entry("HSIE sweeping level" , "1", Patterns::Integer(), "Hierarchy level to be used. 1: normal sweeping. 2: two level hierarchy, i.e sweeping in sweeping. 3: three level sweeping, i.e. sweeping in sweeping in swepping.");
        run_prm.declare_entry("cell count x" , "20", Patterns::Integer(), "Number of cells a single process has in x-direction.");
        run_prm.declare_entry("cell count y" , "20", Patterns::Integer(), "Number of cells a single process has in y-direction.");
        run_prm.declare_entry("cell count z" , "20", Patterns::Integer(), "Number of cells a single process has in z-direction.");
    }
    run_prm.leave_subsection();

    case_prm.enter_subsection("Case parameters");
    {
        case_prm.declare_entry("source type", "0", Patterns::Integer(), "PointSourceField is 0: empty, 1: cos()cos(), 2: Hertz Dipole, 3: Waveguide");
        case_prm.declare_entry("geometry size x", "5.0", Patterns::Double(), "Size of the computational domain in x-direction.");
        case_prm.declare_entry("geometry size y", "5.0", Patterns::Double(), "Size of the computational domain in y-direction.");
        case_prm.declare_entry("geometry size z", "5.0", Patterns::Double(), "Size of the computational domain in z-direction.");
        case_prm.declare_entry("epsilon in", "1.0", Patterns::Double(), "Epsilon r inside the material.");
        case_prm.declare_entry("epsilon out", "1.0", Patterns::Double(), "Epsilon r outside the material.");
        case_prm.declare_entry("mu in", "1.0", Patterns::Double(), "Mu r inside the material.");
        case_prm.declare_entry("mu out", "1.0", Patterns::Double(), "Mu r outside the material.");
        case_prm.declare_entry("signal amplitude", "1.0", Patterns::Double(), "Amplitude of the input signal or PointSourceField");
        case_prm.declare_entry("width of waveguide", "2.0", Patterns::Double(), "Width of the Waveguide core.");
        case_prm.declare_entry("height of waveguide", "1.8", Patterns::Double(), "Height of the Waveguide core.");
        case_prm.declare_entry("Enable Parameter Run", "false", Patterns::Bool(), "For a series of Local solves, this can be set to true");
        case_prm.declare_entry("N Kappa 0 Steps", "20", Patterns::Integer(), "Steps for kappa discretization.");
        case_prm.declare_entry("Min HSIE Order", "1", Patterns::Integer(), "Minimal HSIE Element order for parameter run.");
        case_prm.declare_entry("Max HSIE Order", "21", Patterns::Integer(), "Maximal HSIE Element order for parameter run.");
    }
    case_prm.leave_subsection();
}

Parameters ParameterReader::read_parameters(const std::string run_file,const std::string case_file) {
    std::ifstream run_file_stream(run_file, std::ifstream::in);
    std::ifstream case_file_stream(case_file, std::ifstream::in);
    struct Parameters ret;
    declare_parameters();
    run_prm.parse_input(run_file_stream);
    run_prm.enter_subsection("Run parameters");
    {
        ret.Perform_Optimization = run_prm.get_bool("perform optimization");
        ret.Solver_Precision = run_prm.get_double("solver precision");
        ret.GMRES_Steps_before_restart = run_prm.get_integer("GMRES restart after");
        ret.GMRES_max_steps = run_prm.get_integer("GMRES maximum steps");
        ret.HSIE_polynomial_degree = run_prm.get_integer("HSIE polynomial degree");
        ret.kappa_0_angle = run_prm.get_double("kappa angle");
        ret.Nedelec_element_order = run_prm.get_integer("fem order");
        ret.Blocks_in_x_direction = run_prm.get_integer("processes in x");
        ret.Blocks_in_y_direction = run_prm.get_integer("processes in y");
        ret.Blocks_in_z_direction = run_prm.get_integer("processes in z");
        ret.HSIE_SWEEPING_LEVEL = run_prm.get_integer("HSIE sweeping level");
        ret.Cells_in_x = run_prm.get_integer("cell count x");
        ret.Cells_in_y = run_prm.get_integer("cell count y");
        ret.Cells_in_z = run_prm.get_integer("cell count z");
    }
    case_prm.parse_input(case_file_stream);
    case_prm.enter_subsection("Case parameters");
    {
        ret.Point_Source_Type = case_prm.get_integer("source type");
        ret.Geometry_Size_X = case_prm.get_double("geometry size x");
        ret.Geometry_Size_Y = case_prm.get_double("geometry size y");
        ret.Geometry_Size_Z = case_prm.get_double("geometry size z");
        ret.Epsilon_R_in_waveguide = case_prm.get_double("epsilon in");
        ret.Epsilon_R_outside_waveguide = case_prm.get_double("epsilon out");
        ret.Mu_R_in_waveguide = case_prm.get_double("mu in");
        ret.Mu_R_outside_waveguide = case_prm.get_double("mu out");
        ret.Amplitude_of_input_signal = case_prm.get_double("signal amplitude");
        ret.Width_of_waveguide = case_prm.get_double("width of waveguide");
        ret.Width_of_waveguide = case_prm.get_double("height of waveguide");
    }

    return ret;
}
