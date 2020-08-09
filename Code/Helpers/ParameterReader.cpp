#ifndef ParameterReaderCppFlag
#define ParameterReaderCppFlag

#include "ParameterReader.h"
#include <deal.II/base/patterns.h>

using namespace dealii;

ParameterReader::ParameterReader(ParameterHandler &prmhandler)
    : prm(prmhandler) {}

void ParameterReader::declare_parameters() {
    prm.enter_subsection("Run parameters");
    {
        prm.declare_entry("Perform optimization", "false", Patterns::Bool());
    }
    prm.leave_subsection();

    prm.enter_subsection("Scheme properties");
    {
        prm.declare_entry("Kappa angle", "1.0", Patterns::Double());
        prm.declare_entry("HSIE polynomial degree", "5", Patterns::Integer());
        prm.declare_entry("Processes in x", "1", Patterns::Integer());
        prm.declare_entry("Processes in y", "1", Patterns::Integer());
        prm.declare_entry("Processes in z", "2", Patterns::Integer());
        prm.declare_entry("HSIE sweeping level", "1", Patterns::Integer());
    }
    prm.leave_subsection();

    prm.enter_subsection("Waveguide properties");
    {
        prm.declare_entry("Width of waveguide", "1.0", Patterns::Double());
        prm.declare_entry("Heigth of waveguide", "1.0", Patterns::Double());
        prm.declare_entry("X shift", "0.0", Patterns::Double());
        prm.declare_entry("Y shift", "0.0", Patterns::Double());
        prm.declare_entry("epsilon in", "1.0", Patterns::Double());
        prm.declare_entry("epsilon out", "1.0", Patterns::Double());
        prm.declare_entry("mu in", "1.0", Patterns::Double());
        prm.declare_entry("mu out", "1.0", Patterns::Double());
        prm.declare_entry("mode amplitude", "1.0", Patterns::Double());
        prm.declare_entry("geometry size x", "1.0", Patterns::Double());
        prm.declare_entry("geometry size y", "1.0", Patterns::Double());
        prm.declare_entry("geometry size z", "1.0", Patterns::Double());
    }
    prm.leave_subsection();
}

void ParameterReader::read_parameters(const std::string inputfile) {
    declare_parameters();
    std::ifstream ifile(inputfile, std::ifstream::in);
    prm.parse_input_from_xml(ifile);
}

#endif
