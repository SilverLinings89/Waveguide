#ifndef ParameterReaderCppFlag
#define ParameterReaderCppFlag

#include "ParameterReader.h"

using namespace dealii;

ParameterReader::ParameterReader	(ParameterHandler &prmhandler ): prm(prmhandler)	 {

}

void ParameterReader::declare_parameters()
{
	prm.enter_subsection("Output");
		prm.declare_entry("Output Grid", "false", Patterns::Bool() , "Determines if Grid should be written to .eps file for visualization.");
		prm.declare_entry("Output Dofs", "true", Patterns::Bool() , "Determines if details about Degrees of freedom should be written to the console.");
		prm.declare_entry("Output Active Cells", "true", Patterns::Bool() , "Determines if the number of active cells should be written to the console.");
		prm.declare_entry("Verbose Output", "true", Patterns::Bool() , "Determines if a lot of helpful data should be written to the console.");
	prm.leave_subsection();

	prm.enter_subsection("Measures");
		prm.enter_subsection("Connectors");
			prm.declare_entry("Type in", "Circle", Patterns::Selection("Circle|Ellipse|Square"), "Describes the shape of the input connector.");
			prm.declare_entry("Type out", "Circle", Patterns::Selection("Circle|Ellipse|Square"), "Describes the shape of the input connector.");
			prm.declare_entry("Radius in", "2.5", Patterns::Double(0), "Radius / Diameter for Circle / Square input connector. Ellipse not implemented.");
			prm.declare_entry("Radius out", "2.5", Patterns::Double(0), "Radius / Diameter for Circle / Square output connector. Ellipse not implemented.");
			prm.declare_entry("Tilt in", "0.0", Patterns::Double(0), "Tilt of the input connector. In the documentation this values is named v_0.");
			prm.declare_entry("Tilt out", "0.0", Patterns::Double(0), "Tilt of the output connector. In the documentation this values is named v_1.");
		prm.leave_subsection();

		prm.enter_subsection("Region");
			prm.declare_entry("XLength", "10", Patterns::Integer(0), "Length of the system in x-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
			prm.declare_entry("YLength", "10", Patterns::Integer(0), "Length of the system in y-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
			prm.declare_entry("ZLength", "2", Patterns::Integer(0), "Length of the system in z-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
		prm.leave_subsection();

		prm.enter_subsection("Waveguide");
			prm.declare_entry("Delta", "1.0", Patterns::Double(0), "Offset between the two connectors measured in micrometres.");
			prm.declare_entry("epsilon in", "2.2", Patterns::Double(0), "Material-Property of the optical fiber (optical thickness).");
			prm.declare_entry("epsilon out", "1.0", Patterns::Double(0), "Material-Property of environment of the fiber (optical thickness).");
			prm.declare_entry("Lambda", "5.6328", Patterns::Double(0), "Vacuum-wavelength of the incoming wave.");
			prm.declare_entry("Sectors", "2", Patterns::Integer(1), "Number of Sectors used for Modelling of the Waveguide.");
		prm.leave_subsection();

		prm.enter_subsection("Boundary Conditions");
			prm.declare_entry("Type", "PML", Patterns::Selection("PML|HSIE"), "The way the output-connector is modeled. HSIE uses the Hardy-space infinite element for setting boundary conditions but isn't implemented yet.");
			prm.declare_entry("XY in" , "0.2" , Patterns::Double(0), "Thickness of the PML area on the side of the input connector.");
			prm.declare_entry("XY out" , "0.2" , Patterns::Double(0), "Thickness of the PML area on the side of the output connector.");
			prm.declare_entry("Mantle" , "10" , Patterns::Double(0), "Thickness of the PML area on 4 non-connector sides, the mantle.");
			prm.declare_entry("KappaXMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("KappaYMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("KappaZMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("SigmaXMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("SigmaYMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("SigmaZMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("DampeningExponentM", "3" , Patterns::Integer(0), "Dampening Exponent M for the intensety of dampening in the PML region.");
		prm.leave_subsection();

	prm.leave_subsection();

	prm.enter_subsection("Discretization");
		prm.declare_entry("refinement", "adaptive", Patterns::Selection("global|adaptive"), "This value describes if the XY-plane discretization should be refined homogeneously or adaptively. The latter is not implemented yet.");
		prm.declare_entry("XY", "2", Patterns::Integer(1), "Number of refinement steps used in the XY-plane.");
		prm.declare_entry("Z" , "3", Patterns::Integer(1), "Number of layers in the z-direction.");
	prm.leave_subsection();

	prm.enter_subsection("Assembly");
		prm.declare_entry("Threads", "4", Patterns::Integer(1), "Number of threads used in the assembly process.");
	prm.leave_subsection();

	prm.enter_subsection("Solver");
		prm.declare_entry("Solver", "GMRES", Patterns::Selection("CG|GMRES|UMFPACK|Richardson|Relaxation"), "Which Solver to use for the solution of the system matrix");
		prm.declare_entry("GMRESSteps", "1200", Patterns::Integer(1), "Steps until restart of Krylow subspace generation");
		prm.declare_entry("Preconditioner", "Identity", Patterns::Selection("SOR|SSOR|Identity|Jacobi|ILU|Block_Jacobi|ParaSails|LU|ICC|BoomerAMG|Eisenstat"), "Which preconditioner to use");
		prm.declare_entry("PreconditionerBlockCount", "20", Patterns::Integer(1), "Number of Blocks for Block-Preconditioners.");
		prm.declare_entry("Steps", "90000", Patterns::Integer(1), "Number of Steps the Solver is supposed to do.");
		prm.declare_entry("Precision", "1e-5", Patterns::Double(0), "Minimal error value, the solver is supposed to accept as correct solution.");
	prm.leave_subsection();

	prm.enter_subsection("Constants");
		prm.declare_entry("AllOne", "true", Patterns::Bool(), "If this is set to true, EpsilonZero and MuZero are set to 1.");
		prm.declare_entry("EpsilonZero", "8.854e-18", Patterns::Double(0), "Physical constant Epsilon zero. The standard value is measured in micrometers.");
		prm.declare_entry("MuZero", "1.257e-12", Patterns::Double(0), "Physical constant Mu zero. The standard value is measured in micrometers.");
		prm.declare_entry("Pi", "3.14159265", Patterns::Double(0), "Mathematical constant Pi.");
	prm.leave_subsection();
	
	prm.enter_subsection("Optimization");
		prm.declare_entry("InitialStepWidth", "1.0", Patterns::Double(0), "Step width for the first step.");
		prm.declare_entry("MaxCases", "100", Patterns::Integer(1), "Number of Cases the Optimization is supposed to use.");
	prm.leave_subsection();


}


void ParameterReader::read_parameters(const std::string inputfile) {
	declare_parameters();
	//std::cout << "Test" << std::endl;
	//prm.print_parameters(std::cout, ParameterHandler::OutputStyle::XML);
	std::ifstream ifile (inputfile, std::ifstream::in);
	prm.read_input_from_xml(ifile);
}

#endif