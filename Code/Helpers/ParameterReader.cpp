#ifndef ParameterReaderCppFlag
#define ParameterReaderCppFlag

#include "ParameterReader.h"

using namespace dealii;

ParameterReader::ParameterReader	(ParameterHandler &prmhandler ): prm(prmhandler)	 {

}

void ParameterReader::declare_parameters()
{
  prm.enter_subsection("Output");
      prm.enter_subsection("Optimization");
          prm.enter_subsection("Gnuplot");
              prm.declare_entry("Optimization History Live", "false", Patterns::Bool() , "Currently not implemented. This will open an X-window at runtime and show the current shape to allow for full tracking of the current procedure while it runs (to be able to abort as early as possible.");
              prm.declare_entry("Optimization History Shapes", "true", Patterns::Bool() , "If this value is set to 'true', after every step a plot and a data file of the current shape will be generated. The plot shows single (tubular) or multiple (rectangular) crossections of the current waveguide shape.");
              prm.declare_entry("Optimization History", "true", Patterns::Bool() , "If this value is set to 'true', a plot and data file will be generated logging the values of the signal quality after every step of the current optimization scheme.");
          prm.leave_subsection();
          prm.enter_subsection("VTK");
              prm.enter_subsection("TransformationWeights");
                  prm.declare_entry("TransformationWeightsAll", "false", Patterns::Bool() , "If this is enabled, a .vtk file is generated in every step logging the norm of the transformation tensor as 3D data.");
                  prm.declare_entry("TransformationWeightsFirst", "false", Patterns::Bool() , "If this is enabled, a .vtk file is generated in the first step logging the norm of the transformation tensor as 3D data.");
                  prm.declare_entry("TransformationWeightsLast", "false", Patterns::Bool() , "If this is enabled, a .vtk file is generated in the last step logging the norm of the transformation tensor as 3D data.");
              prm.leave_subsection();
              prm.enter_subsection("Solution");
                  prm.declare_entry("SolutionAll", "false", Patterns::Bool() , "If this is enabled, a .vtk file is generated in every step logging the solution as 3D data.");
                  prm.declare_entry("SolutionFirst", "true", Patterns::Bool() , "If this is enabled, a .vtk file is generated in the first step logging the solution as 3D data.");
                  prm.declare_entry("SolutionLast", "true", Patterns::Bool() , "If this is enabled, a .vtk file is generated in the last step logging the solution as 3D data.");
              prm.leave_subsection();
          prm.leave_subsection();
      prm.leave_subsection();
      prm.enter_subsection("Convergence");
          prm.enter_subsection("DataFiles");
              prm.declare_entry("ConvergenceFirst", "false", Patterns::Bool() , "This causes the code to generate a datafile with the data of the system matrix solution process of the first step.");
              prm.declare_entry("ConvergenceLast", "false", Patterns::Bool() , "This causes the code to generate a datafile with the data of the system matrix solution process of the last step.");
              prm.declare_entry("ConvergenceAll", "false", Patterns::Bool() , "This causes the code to generate a datafile with the data of the system matrix solution process of all steps.");
          prm.leave_subsection();
          prm.enter_subsection("Plots");
              prm.declare_entry("ConvergenceFirst", "false", Patterns::Bool() , "This causes the code to generate a plot of the data of the system matrix solution process of the first step.");
              prm.declare_entry("ConvergenceLast", "false", Patterns::Bool() , "This causes the code to generate a plot of the data of the system matrix solution process of the last step.");
              prm.declare_entry("ConvergenceAll", "false", Patterns::Bool() , "This causes the code to generate a plot of the data of the system matrix solution process of all steps.");
          prm.leave_subsection();
      prm.leave_subsection();
      prm.enter_subsection("General");
          prm.declare_entry("SummaryFile", "true", Patterns::Bool() , "This generates an output of the simulation (mathematical terms. Convergence rates, residuals, parameters etc.)");
          prm.declare_entry("LogFile", "true", Patterns::Bool() , "This generates a general log (Code steps, warnings, errors, diagnostics)");
      prm.leave_subsection();
  prm.leave_subsection();
  prm.enter_subsection("Measures");
      prm.enter_subsection("Connectors");
          prm.declare_entry("Shape", "Circle", Patterns::Selection("Circle|Rectangle"), "Describes the shape of the input connector.");
          prm.declare_entry("Dimension1 In", "2.0", Patterns::Double(0), "First dimension of the input connector. For a circular waveguide this is the radius. For a rectangular waveguide this is the width.");
          prm.declare_entry("Dimension2 In", "2.0", Patterns::Double(0), "Second dimension of the input connector. For a circular waveguide this has no meaning. For a rectangular waveguide this is the height.");
          prm.declare_entry("Dimension1 Out", "2.0", Patterns::Double(0), "First dimension of the output connector. For a circular waveguide this is the radius. For a rectangular waveguide this is the width.");
          prm.declare_entry("Dimension2 Out", "2.0", Patterns::Double(0), "Second dimension of the output connector. For a circular waveguide this has no meaning. For a rectangular waveguide this is the height.");
      prm.leave_subsection();
      prm.enter_subsection("Region");
          prm.declare_entry("XLength", "10.0", Patterns::Double(0), "Length of the system in x-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
          prm.declare_entry("YLength", "10.0", Patterns::Double(0), "Length of the system in y-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
          prm.declare_entry("ZLength", "6.0", Patterns::Double(0), "Length of the system in z-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
      prm.leave_subsection();
      prm.enter_subsection("Waveguide");
          prm.declare_entry("Delta", "1.0", Patterns::Double(0), "Offset between the two connectors measured in micrometres.");
          prm.declare_entry("epsilon in", "2.21", Patterns::Double(0), "Material-Property of the optical fiber (optical thickness).");
          prm.declare_entry("epsilon out", "2.2", Patterns::Double(0), "Material-Property of environment of the fiber (optical thickness).");
          prm.declare_entry("Lambda", "5.6328", Patterns::Double(0), "Vacuum-wavelength of the incoming wave.");
          prm.declare_entry("Sectors", "2", Patterns::Integer(1), "Number of Sectors used for Modelling of the Waveguide.");
      prm.leave_subsection();
      prm.enter_subsection("Boundary Conditions");
          prm.declare_entry("Type", "PML", Patterns::Selection("PML|HSIE"), "The way the output-connector is modeled. HSIE uses the Hardy-space infinite element for setting boundary conditions but isn't implemented yet.");
          prm.declare_entry("ZPlus", "1", Patterns::Integer(0), "Thickness of the PML area on the side of the output connector. Measused in sectors of normal size of a sector.");
          prm.declare_entry("XMinus", "1.0", Patterns::Double(0), "Thickness of the PML on the negative X-axis. Measured in micrometers");
          prm.declare_entry("XPlus", "1.0", Patterns::Double(0), "Thickness of the PML on the positive X-axis. Measured in micrometers");
          prm.declare_entry("YMinus", "1.0", Patterns::Double(0), "Thickness of the PML on the negative Y-axis. Measured in micrometers");
          prm.declare_entry("YPlus", "1.0", Patterns::Double(0), "Thickness of the PML on the positive Y-axis. Measured in micrometers");
          prm.declare_entry("KappaXMax", "10.0", Patterns::Double(0), "PML Tuning Parameter");
          prm.declare_entry("KappaYMax", "10.0", Patterns::Double(0), "PML Tuning Parameter");
          prm.declare_entry("KappaZMax", "10.0", Patterns::Double(0), "PML Tuning Parameter");
          prm.declare_entry("SigmaXMax", "10.0", Patterns::Double(0), "PML Tuning Parameter");
          prm.declare_entry("SigmaYMax", "10.0", Patterns::Double(0), "PML Tuning Parameter");
          prm.declare_entry("SigmaZMax", "10.0", Patterns::Double(0), "PML Tuning Parameter");
          prm.declare_entry("DampeningExponentM", "3", Patterns::Integer(3), "Dampening Exponent M for the intensety of dampening in the PML region.");
      prm.leave_subsection();
  prm.leave_subsection();
  prm.enter_subsection("Schema");
      prm.declare_entry("Homogeneity", "false", Patterns::Bool() , "If this is enabled, a space transformation is used which is equal the identity on the PML-region for the dampening along the x and y axis.");
      prm.declare_entry("Optimization Schema", "Adjoint", Patterns::Selection("Adjoint|FD"), "If this is set to adjoint, the shape gradient will be computed by means of an adjoint based method. If it is set to FD, finite differences are use.");
      prm.declare_entry("Optimization Steps", "10", Patterns::Integer(1), "Number of Optimization steps to be performed.");
      prm.declare_entry("Stepping Method", "Steepest", Patterns::Selection("Steepest|CG|LineSearch"), "Method of step computation. Steepest uses steepest descent. CG uses a conjugate gradient method to compute the next step. Line Search only works based on an adjoint optimization setting, where searches can be performed cheaply.");
      // prm.declare_entry("Step Width", "Adjoint", Patterns::Selection("Adjoint|Experimental"), "This parameter descibes the scheme used to compute the next step width. This can be adjoint (if an adjoint schema is used, which causees the computation of multiple shape hradient with differing step widths in the parameters. This would be too costly for FD and ist therefore not available in that mode. An experimental approach can be used which tries to use information from the gradient and a seperate step width control to compute the step. ");
  prm.leave_subsection();
  prm.enter_subsection("Solver");
      prm.declare_entry("Solver", "GMRES", Patterns::Selection("GMRES|UMFPACK|MINRES"), "Which Solver to use for the solution of the system matrix");
      prm.declare_entry("GMRESSteps", "30", Patterns::Integer(1), "Steps until restart of Krylow subspace generation");
      prm.declare_entry("Preconditioner", "Sweeping", Patterns::Selection("Sweeping|Amesos_Lapack|Amesos_Scalapack|Amesos_Klu|Amesos_Umfpack|Amesos_Pardiso|Amesos_Taucs|Amesos_Superlu|Amesos_Superludist|Amesos_Dscpack|Amesos_Mumps"), "Which preconditioner to use");
      prm.declare_entry("Steps", "30", Patterns::Integer(1), "Number of Steps the Solver is supposed to do.");
      prm.declare_entry("Precision", "1e-6", Patterns::Double(0), "Minimal error value, the solver is supposed to accept as correct solution.");
  prm.leave_subsection();
  prm.enter_subsection("Constants");
      prm.declare_entry("AllOne", "true", Patterns::Bool() , "If this is set to true, EpsilonZero and MuZero are set to 1.");
      prm.declare_entry("EpsilonZero", "8.854e-18", Patterns::Double(0), "Physical constant Epsilon zero. The standard value is measured in micrometers.");
      prm.declare_entry("MuZero", "1.257e-12", Patterns::Double(0), "Physical constant Mu zero. The standard value is measured in micrometers.");
      prm.declare_entry("Pi", "3.14159265", Patterns::Double(0), "Mathematical constant Pi.");
  prm.leave_subsection();
  prm.enter_subsection("Refinement");
      prm.declare_entry("Global", "1", Patterns::Integer(0), "Global refinement-steps.");
      prm.declare_entry("SemiGlobal", "1", Patterns::Integer(0), "Semi-Global refinement-steps (close to the Waveguide-boundary and inside).");
      prm.declare_entry("Internal", "1", Patterns::Integer(0), "Internal refinement-steps.");
  prm.leave_subsection();
}

void ParameterReader::read_parameters(const std::string inputfile) {
	declare_parameters();
	std::ifstream ifile (inputfile, std::ifstream::in);
	prm.parse_input_from_xml(ifile);
}

#endif
