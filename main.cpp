#ifndef MainCppFlag
#define MainCppFlag

#include "Waveguide.h"
#include "WaveguideStructure.h"
#include "staticfunctions.cpp"
#include "Optimization.h"
#include "Parameters.h"
#include <deal.II/base/parameter_handler.h>
#include "ParameterReader.h"
#include "ParameterReader.cpp"
#include "FileLogger.cpp"
#include "FileLoggerData.cpp"
#include "Optimization.cpp"
#include "ParameterReader.cpp"
#include "Parameters.cpp"
#include "RightHandSide.cpp"
#include "Sector.cpp"
#include "Waveguide.cpp"
#include "WaveguideStructure.cpp"

using namespace dealii;

Parameters GetParameters() {
	ParameterHandler prm;
	ParameterReader param(prm);
	param.read_parameters("parameters.prh");
	struct Parameters ret;
	prm.enter_subsection("Output");
	{
		ret.PRM_O_Grid	=	prm.get_bool("Output Grid");
		ret.PRM_O_Dofs	=	prm.get_bool("Output Dofs");
		ret.PRM_O_ActiveCells	=	prm.get_bool("Output Active Cells");
		ret.PRM_O_VerboseOutput = prm.get_bool("Verbose Output");
	}
	prm.leave_subsection();

	prm.enter_subsection("Measures");
	{
		prm.enter_subsection("Connectors");
		{
			ret.PRM_M_C_TypeIn	= prm.get("Type in");
			ret.PRM_M_C_TypeOut	= prm.get("Type out");
			ret.PRM_M_C_RadiusIn	= prm.get_double("Radius in");
			ret.PRM_M_C_RadiusOut	= prm.get_double("Radius out");
			ret.PRM_M_C_TiltIn	= prm.get_double("Tilt in");
			ret.PRM_M_C_TiltOut	= prm.get_double("Tilt out");
		}
		prm.leave_subsection();

		prm.enter_subsection("Region");
		{
			ret.PRM_M_R_XLength = prm.get_integer("XLength");
			ret.PRM_M_R_YLength = prm.get_integer("YLength");
			ret.PRM_M_R_ZLength = prm.get_integer("ZLength");
		}
		prm.leave_subsection();

		prm.enter_subsection("Waveguide");
		{
				ret.PRM_M_W_Delta = prm.get_double("Delta");
				ret.PRM_M_W_EpsilonIn = prm.get_double("epsilon in");
				ret.PRM_M_W_EpsilonOut = prm.get_double("epsilon out");
				ret.PRM_M_W_Lambda = prm.get_double("Lambda");
				ret.PRM_M_W_Sectors = prm.get_integer("Sectors");
		}
		prm.leave_subsection();

		prm.enter_subsection("Boundary Conditions");
		{
			ret.PRM_M_BC_Type = prm.get("Type");
			ret.PRM_M_BC_XYin = prm.get_double("XY in");
			ret.PRM_M_BC_XYout = prm.get_double("XY out");
			ret.PRM_M_BC_Mantle = prm.get_double("Mantle");
			ret.PRM_M_BC_KappaXMax = prm.get_double("KappaXMax");
			ret.PRM_M_BC_KappaYMax = prm.get_double("KappaYMax");
			ret.PRM_M_BC_KappaZMax = prm.get_double("KappaZMax");
			ret.PRM_M_BC_SigmaXMax = prm.get_double("SigmaXMax");
			ret.PRM_M_BC_SigmaYMax = prm.get_double("SigmaYMax");
			ret.PRM_M_BC_SigmaZMax = prm.get_double("SigmaZMax");
			ret.PRM_M_BC_M = prm.get_integer("DampeningExponentM");
		}
		prm.leave_subsection();

	}
	prm.leave_subsection();

	prm.enter_subsection("Discretization");
	{
		ret.PRM_D_Refinement = prm.get("refinement");
		ret.PRM_D_XY = prm.get_integer("XY");
		ret.PRM_D_Z = prm.get_integer("Z");
	}
	prm.leave_subsection();

	prm.enter_subsection("Assembly");
	{
		ret.PRM_A_Threads = prm.get_integer("Threads");
	}
	prm.leave_subsection();

	prm.enter_subsection("Solver");
	{
		ret.PRM_S_Solver = prm.get("Solver");
		ret.PRM_S_GMRESSteps = prm.get_integer("GMRESSteps");
		ret.PRM_S_Preconditioner = prm.get("Preconditioner");
		ret.PRM_S_PreconditionerBlockCount = prm.get_integer("PreconditionerBlockCount");
		ret.PRM_S_Steps = prm.get_integer("Steps");
		ret.PRM_S_Precision = prm.get_double("Precision");
	}
	prm.leave_subsection();

	prm.enter_subsection("Constants");
	{
		ret.PRM_C_PI = prm.get_double("Pi");
		if(! prm.get_bool("AllOne")){
			ret.PRM_C_Eps0 = prm.get_double("EpsilonZero");
			ret.PRM_C_Mu0 = prm.get_double("MuZero");
			ret.PRM_C_c = 1/sqrt(ret.PRM_C_Eps0 * ret.PRM_C_Mu0);
			ret.PRM_C_f0 = ret.PRM_C_c/0.63;
			ret.PRM_C_omega = 2 * ret.PRM_C_PI * ret.PRM_C_f0;
		} else {
			ret.PRM_C_Eps0 = 1.0;
			ret.PRM_C_Mu0 = 1.0;
			ret.PRM_C_c = 1.0;
			ret.PRM_C_f0 = 1.0;
			ret.PRM_C_omega = 1.0;
		}
	}
	prm.leave_subsection();

	prm.enter_subsection("Optimization");
	{
		ret.PRM_Op_MaxCases = prm.get_integer("MaxCases");
		ret.PRM_Op_InitialStepWidth = prm.get_double("InitialStepWidth");
	}
	prm.leave_subsection();

	return ret;
}


int main (int argc, char *argv[])
{
	// Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	Parameters System_Parameters = GetParameters();
	// Waveguide<PETScWrappers::SparseMatrix, PETScWrappers::Vector > waveguide(prm.PRM);
	// Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::Vector > waveguide(prm.PRM);
	double r_0, r_1, deltaY, epsilon_M, epsilon_K, sectors;
	WaveguideStructure structure(System_Parameters);
	Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> > waveguide(System_Parameters, structure);
	Optimization opt(System_Parameters, waveguide, structure);
	opt.run();
	return 0;
}

#endif
