#include <deal.II/base/parameter_handler.h>
#include "ParameterReader.cc"

using namespace dealii;

struct Parameters {
	bool			PRM_O_Grid, PRM_O_Dofs, PRM_O_ActiveCells, PRM_O_VerboseOutput;
	std::string		PRM_M_C_TypeIn, PRM_M_C_TypeOut;
	double			PRM_M_C_RadiusIn, PRM_M_C_RadiusOut;
	int				PRM_M_R_XLength, PRM_M_R_YLength, PRM_M_R_ZLength;
	double			PRM_M_W_Delta, PRM_M_W_EpsilonIn, PRM_M_W_EpsilonOut, PRM_M_W_Lambda;
	std::string		PRM_M_BC_Type;
	double			PRM_M_BC_XYin, PRM_M_BC_XYout, PRM_M_BC_Mantle, PRM_M_BC_KappaXMax, PRM_M_BC_KappaYMax, PRM_M_BC_KappaZMax, PRM_M_BC_SigmaXMax, PRM_M_BC_SigmaYMax, PRM_M_BC_SigmaZMax;
	int				PRM_M_BC_M;
	std::string 	PRM_D_Refinement;
	int 			PRM_D_XY, PRM_D_Z;
	int 			PRM_S_Steps;
	int				PRM_S_GMRESSteps, PRM_S_PreconditionerBlockCount;
	int				PRM_A_Threads;
	double			PRM_S_Precision;
	std::string 	PRM_S_Solver, PRM_S_Preconditioner;
	int 			PRM_M_W_Sectors;
	double 			PRM_C_PI, PRM_C_Eps0, PRM_C_Mu0, PRM_C_c, PRM_C_f0, PRM_C_omega;
	int				PRM_Op_MaxCases;
	double			PRM_Op_InitialStepWidth;
};


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
	
	prm.enter_subsection("Constants");
	{
		ret.PRM_Op_MaxCases = prm.get_integer("MaxCases");
		ret.PRM_Op_InitialStepWidth = prm.get_double("InitialStepWidth");
	}
	prm.leave_subsection();
	
	return ret;
} 

