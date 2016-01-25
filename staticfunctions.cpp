#ifndef StaticFunctionsFlag
#define StaticFunctionsFlag

#include "ParameterReader.h"

using namespace dealii;

std::string constraints_filename 	= "constraints.log";
std::string assemble_filename 		= "assemble.log";
std::string precondition_filename 	= "precondition.log";
std::string solver_filename 		= "solver.log";
std::string total_filename 			= "total.log";

int 	StepsR 			= 10;
int 	StepsPhi 		= 10;

static Parameters GetParameters() {
	ParameterHandler prm;
	ParameterReader param(prm);
	param.read_parameters("./Parameters.xml");
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
			// PRM_M_R_XLength = ret.PRM_M_R_XLength;
			ret.PRM_M_R_YLength = prm.get_integer("YLength");
			// PRM_M_R_YLength = ret.PRM_M_R_YLength;
			ret.PRM_M_R_ZLength = prm.get_integer("ZLength");
			// PRM_M_R_ZLength = ret.PRM_M_R_ZLength;
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
		ret.PRM_S_Library = prm.get("Library");
		ret.PRM_S_Solver = prm.get("Solver");
		ret.PRM_S_GMRESSteps = prm.get_integer("GMRESSteps");
		ret.PRM_S_Preconditioner = prm.get("Preconditioner");
		ret.PRM_S_PreconditionerBlockCount = prm.get_integer("PreconditionerBlockCount");
		ret.PRM_S_Steps = prm.get_integer("Steps");
		ret.PRM_S_Precision = prm.get_double("Precision");
		ret.PRM_S_MPITasks = prm.get_integer("MPITasks");
	}
	prm.leave_subsection();

	prm.enter_subsection("Constants");
	{
		ret.PRM_C_PI = dealii::numbers::PI;
		if(! prm.get_bool("AllOne")){
			ret.PRM_C_Eps0 = prm.get_double("EpsilonZero");
			ret.PRM_C_Mu0 = prm.get_double("MuZero");
			ret.PRM_C_c = 1/sqrt(ret.PRM_C_Eps0 * ret.PRM_C_Mu0);
			ret.PRM_C_f0 = ret.PRM_C_c/0.63;
			ret.PRM_C_omega = 2 * ret.PRM_C_PI * ret.PRM_C_f0;
		} else {
			ret.PRM_C_Eps0 = 1.0;
			ret.PRM_C_Mu0 = 1.0;
			ret.PRM_C_c			= 1.0/sqrt(ret.PRM_C_Eps0 * ret.PRM_C_Mu0);
			ret.PRM_C_f0		= ret.PRM_C_c/ret.PRM_M_W_Lambda; 			// Overwritten by Param-Reader
			ret.PRM_C_k0		= 2.0 * ret.PRM_C_PI * ret.PRM_M_W_Lambda;
			ret.PRM_C_omega 	= 2.0 * ret.PRM_C_PI * ret.PRM_C_f0;
		}
	}
	prm.leave_subsection();

	prm.enter_subsection("Optimization");
	{
		ret.PRM_Op_MaxCases = prm.get_integer("MaxCases");
		ret.PRM_Op_InitialStepWidth = prm.get_double("InitialStepWidth");
		ret.PRM_S_DoOptimization = prm.get_bool("DoOptimization");
	}
	prm.leave_subsection();
	prm.enter_subsection("Mesh-Refinement");
	{
		ret.PRM_R_Global = prm.get_integer("Global");
		ret.PRM_R_Semi = prm.get_integer("SemiGlobal");
		ret.PRM_R_Internal = prm.get_integer("Internal");
	}
	prm.leave_subsection();


	ret.PRM_M_R_XLength = (15.5/4.3)* ((ret.PRM_M_C_RadiusIn+ ret.PRM_M_C_RadiusOut)/2.0);
	ret.PRM_M_R_YLength = (15.5/4.3)* ((ret.PRM_M_C_RadiusIn+ ret.PRM_M_C_RadiusOut)/2.0);
	return ret;
}

static double Distance2D (Point<3> position, Point<3> to = Point<3>()) {
		return sqrt((position(0)-to(0))*(position(0)-to(0)) + (position(1)-to(1))*(position(1)-to(1)));
}

inline Tensor<1, 3 , double> crossproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b) {
	Tensor<1,3,double> ret;
	ret[0] = a[1] * b[2] - a[2] * b[1];
	ret[1] = a[2] * b[0] - a[0] * b[2];
	ret[2] = a[0] * b[1] - a[1] * b[0];
	return ret;
}

inline double dotproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<int dim> static void mesh_info(const Triangulation<dim> &tria, const std::string &filename)
{
	std::cout << "Mesh info:" << std::endl << " dimension: " << dim << std::endl << " no. of cells: " << tria.n_active_cells() << std::endl;
	{
		std::map<unsigned int, unsigned int> boundary_count;
		typename Triangulation<dim>::active_cell_iterator
		cell = tria.begin_active(),
		endc = tria.end();
		for (; cell!=endc; ++cell)
		{
			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
			{
				if (cell->face(face)->at_boundary())
					boundary_count[cell->face(face)->boundary_id()]++;
			}
		}
		std::cout << " boundary indicators: ";
		for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
				it!=boundary_count.end();
				++it)
		{
			std::cout << it->first << "(" << it->second << " times) ";
		}
		std::cout << std::endl;
	}
	std::ofstream out (filename.c_str());
	GridOut grid_out;
	grid_out.write_vtk (tria, out);
	out.close();
	std::cout << " written to " << filename << std::endl << std::endl;
}

static Point<3> Triangulation_Stretch_X (const Point<3> &p)
{
  Point<3> q = p;
  q[0] *= 15.5 * (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) /8.7 ;
  return q;
}


static Point<3> Triangulation_Stretch_Y (const Point<3> &p)
{
  Point<3> q = p;
  q[1] *= 15.5 * (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) /8.7;
  return q;
}


static Point<3> Triangulation_Stretch_Z (const Point<3> &p)
{
  Point<3> q = p;
  q[2] *= ( GlobalParams.PRM_M_BC_XYout + GlobalParams.PRM_M_BC_XYin+ GlobalParams.PRM_M_R_ZLength) /2.0;
  q[2] += ( GlobalParams.PRM_M_BC_XYout - GlobalParams.PRM_M_BC_XYin)/2.0;
  return q;
}

static Point<3> Triangulation_Stretch_Real (const Point<3> &p)
{
	double z = ((p[2] + GlobalParams.PRM_M_R_ZLength/2)/GlobalParams.PRM_M_R_ZLength);
	if(z<0) z=0;
	if(z>1) z = 1;
	double r = (2* GlobalParams.PRM_M_C_RadiusIn - 2*GlobalParams.PRM_M_C_RadiusOut) * z * z * z  - (3* GlobalParams.PRM_M_C_RadiusIn - 3*GlobalParams.PRM_M_C_RadiusOut)*z*z + GlobalParams.PRM_M_C_RadiusIn;
	r *= 2 / (GlobalParams.PRM_M_C_RadiusIn  +GlobalParams.PRM_M_C_RadiusOut);
	double m = (2*GlobalParams.PRM_M_W_Delta)*z*z*z - (3 * GlobalParams.PRM_M_W_Delta)*z*z + GlobalParams.PRM_M_W_Delta/2;
	double sigma = (sqrt(p[0]*p[0] +  p[1]*p[1]) - (1/7.12644))/(r * (GlobalParams.PRM_M_R_YLength - m - r));
	if(sigma < 0.0) sigma = 0.0;
	if(sigma > 1.0) sigma = 1.0;
	Point<3> q = p;
	q[0] =  ( sigma * (r * 7.12644 ) + (1-sigma)* GlobalParams.PRM_M_R_YLength)  * p[0];
	q[1] =  ( sigma * (r * 7.12644 + m) + (1-sigma)* GlobalParams.PRM_M_R_YLength)  * p[1];
	return q;
}

static bool System_Coordinate_in_Waveguide(Point<3> p){
	double value = Distance2D(p);
	return ( value < (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0);
}

static double TEMode00 (Point<3, double> p ,const unsigned int component)
{

	if(component == 0) {
		double d2 = (Distance2D(p)) / (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) ;
		return exp(-d2*d2);
	}
	return 0.0;
}

static double Solution (Point<3, double> p ,const unsigned int component)
{
	if(component == 0) {
		double res =TEMode00(p,0);
		std::complex<double> i (0,1);
		std::complex<double> ret = -1.0 * res * std::exp(-i*(GlobalParams.PRM_C_PI/2) *(p[2]+GlobalParams.PRM_M_R_ZLength/2.0 + GlobalParams.PRM_M_BC_XYin));
		return ret.real();
	}
	if(component == 3) {
		double res =TEMode00(p,0);
		std::complex<double> i (0,1);
		std::complex<double> ret = -1.0 * res * std::exp(-i*(GlobalParams.PRM_C_PI/2) *(p[2]+GlobalParams.PRM_M_R_ZLength/2.0 + GlobalParams.PRM_M_BC_XYin));
		return ret.imag();
	}

	return 0.0;
}

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}



#endif
