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
double 	PI 				= 1.0;				// Overwritten by Param-Reader
double 	Eps0 			= 1.0;				// Overwritten by Param-Reader
double 	Mu0 			= 1.0; 				// Overwritten by Param-Reader
double 	c 				= 1/sqrt(Eps0 * Mu0); 	// Overwritten by Param-Reader
double 	f0 				= c/0.63; 			// Overwritten by Param-Reader
double 	omega 			= 2 * PI * f0; 	// Overwritten by Param-Reader
int 	PRM_M_R_XLength = 2.0;		// Overwritten by Param-Reader
int 	PRM_M_R_YLength = 2.0;		// Overwritten by Param-Reader
int 	PRM_M_R_ZLength = 2.0;		// Overwritten by Param-Reader

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
					boundary_count[cell->face(face)->boundary_indicator()]++;
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
	std::cout << " written to " << filename << std::endl << std::endl;
}

static Point<3> Triangulation_Stretch_X (const Point<3> &p)
{
  Point<3> q = p;
  q[0] *= PRM_M_R_XLength /2.0;
  return q;
}


static Point<3> Triangulation_Stretch_Y (const Point<3> &p)
{
  Point<3> q = p;
  q[1] *= PRM_M_R_YLength /2.0;
  return q;
}


static Point<3> Triangulation_Stretch_Z (const Point<3> &p)
{
  Point<3> q = p;
  q[2] *= PRM_M_R_ZLength /2.0;
  return q;
}

static bool System_Coordinate_in_Waveguide(Point<3> p){
	double value = Distance2D(p);
	double reference = PRM_M_R_XLength * 4.35 /15.5;
	return ( value < reference);
}

static double TEMode00 (Point<3, double> p ,const unsigned int component)
{
	if(System_Coordinate_in_Waveguide(p)){
		if(p[2] < 0) {
			if(component == 0) {
				double d2 = Distance2D(p) * 2.0 /(PRM_M_R_XLength *4.35 / 15.5);
				//return 1.0;
				return exp(-d2*d2);
			}
		}
	}
	return 0.0;
}


inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}



#endif
