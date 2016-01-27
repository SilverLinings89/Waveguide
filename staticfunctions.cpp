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

inline double InterpolationPolynomial(double in_z, double in_val_zero, double in_val_one, double in_derivative_zero, double in_derivative_one) {
	if (in_z < 0.0) return in_val_zero;
	if (in_z > 1.0) return in_val_one;
	return (2*(in_val_zero - in_val_one) + in_derivative_zero + in_derivative_one) * pow(in_z,3) + (3*(in_val_one - in_val_zero) - (2*in_derivative_zero) - in_derivative_one)*pow(in_z,2) + in_derivative_zero*in_z + in_val_zero;
}

inline double InterpolationPolynomialZeroDerivative(double in_z , double in_val_zero, double in_val_one) {
	return InterpolationPolynomial(in_z, in_val_zero, in_val_one, 0.0, 0.0);
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

/**
static Point<3> Triangulation_Stretch_X (const Point<3> &p)
{
  Point<3> q = p;
  q[0] *= 15.5 * (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) /8.7 ;
  return q;
}

static Point<3> Triangulation_Stretch_Y (const Point<3> &p)
{
  Point<3> q = p;
  q[1] *= 15.5 * (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) /8.7 ;
  return q;
}
**/

static double sigma (double in_z, double min, double max) {
	if( min == max ) return (in_z < min )? 0.0 : 1.0;
	if(in_z < min) return 0.0;
	if(in_z > max) return 1.0;
	double ret = 0;
	ret = (in_z - min) / ( max - min);
	if(ret < 0.0) ret = 0.0;
	if(ret > 1.0) ret = 1.0;
	return ret;
}


static Point<3> Triangulation_Stretch_X (const Point<3> &p)
{
  Point<3> q = p;
  q[0] *= GlobalParams.PRM_M_R_XLength / 2.0 ;
  return q;
}

static Point<3> Triangulation_Stretch_Y (const Point<3> &p)
{
  Point<3> q = p;
  q[1] *= GlobalParams.PRM_M_R_YLength / 2.0 ;
  return q;
}

static Point<3> Triangulation_Stretch_Z (const Point<3> &p)
{
  Point<3> q = p;
  double total_length = structure.System_Length();
  q[2] *= total_length / 2.0;
  return q;
}

static Point<3> Triangulation_Shift_Z (const Point<3> &p)
{
  Point<3> q = p;
  double sector_length = structure.Sector_Length();
  q[2] += (sector_length/2.0) * GlobalParams.PRM_M_BC_XYout;
  return q;
}

static Point<3> Triangulation_Stretch_Real_Radius (const Point<3> &p)
{
	double r_goal = structure.get_r(structure.Z_to_Sector_and_local_z(p[2]).second);
	double shift = structure.get_m(structure.Z_to_Sector_and_local_z(p[2]).second);
	double r_current = (GlobalParams.PRM_M_R_XLength / 2.0 ) / 7.12644;
	double r_max = (GlobalParams.PRM_M_R_XLength / 2.0 ) * (1.0 - GlobalParams.PRM_M_BC_Mantle);
	double r_point = sqrt(p[0]*p[0] + p[1]*p[1]);
	double stretch = sigma(r_point, r_current, r_max);
	double factor = stretch * r_goal/r_current + (1-stretch);
	Point<3> q = p;
	q[0] *= factor;
	q[1] *= factor;
	q[1] += factor * shift;
	return p;
}


static Point<3> Triangulation_Stretch_Computational_Radius (const Point<3> &p)
{
	double r_goal = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0;
	double r_current = (GlobalParams.PRM_M_R_XLength ) / 7.12644;
	double r_max = (GlobalParams.PRM_M_R_XLength / 2.0 ) * (1.0 - GlobalParams.PRM_M_BC_Mantle);
	double r_point = sqrt(p[0]*p[0] + p[1]*p[1]);
	double factor = InterpolationPolynomialZeroDerivative(sigma(r_point, r_current, r_max), r_goal/r_current , 1.0);
	Point<3> q = p;
	q[0] *= factor;
	q[1] *= factor;
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
