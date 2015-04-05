/**
 * This Code is the core file of the Waveguide-Problem Solver. It is the main part of my Masters Thesis implementation.
 * The Goal of this project is, to solve an optimization problem concerning the perfect shape for minimal signal-loss in an S-shaped waveguide.
 * The solver is implemented using the code-framework DealII (see <a href="http://dealii.org/">here</a> for more information). Mathematically the main idea is, to focus on the shape-representation problem. One  of the biggest problem in any solution of this problem, is to represent the used shapes optimally. In this case we will always use the same discretization because we use transformation optics to transform the real physical problem on a s-shaped waveguide with simple material-parameters, to a mathematical representation, consisting of a cylindrical quasi-waveguide made up of an inhomogenous material.
 *
 * @author Pascal Kraft
 * @date 2015
 */
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/lac/matrix_out.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>

using namespace dealii;

const double PI =  3.141592653589793238462643383279502884197169399;
//const double Eps0 = 1;
//const double Mu0 = 1;

const double Eps0 = 8.854e-18;
const double Mu0 = 1.257e-12;
const double c = 2.998e14;
const double f0 = c/0.63;
const double omega = 2 * PI * f0;

template<int dim> void mesh_info(const Triangulation<dim> &tria, const std::string &filename)
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

/**
 * The ParameterReader contains all data required for the reading and exposing of parameter values. It encapsulates a ParameterHandler object for reading the file and some more functionality for ease of use. It is used to load a file a runtime an insert constant values. This removes the necessity of recompiling upon changing parameter values.
 */
class ParameterReader : public Subscriptor
{
public:

	/**
	 * The constructor takes a ParameterHandler object (simple to instantiate)
	 * @param prmhandler This object handles reading the input file.
	 */
	ParameterReader	(ParameterHandler & prmhandler);

	/**
	 * This function handles the workflow of the parameter insertion process.
	 * @param inputfile The filename of the parameterfile. In this case I choose to adress relatively.
	 */
	void read_parameters	(const std::string inputfile);

private: 

	/**
	 * This function gives the implementation of most of the functionality. It declares all the values, that should be read from the file and which variables they should be parsed into.
	 */
	void declare_parameters();
	ParameterHandler &prm;

};


/*
 * Distance in 2D. Since I have a cylindrical system, the distance to the middle-axis is an important criterium.
 * @param position this is the main point to calculate the distance for. Usually this will be used to calculate the distance to the 2D origin, in which the second argument is left out.
 * @param to if set, the result will not be positions distance to the origin but the xy-flat distance of vectors position and to.
 */
static double Distance2D (Point<3> position, Point<3> to = Point<3>()) {
		return sqrt((position(0)-to(0))*(position(0)-to(0)) + (position(1)-to(1))*(position(1)-to(1)));
}


/**
 * This function calculates the inverse of an imaginary number. It gives back the real OR imaginary component, depending on the arguments.
 * @param in_real - Real part of the input number.
 * @param in_imaginary - Imaginary part of the input number.
 * @param real - determines if real or imaginary part should be given back.
 */
ParameterReader::ParameterReader	( ParameterHandler &prmhandler) : 	prm(prmhandler) {}

void ParameterReader::declare_parameters	()
{
	prm.enter_subsection("Output");
	{
		prm.declare_entry("Output Grid", "false", Patterns::Bool() , "Determines if Grid should be written to .eps file for visualization.");
		prm.declare_entry("Output Dofs", "false", Patterns::Bool() , "Determines if details about Degrees of freedom should be written to the console.");
		prm.declare_entry("Output Active Cells", "false", Patterns::Bool() , "Determines if the number of active cells should be written to the console.");
		prm.declare_entry("Verbose Output", "false", Patterns::Bool() , "Determines if a lot of helpful data should be written to the console.");
	}
	prm.leave_subsection();

	prm.enter_subsection("Measures");
	{
		prm.enter_subsection("Connectors");
		{
			prm.declare_entry("Type in", "Circle", Patterns::Selection("Circle|Ellipse|Square"), "Describes the shape of the input connector.");
			prm.declare_entry("Type out", "Circle", Patterns::Selection("Circle|Ellipse|Square"), "Describes the shape of the input connector.");
			prm.declare_entry("Radius in", "1.5", Patterns::Double(0), "Radius / Diameter for Circle / Square input connector. Ellipse not implemented.");
			prm.declare_entry("Radius out", "1.5", Patterns::Double(0), "Radius / Diameter for Circle / Square output connector. Ellipse not implemented.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Region");
		{
			prm.declare_entry("XLength", "10", Patterns::Integer(0), "Length of the system in x-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
			prm.declare_entry("YLength", "10", Patterns::Integer(0), "Length of the system in y-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
			prm.declare_entry("ZLength", "450", Patterns::Integer(0), "Length of the system in z-Direction (Connectors lie in the XY-plane and the offset lies in the y-direction. Measured in micrometres");
		}
		prm.leave_subsection();

		prm.enter_subsection("Waveguide");
		{
			prm.declare_entry("Delta", "0.0", Patterns::Double(0), "Offset between the two connectors measured in micrometres.");
			prm.declare_entry("epsilon in", "2.2", Patterns::Double(0), "Material-Property of the optical fiber (optical thickness).");
			prm.declare_entry("epsilon out", "1.0", Patterns::Double(0), "Material-Property of environment of the fiber (optical thickness).");
			prm.declare_entry("Lambda", "0.6328", Patterns::Double(0), "Vacuum-wavelength of the incoming wave.");
		}
		prm.leave_subsection();

		prm.enter_subsection("Boundary Conditions");
		{
			prm.declare_entry("Type", "PML", Patterns::Selection("PML|HSIE"), "The way the output-connector is modeled. HSIE uses the Hardy-space infinite element for setting boundary conditions but isn't implemented yet.");
			prm.declare_entry("XY in" , "10.0" , Patterns::Double(0), "Thickness of the PML area on the side of the input connector.");
			prm.declare_entry("XY out" , "10.0" , Patterns::Double(0), "Thickness of the PML area on the side of the output connector.");
			prm.declare_entry("Mantle" , "4.0" , Patterns::Double(0), "Thickness of the PML area on 4 non-connector sides, the mantle.");
			prm.declare_entry("KappaXMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("KappaYMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("KappaZMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("SigmaXMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("SigmaYMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("SigmaZMax" , "10.0" , Patterns::Double(0), "PML Tuning Parameter");
			prm.declare_entry("DampeningExponentM", "3" , Patterns::Integer(0), "Dampening Exponent M for the intensety of dampening in the PML region.");
		}
		prm.leave_subsection();

	}
	prm.leave_subsection();

	prm.enter_subsection("Discretization");
	{
		prm.declare_entry("refinement", "global", Patterns::Selection("global|adaptive"), "This value describes if the XY-plane discretization should be refined homogeneously or adaptively. The latter is not implemented yet.");
		prm.declare_entry("XY", "4", Patterns::Integer(1), "Number of refinement steps used in the XY-plane.");
		prm.declare_entry("Z" , "100", Patterns::Integer(1), "Number of layers in the z-direction.");
	}
	prm.leave_subsection();
}

void ParameterReader::read_parameters(const std::string inputfile) {
	declare_parameters();
	prm.read_input(inputfile);
}

template <int dim>
class RightHandSide : public Function<dim>
{
	public:
		RightHandSide () : Function<dim>(6) {}
		virtual double value (const Point<dim> &p, const unsigned int component ) const;
		virtual void vector_value (const Point<dim> &p,	Vector<double> &value) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p , const unsigned int component) const
{
	if(p[2] < 0){
		if(p(0)*p(0) + p(1)*p(1) < 0.2828){
			if(component == 0 || component == 4) return (p(0)*p(0) + p(1)*p(1))*(0.2828- (p(0)*p(0) + p(1)*p(1)))/(p(0)*p(0)) * 1e5 ;
			if(component == 1 || component == 3) return (p(0)*p(0) + p(1)*p(1))*(0.2828- (p(0)*p(0) + p(1)*p(1)))/(p(1)*p(1)) * 1e5;

		}
			else return 0.0;
	}
	return 0.0;
}

template <int dim>
void RightHandSide<dim>::vector_value (const Point<dim> &p,	Vector<double> &values) const
{
	for (unsigned int c=0; c<6; ++c) values(c) = RightHandSide<dim>::value (p, c);
}

class Waveguide
{
	public:
		Waveguide (ParameterHandler &);
		void run ();

	private:
		void 	read_values ();
		void 	make_grid ();
		void 	setup_system ();
		void 	assemble_system ();
		void 	solve ();
		void 	output_results () const;
		bool	PML_in_X(Point<3> &);
		bool	PML_in_Y(Point<3> &);
		bool	PML_in_Z(Point<3> &);
		double 	PML_X_Distance(Point<3> &);
		double 	PML_Y_Distance(Point<3> &);
		double 	PML_Z_Distance(Point<3> &);
		double  RHS_value(const Point<3> &, const unsigned int component);
		Tensor<2,3, std::complex<double>> get_Tensor(Point<3> &, bool, bool);
		Tensor<2,3, std::complex<double>> Transpose_Tensor(Tensor<2,3, std::complex<double>> );
		Tensor<1,3, std::complex<double>> Transpose_Vector(Tensor<1,3, std::complex<double>> );
		Point<3> Triangulation_Stretch_X (const Point<3> &);
		Point<3> Triangulation_Stretch_Y (const Point<3> &);
		Point<3> Triangulation_Stretch_Z (const Point<3> &);

		Triangulation<3>	triangulation;
		FESystem<3>		fe;
		DoFHandler<3>		dof_handler;

		SparsityPattern			sparsity_pattern;
		SparseMatrix<double>	system_matrix;
		ParameterHandler 		&prm;
		ConstraintMatrix boundary_value_constraints_imaginary;
		ConstraintMatrix boundary_value_constraints_real;

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
		Vector<double>	solution;
		Vector<double>	system_rhs;
};

Waveguide::Waveguide (ParameterHandler &param)
  :
  fe (FE_Nedelec<3> (0), 2),
  dof_handler (triangulation),
  prm(param)
{ }

Tensor<2,3, std::complex<double>> Waveguide::get_Tensor(Point<3> & position, bool inverse , bool epsilon) {
	Tensor<2,3, std::complex<double>> ret;
	std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);

	double omegaepsilon0 = (2* PI / PRM_M_W_Lambda) * 2.998 * 8.85418781762e-4;
	std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);
	if(PML_in_X(position)){
		double r,d, sigmax;
		r = PML_X_Distance(position);
		d = PRM_M_R_XLength * 2.0;
		sigmax = pow(d/r , PRM_M_BC_M) * PRM_M_BC_SigmaXMax;
		sx.real( 1 + pow(d/r , PRM_M_BC_M) * PRM_M_BC_KappaXMax);
		sx.imag( sigmax / (PRM_M_W_EpsilonOut * omegaepsilon0));
		S1 /= sx;
		S2 *= sx;
		S3 *= sx;
	}
	if(PML_in_Y(position)){
		double r,d, sigmay;
		r = PML_Y_Distance(position);
		d = PRM_M_R_YLength * 2.0;
		sigmay = pow(d/r , PRM_M_BC_M) * PRM_M_BC_SigmaYMax;
		sy.real( 1 + pow(d/r , PRM_M_BC_M) * PRM_M_BC_KappaYMax);
		sy.imag( sigmay / (PRM_M_W_EpsilonOut * omegaepsilon0));
		S1 *= sy;
		S2 /= sy;
		S3 *= sy;
	}
	if(PML_in_Z(position)){
		double r,d, sigmaz;
		r = PML_Z_Distance(position);
		d = PRM_M_R_ZLength * 2.0;
		sigmaz = pow(d/r , PRM_M_BC_M) * PRM_M_BC_SigmaZMax;
		sz.real( 1 + pow(d/r , PRM_M_BC_M) * PRM_M_BC_KappaZMax);
		sz.imag( sigmaz / (((Distance2D(position)< 0.2828)?PRM_M_W_EpsilonIn : PRM_M_W_EpsilonOut) * omegaepsilon0 ));
		S1 *= sz;
		S2 *= sz;
		S3 /= sz;
	}

	if(inverse) {
		std::complex<double> temp(1.0, 0.0);
		S1 = temp / S1;
		S2 = temp / S2;
		S3 = temp / S3;
	}

	ret[0][0] = S1;
	ret[1][1] = S2;
	ret[2][2] = S3;

	if(inverse) {
		if(epsilon) {
			if(position(0)*position(0) + position(1)*position(1) < 0.2828 ) {
				ret *= PRM_M_W_EpsilonIn;
			} else {
				ret *= PRM_M_W_EpsilonOut;
			}
			ret *= Eps0;
		} else {
			ret*= Mu0;
		}
	} else {
		if(epsilon) {
			if(position(0)*position(0) + position(1)*position(1) < 0.2828 ) {
				ret /= PRM_M_W_EpsilonIn;
			} else {
				ret /= PRM_M_W_EpsilonOut;
			}
			ret /= Eps0;
		} else {
			ret /= Mu0;
		}
	}


	return ret;
}

Tensor<2,3, std::complex<double>> Waveguide::Transpose_Tensor(Tensor<2,3, std::complex<double>> input) {
	Tensor<2,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		for(int j = 0; j<3; j++){
			ret[i][j].real(input[i][j].real());
			ret[i][j].imag( - input[i][j].imag());
		}
	}
	return ret;
}

Tensor<1,3, std::complex<double>> Waveguide::Transpose_Vector(Tensor<1,3, std::complex<double>> input) {
	Tensor<1,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		ret[i].real(input[i].real());
		ret[i].imag( - input[i].imag());

	}
	return ret;
}

bool Waveguide::PML_in_X(Point<3> &p) {
	return  p(0) > ((PRM_M_R_XLength / 2.0) - (PRM_M_R_XLength * PRM_M_BC_Mantle/100.0)) ||  p(0) < (-(PRM_M_R_XLength / 2.0) + (PRM_M_R_XLength * PRM_M_BC_Mantle/100.0));
}

bool Waveguide::PML_in_Y(Point<3> &p) {
	return  p(1) > ((PRM_M_R_YLength / 2.0) - (PRM_M_R_YLength * PRM_M_BC_Mantle/100.0)) ||  p(1) < (-(PRM_M_R_YLength / 2.0) + (PRM_M_R_YLength * PRM_M_BC_Mantle/100.0));
}

bool Waveguide::PML_in_Z(Point<3> &p) {
	return  p(2) > ((PRM_M_R_ZLength / 2.0) - (PRM_M_R_ZLength * PRM_M_BC_XYout/100.0)) ||  (p(2) < (-(PRM_M_R_ZLength / 2.0) + (PRM_M_R_ZLength * PRM_M_BC_XYin/100.0)  && Distance2D(p)<0.2828));
}

double Waveguide::PML_X_Distance(Point<3> &p){
	if(p(0) >0){
		return p(0) - ((PRM_M_R_XLength / 2.0) - (PRM_M_R_XLength* PRM_M_BC_Mantle/100.0));
	} else {
		return -(p(0) + ((PRM_M_R_XLength / 2.0) - (PRM_M_R_XLength* PRM_M_BC_Mantle/100.0)));
	}
}

double Waveguide::PML_Y_Distance(Point<3> &p){
	if(p(1) >0){
			return p(1) - ((PRM_M_R_YLength / 2.0) - (PRM_M_R_YLength* PRM_M_BC_Mantle/100.0));
		} else {
			return -(p(1) + ((PRM_M_R_YLength / 2.0) - (PRM_M_R_YLength* PRM_M_BC_Mantle/100.0)));
		}
}

double Waveguide::PML_Z_Distance(Point<3> &p){
	if(p(2) >0){
			return p(2) - ((PRM_M_R_ZLength / 2.0) - (PRM_M_R_ZLength * PRM_M_BC_XYout/100));
		} else {
			return -(p(2) + ((PRM_M_R_ZLength / 2.0) - (PRM_M_R_ZLength * PRM_M_BC_XYin/100)));
		}
}

double Waveguide::RHS_value (const Point<3> &p , const unsigned int component)
{
	if(p[2] < 0){
		if(p(0)*p(0) + p(1)*p(1) < 0.2828){
			if(component == 0 || component == 4) return (p(0)*p(0) + p(1)*p(1))*(0.2828- (p(0)*p(0) + p(1)*p(1)))/(p(0)*p(0)) * 1e5 ;
			if(component == 1 || component == 3) return (p(0)*p(0) + p(1)*p(1))*(0.2828- (p(0)*p(0) + p(1)*p(1)))/(p(1)*p(1)) * 1e5;

		}
			else return 0.0;
	}
	return 0.0;
}

Point<3> Waveguide::Triangulation_Stretch_X (const Point<3> &p)
{
  Point<3> q = p;
  q[0] *= PRM_M_R_XLength /2.0;
  return q;
}

Point<3> Waveguide::Triangulation_Stretch_Y (const Point<3> &p)
{
  Point<3> q = p;
  q[1] *= PRM_M_R_YLength /2.0;
  return q;
}

Point<3> Waveguide::Triangulation_Stretch_Z (const Point<3> &p)
{
  Point<3> q = p;
  q[2] *= PRM_M_R_ZLength /2.0;
  return q;
}

void Waveguide::read_values() {
	prm.enter_subsection("Output");
	{
		PRM_O_Grid	=	prm.get_bool("Output Grid");
		PRM_O_Dofs	=	prm.get_bool("Output Dofs");
		PRM_O_ActiveCells	=	prm.get_bool("Output Active Cells");
		PRM_O_VerboseOutput = prm.get_bool("Verbose Output");
	}
	prm.leave_subsection();

	prm.enter_subsection("Measures");
	{
		prm.enter_subsection("Connectors");
		{
			PRM_M_C_TypeIn	= prm.get("Type in");
			PRM_M_C_TypeOut	= prm.get("Type out");
			PRM_M_C_RadiusIn	= prm.get_double("Radius in");
			PRM_M_C_RadiusOut	= prm.get_double("Radius out");
		}
		prm.leave_subsection();

		prm.enter_subsection("Region");
		{
			PRM_M_R_XLength = prm.get_integer("XLength");
			PRM_M_R_YLength = prm.get_integer("YLength");
			PRM_M_R_ZLength = prm.get_integer("ZLength");
		}
		prm.leave_subsection();

		prm.enter_subsection("Waveguide");
		{
				PRM_M_W_Delta = prm.get_double("Delta");
				PRM_M_W_EpsilonIn = prm.get_double("epsilon in");
				PRM_M_W_EpsilonOut = prm.get_double("epsilon out");
				PRM_M_W_Lambda = prm.get_double("Lambda");
		}
		prm.leave_subsection();

		prm.enter_subsection("Boundary Conditions");
		{
			PRM_M_BC_Type = prm.get("Type");
			PRM_M_BC_XYin = prm.get_double("XY in");
			PRM_M_BC_XYout = prm.get_double("XY out");
			PRM_M_BC_Mantle = prm.get_double("Mantle");
			PRM_M_BC_KappaXMax = prm.get_double("KappaXMax");
			PRM_M_BC_KappaYMax = prm.get_double("KappaYMax");
			PRM_M_BC_KappaZMax = prm.get_double("KappaZMax");
			PRM_M_BC_SigmaXMax = prm.get_double("SigmaXMax");
			PRM_M_BC_SigmaYMax = prm.get_double("SigmaYMax");
			PRM_M_BC_SigmaZMax = prm.get_double("SigmaZMax");
			PRM_M_BC_M = prm.get_integer("DampeningExponentM");
		}
		prm.leave_subsection();

	}
	prm.leave_subsection();

	prm.enter_subsection("Discretization");
	{
		PRM_D_Refinement = prm.get("refinement");
		PRM_D_XY = prm.get_integer("XY");
		PRM_D_Z = prm.get_integer("Z");
	}
	prm.leave_subsection();

}

void Waveguide::make_grid ()
{
	const double outer_radius = 1.0;
	GridGenerator::subdivided_hyper_cube (triangulation, 5, -outer_radius, outer_radius);

	static const CylindricalManifold<3, 3> round_description(2, 0.0001);
	unsigned int temp = 1;
	triangulation.set_manifold (temp, round_description);
	Triangulation<3>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();

	for (; cell!=endc; ++cell){
		double distance_from_center = 0;
		for( int j = 0; j<4; j++) distance_from_center += Distance2D(Point<3> (cell->vertex(j)));
		if (distance_from_center < 3 ) {
			cell->set_all_manifold_ids(1);
		}
	}

	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		double distance_from_center = 0;
		for( int j = 0; j<4; j++) distance_from_center += Distance2D(Point<3> (cell->vertex(j)));
		if (distance_from_center < 1.2) {
			cell->set_manifold_id(0);
		}
	}

	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		if(cell->at_boundary()){
			for(int j = 0; j<6; j++){
				if(cell->face(j)->at_boundary()){
					Point<3> ctr =cell->face(j)->center(true, false);
					if(Distance2D(ctr) < 0.28){
						if(ctr(2) < 0) cell->face(j)->set_all_boundary_indicators(1);
						else cell->face(j)->set_all_boundary_indicators(2);
					}
				}
			}
		}
	}

	if(PRM_D_Refinement == "global") triangulation.refine_global (PRM_D_XY-1);

	if(PRM_O_Grid) {
		if(PRM_O_VerboseOutput) std::cout<< "Writing Mesh data to file \"grid-3D.vtk\"" << std::endl;
		mesh_info(triangulation, "grid-3D.vtk");
		if(PRM_O_VerboseOutput) std::cout<< "Done" << std::endl;
	}


}

void Waveguide::setup_system ()
{
	if(PRM_O_VerboseOutput && PRM_O_Dofs) {
		std::cout << "Distributing Degrees of freedom." << std::endl;
	}
	dof_handler.distribute_dofs (fe);
	if(PRM_O_Dofs) {
		std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
	}

	if(PRM_O_VerboseOutput) {
		std::cout << "Calculating compressed Sparsity Pattern..." << std::endl;
	}

	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
	sparsity_pattern.copy_from(c_sparsity);

	system_matrix.reinit (sparsity_pattern);

	solution.reinit (dof_handler.n_dofs());
	system_rhs.reinit (dof_handler.n_dofs());
	if(PRM_O_VerboseOutput) {
			std::cout << "Done." << std::endl;
	}
}

void Waveguide::assemble_system ()
{
	QGauss<3>  quadrature_formula(2);
	const FEValuesExtractors::Vector real (0);
	const FEValuesExtractors::Vector imag (3);
	FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;

	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();

	if(PRM_O_VerboseOutput) {
		std::cout << "Dofs per cell: " << dofs_per_cell << std::endl << "Quadrature Formula Size: " << n_q_points << std::endl;
		std::cout << "Dofs per face: " << fe.dofs_per_face << std::endl << "Dofs per line: " << fe.dofs_per_line << std::endl;
	}

	FullMatrix<double>	cell_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>		cell_rhs (dofs_per_cell);
	Tensor<2,3, std::complex<double>> 		epsilon, mu;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	DoFHandler<3>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
    {
		fe_values.reinit (cell);
		quadrature_points = fe_values.get_quadrature_points();
		cell_matrix = 0;
		cell_rhs = 0;

		for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
			epsilon = get_Tensor(quadrature_points[q_index],  false, true);
			mu = get_Tensor(quadrature_points[q_index], true, false);
			const double JxW = fe_values.JxW(q_index);
			for (unsigned int i=0; i<dofs_per_cell; i++){
				Tensor<1,3, std::complex<double>> I_Curl;
				Tensor<1,3, std::complex<double>> I_Val;
				for(int k = 0; k<3; k++){
					I_Curl[k].imag(fe_values[imag].curl(i, q_index)[k]);
					I_Curl[k].real(fe_values[real].curl(i, q_index)[k]);
					I_Val[k].imag(fe_values[imag].value(i, q_index)[k]);
					I_Val[k].real(fe_values[real].value(i, q_index)[k]);
				}



				for (unsigned int j=0; j<dofs_per_cell; j++){
					Tensor<1,3, std::complex<double>> J_Curl;
					Tensor<1,3, std::complex<double>> J_Val;
					for(int k = 0; k<3; k++){
						J_Curl[k].imag(fe_values[imag].curl(j, q_index)[k]);
						J_Curl[k].real(fe_values[real].curl(j, q_index)[k]);
						J_Val[k].imag(fe_values[imag].value(j, q_index)[k]);
						J_Val[k].real(fe_values[real].value(j, q_index)[k]);
					}
					//for(int kl = 0; kl<3; kl++) std::cout << I_Curl[kl];
					//std::cout << std::endl;
					cell_matrix(i,j) += ((mu * I_Curl) * Transpose_Vector(J_Curl) ).real() *JxW ;
					cell_matrix(i,j) -= (I_Val  *(epsilon * Transpose_Vector(J_Val))).real() /(omega * omega);

				}
			}
        }



		cell->get_dof_indices (local_dof_indices);

		for (unsigned int i=0; i<dofs_per_cell; ++i)
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

		for (unsigned int i=0; i<dofs_per_cell; ++i)
			system_rhs(local_dof_indices[i]) += cell_rhs(i);

    }

	std::vector<bool> boundary_dofs (dof_handler.n_dofs());
	std::set<unsigned char> boundary_indicators;
	boundary_indicators.insert (0);
	boundary_indicators.insert (2);
	DoFTools::extract_boundary_dofs (dof_handler, fe.component_mask(real), boundary_dofs, boundary_indicators);
	std::map<unsigned int, double> boundary_values;
	for (unsigned int i=0; i<dof_handler.n_dofs(); ++i) if (boundary_dofs[i] == true) boundary_values[i] = 0.;
	DoFTools::extract_boundary_dofs (dof_handler, fe.component_mask(imag), boundary_dofs, boundary_indicators);
	for (unsigned int i=0; i<dof_handler.n_dofs(); ++i) if (boundary_dofs[i] == true) boundary_values[i] = 0.;
	MatrixTools::apply_boundary_values (boundary_values, system_matrix, solution, system_rhs);

/**
	boundary_dofs.clear();
	boundary_indicators.clear();
	boundary_indicators.insert (1);
	DoFTools::extract_boundary_dofs (dof_handler, fe.component_mask(real), boundary_dofs, boundary_indicators);
	boundary_values.clear();
	for (unsigned int i=0; i<dof_handler.n_dofs(); ++i) if (boundary_dofs[i] == true) boundary_values[i] = 1.;
	DoFTools::extract_boundary_dofs (dof_handler, fe.component_mask(imag), boundary_dofs, boundary_indicators);
	for (unsigned int i=0; i<dof_handler.n_dofs(); ++i) if (boundary_dofs[i] == true) boundary_values[i] = 1.;
	MatrixTools::apply_boundary_values (boundary_values, system_matrix, solution, system_rhs);
**/

	int counter = 0;
	boundary_values.clear();
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if(cell->at_boundary()){
			std::vector<unsigned int> current_dofs(dofs_per_cell);
			cell->get_dof_indices(current_dofs);
			for(int i = 0; i < 24; i++) {
				std::pair<unsigned int , double> real_the_boundary;
				real_the_boundary.first = current_dofs[i];
				real_the_boundary.second = 0;
				boundary_values.insert(real_the_boundary);
			}
		}
		if(cell->at_boundary() && Distance2D(cell->center(true, false)) < 0.2828 && cell->center(true, false)(2) < 0 ){

			std::vector<unsigned int> current_dofs(dofs_per_cell);
			cell->get_dof_indices(current_dofs);

			if(counter  == 0){
				std::cout<< "DofIndices: ";
				for(int i = 0; i< dofs_per_cell; i++) {
					std::cout<< current_dofs[i] << " ";
				}
				std::cout << std::endl;

			}
			for(int i = 0; i< 6; i++){
				if(cell->face(i)->center()(2) < 0){
					for(int j = 0; j<4; j++){
						Point<3> p1 = cell->face(i)->line(j)->vertex(0);
						Point<3> p2 = cell->face(i)->line(j)->vertex(1);
						Point<3> con = p2-p1;
						Point<3> center = cell->face(i)->line(j)->center();
						if(counter  == 0){
							std::cout<< "P1: ";
							for(int i = 0; i< 3; i++) {
								std::cout<< p1[i] << " ";
							}
							std::cout << std::endl;
							std::cout<< "P2: ";
							for(int i = 0; i< 3; i++) {
								std::cout<< p2[i] << " ";
							}
							std::cout << std::endl;
							std::cout<< "Con: ";
							for(int i = 0; i< 3; i++) {
								std::cout<< con[i] << " ";
							}

							std::cout << std::endl;
						}
						double r_val = 0;
						r_val += con(0)* RHS_value(center,0);
						r_val += con(1)* RHS_value(center,1);
						if(counter  == 0){
							std::cout<< "Value: " << r_val;
							std::cout << std::endl;
						}
						r_val = r_val / con.norm_square();
						if(counter  == 0){
							std::cout<< "Value (normiert): " << r_val;
							std::cout << std::endl;
						}
						std::pair<unsigned int , double> real_the_boundary;
						real_the_boundary.first = current_dofs[2*GeometryInfo<3>::face_to_cell_lines(i, j)];
						real_the_boundary.second = r_val;
						std::map<unsigned int, double>::iterator it = boundary_values.find(real_the_boundary.first);
						if(it != boundary_values.end()) it->second = r_val;



						double i_val = 0;
						i_val += con(0)* RHS_value(center,3);
						i_val += con(1)* RHS_value(center,4);
						i_val = i_val / con.norm_square();
						i_val=1000;
						std::pair<unsigned int , double> imag_the_boundary;
						imag_the_boundary.first = current_dofs[2*GeometryInfo<3>::face_to_cell_lines(i, j)];
						imag_the_boundary.second = i_val;
						it = boundary_values.find(imag_the_boundary.first);
						if(it != boundary_values.end()) it->second = i_val;

					    counter ++ ;
					}
				}

			}
		}
	}

	std::cout << "Anzahl gesetzter Randwerte: "<< boundary_values.size() << std::endl;
	MatrixTools::apply_boundary_values (boundary_values, system_matrix, solution, system_rhs);
	//std::ofstream matrixfile;
	//matrixfile.open("SystemMatrix.dat");
	//system_matrix.print( matrixfile, true, false);
	//MatrixOut matrix_out;
	//std::ofstream out ("M.gnuplot");
	//matrix_out.build_patches (system_matrix, "M");
	//matrix_out.write_gnuplot (out);

}

void Waveguide::solve ()
{
	SolverControl           solver_control (1000, 1e1);
	SolverGMRES<Vector<double> > solver (solver_control);
	solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity());

}

void Waveguide::output_results () const
{

	DataOut<3> data_out;

	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (solution, "solution");

	data_out.build_patches ();

	std::ofstream output ("solution.gpl");
	std::ofstream outputvtk ("solution.vtk");
	data_out.write_vtk(outputvtk);
	data_out.write_gnuplot (output);
}

void Waveguide::run ()
{
	read_values ();
	make_grid ();
	setup_system ();
	assemble_system ();
	solve ();
	output_results ();
}

int main ()
{
	ParameterHandler prm;
	ParameterReader param(prm);
	param.read_parameters("parameters.prh");
	Waveguide laplace_problem(prm);
	laplace_problem.run ();
	return 0;
}


