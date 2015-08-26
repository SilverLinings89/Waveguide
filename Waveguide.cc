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

#include <deal.II/base/thread_management.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

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
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_relaxation.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_matrix_base.h>
#include <deal.II/lac/block_vector_base.h>

#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_vector_base.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_vector_base.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>


#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include <mpi/mpi.h>
#include "Loggers.cc"
#include <sstream>
#include "ParameterStruct.cc"
#include "Optimization.cc"
#include "ParameterReader.cc"
#include "WaveguideStructure.cc"

using namespace dealii;



std::string constraints_filename = "constraints.log";
std::string assemble_filename = "assemble.log";
std::string precondition_filename = "precondition.log";
std::string solver_filename = "solver.log";
std::string total_filename = "total.log";

double PI =  1.0;				// Overwritten by Param-Reader
double Eps0 = 1.0;				// Overwritten by Param-Reader
double Mu0 = 1.0; 				// Overwritten by Param-Reader
double c = 1/sqrt(Eps0 * Mu0); 	// Overwritten by Param-Reader
double f0 = c/0.63; 			// Overwritten by Param-Reader
double omega = 2 * PI * f0; 	// Overwritten by Param-Reader
int PRM_M_R_XLength = 2.0;		// Overwritten by Param-Reader
int PRM_M_R_YLength = 2.0;		// Overwritten by Param-Reader
int PRM_M_R_ZLength = 2.0;		// Overwritten by Param-Reader

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


/*
 * Distance in 2D. Since I have a cylindrical system, the distance to the middle-axis is an important criterium.
 * @param position this is the main point to calculate the distance for. Usually this will be used to calculate the distance to the 2D origin, in which the second argument is left out.
 * @param to if set, the result will not be positions distance to the origin but the xy-flat distance of vectors position and to.
 */
static double Distance2D (Point<3> position, Point<3> to = Point<3>()) {
		return sqrt((position(0)-to(0))*(position(0)-to(0)) + (position(1)-to(1))*(position(1)-to(1)));
}


static bool System_Coordinate_in_Waveguide(Point<3> p){
	double value = Distance2D(p);
	double reference = PRM_M_R_XLength/5.0;
	reference = reference * sqrt( 2 );
	return ( value < reference);
}


/**
 * This function calculates the distance of a given 3D-Point to the Waveguide-Wall in System-Coordinates.
 * Basically similar to System_Coordinate_in_Waveguide, which however only returns a boolean indicating, if the position is inside.
 * By definition, the return value is positive iff the given position is inside the Waveguide.
 * @param position Position for calculation.
 */
/**
static double System_Coordinate_Distance_To_Waveguide_Wall(Point<3> position) {
	double value = Distance2D(position);
	double reference = PRM_M_R_XLength/10.0;
	reference = sqrt(reference*reference *2 );
	return reference - value;
}
**/

/**
 * This function calculates the inverse of an imaginary number. It gives back the real OR imaginary component, depending on the arguments.
 * @param in_real - Real part of the input number.
 * @param in_imaginary - Imaginary part of the input number.
 * @param real - determines if real or imaginary part should be given back.
 */

template <int dim>
class RightHandSide : public Function<dim, double>
{
	public:
		RightHandSide () : Function<dim>(6) {}
		virtual double value (const Point<dim> &p, const unsigned int component ) const;
		virtual void vector_value (const Point<dim> &p,	Vector<double> &value) const;
};


template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p , const unsigned int component) const
{
	if(System_Coordinate_in_Waveguide(p)){
		if(p[2] < 0) {
			if(component == 0) {
				double d2 = Distance2D(p);
				//return 1.0;
				return exp(-d2*d2/2);
			}
		}
	}
	return 0.0;
}


template <int dim>
void RightHandSide<dim>::vector_value (const Point<dim> &p,	Vector<double> &values) const
{
	for (unsigned int c=0; c<6; ++c) values(c) = RightHandSide<dim>::value (p, c);
}

template <typename MatrixType, typename VectorType>
class Waveguide
{
	public:
		Waveguide (ParameterHandler &);
		void run ();
		void 	assemble_part (unsigned int in_part);

	private:
		void 	read_values ();
		void 	make_grid ();
		void 	setup_system ();
		void 	assemble_system ();
		void	estimate_solution();
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
		void	init_loggers();
		void 	timerupdate();
		void 	store();
		std::string		solutionpath;
		// Point<3> Triangulation_Stretch_X (const Point<3> &);
		// Point<3> Triangulation_Stretch_Y (const Point<3> &);
		// Point<3> Triangulation_Stretch_Z (const Point<3> &);

		Triangulation<3>	triangulation;
		FESystem<3>			fe;
		DoFHandler<3>		dof_handler;
		ConstraintMatrix 	cm;

		SparsityPattern		sparsity_pattern;
		MatrixType			system_matrix;
		//SparseMatrix<double> system_matrix_real;
		//SparseMatrix<double> system_matrix_imag;
		ParameterHandler 		&prm;
		ConstraintMatrix boundary_value_constraints_imaginary;
		ConstraintMatrix boundary_value_constraints_real;

		bool			PRM_O_Grid, PRM_O_Dofs, PRM_O_ActiveCells, PRM_O_VerboseOutput;
		std::string		PRM_M_C_TypeIn, PRM_M_C_TypeOut;
		double			PRM_M_C_RadiusIn, PRM_M_C_RadiusOut;
		//int				PRM_M_R_XLength, PRM_M_R_YLength, PRM_M_R_ZLength;
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
		int 			assembly_progress;


		VectorType	solution;
		VectorType	storage;
		bool		is_stored;
		VectorType	system_rhs;
		//Vector<double>	system_rhs_real;
		//Vector<double>	system_rhs_imag;
		FileLoggerData log_data;
		FileLogger log_constraints, log_assemble, log_precondition, log_solver, log_total;

};

template<typename MatrixType, typename VectorType >
Waveguide<MatrixType, VectorType>::Waveguide (ParameterHandler &param)
  :
  fe (FE_Nedelec<3> (2), 2),
  dof_handler (triangulation),
  prm(param),
  log_data(),
  log_constraints(std::string("constraints.log"), log_data),
  log_assemble(std::string("assemble.log"), log_data),
  log_precondition(std::string("precondition.log"), log_data),
  log_solver(std::string("solver.log"), log_data),
  log_total(std::string("total.log"), log_data)
{
	assembly_progress = 0;
	int i = 0;
	bool dir_exists = true;
	while(dir_exists) {
		std::stringstream out;
		out << "solutions/run";
		out << i;
		solutionpath = out.str();
		struct stat myStat;
		const char *myDir = solutionpath.c_str();
		if ((stat(myDir, &myStat) == 0) && (((myStat.st_mode) & S_IFMT) == S_IFDIR)) {
			i++;
		} else {
			dir_exists = false;
		}
	}

	mkdir(solutionpath.c_str(), ACCESSPERMS);
	std::cout << "Will write solutions to " << solutionpath << std::endl;
	is_stored = false;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::store() {
	storage.reinit(dof_handler.n_dofs());
	for(int i = 0; i < dof_handler.n_dofs(); i++){
		storage[i] = solution[i];
	}
	if(!is_stored) is_stored = true;
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::get_Tensor(Point<3> & position, bool inverse , bool epsilon) {

	Tensor<2,3, std::complex<double>> ret;
	for(int i = 0; i<3; i++ ){
		for(int j = 0; j<3; j++) {
			ret[i][j] = 0.0;
		}
	}
	std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);

	double omegaepsilon0 = (2* PI / PRM_M_W_Lambda) * c ;
	std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);
	if(PML_in_X(position)){
		double r,d, sigmax;
		r = PML_X_Distance(position);
		d = PRM_M_R_XLength * 1.0 * PRM_M_BC_Mantle/100.0;
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
		d = PRM_M_R_YLength * 1.0 * PRM_M_BC_Mantle/100.0;
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
		d = PRM_M_R_ZLength * 1.0 * ((position(2)<0)? PRM_M_BC_XYin : PRM_M_BC_XYout)/100.0;
		sigmaz = pow(d/r , PRM_M_BC_M) * PRM_M_BC_SigmaZMax;
		sz.real( 1 + pow(d/r , PRM_M_BC_M) * PRM_M_BC_KappaZMax);
		sz.imag( sigmaz / ((System_Coordinate_in_Waveguide(position))?PRM_M_W_EpsilonIn : PRM_M_W_EpsilonOut) * omegaepsilon0 );
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
			if(System_Coordinate_in_Waveguide(position)) {
				ret /= PRM_M_W_EpsilonIn;
			} else {
				ret /= PRM_M_W_EpsilonOut;
			}
			ret /= Eps0;
		} else {
			ret /= Mu0;
		}
	} else {
		if(epsilon) {
			if(System_Coordinate_in_Waveguide(position) ) {
				ret *= PRM_M_W_EpsilonIn;
			} else {
				ret *= PRM_M_W_EpsilonOut;
			}
			ret *= Eps0;
		} else {
			ret *= Mu0;
		}
	}

	return ret;
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::Transpose_Tensor(Tensor<2,3, std::complex<double>> input) {
	Tensor<2,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		for(int j = 0; j<3; j++){
			ret[i][j].real(input[i][j].real());
			ret[i][j].imag( - input[i][j].imag());
		}
	}
	return ret;
}

template<typename MatrixType, typename VectorType >
Tensor<1,3, std::complex<double>> Waveguide<MatrixType, VectorType>::Transpose_Vector(Tensor<1,3, std::complex<double>> input) {
	Tensor<1,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		ret[i].real(input[i].real());
		ret[i].imag( - input[i].imag());

	}
	return ret;
}

template<typename MatrixType, typename VectorType >
bool Waveguide<MatrixType, VectorType>::PML_in_X(Point<3> &p) {
	return  p(0) > ((PRM_M_R_XLength / 2.0) - (PRM_M_R_XLength * PRM_M_BC_Mantle/100.0)) ||  p(0) < (-(PRM_M_R_XLength / 2.0) + (PRM_M_R_XLength * PRM_M_BC_Mantle/100.0));
}

template<typename MatrixType, typename VectorType >
bool Waveguide<MatrixType, VectorType>::PML_in_Y(Point<3> &p) {
	return  p(1) > ((PRM_M_R_YLength / 2.0) - (PRM_M_R_YLength * PRM_M_BC_Mantle/100.0)) ||  p(1) < (-(PRM_M_R_YLength / 2.0) + (PRM_M_R_YLength * PRM_M_BC_Mantle/100.0));
}

template<typename MatrixType, typename VectorType >
bool Waveguide<MatrixType, VectorType>::PML_in_Z(Point<3> &p) {
	return  (p(2) > ((PRM_M_R_ZLength / 2.0) - (PRM_M_R_ZLength * PRM_M_BC_XYout/100.0)) ) ||  (p(2) < (-(PRM_M_R_ZLength / 2.0) + (PRM_M_R_ZLength * PRM_M_BC_XYin/100.0)  )  && !System_Coordinate_in_Waveguide(p));
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_X_Distance(Point<3> &p){
	if(p(0) >0){
		return p(0) - ((PRM_M_R_XLength / 2.0) - (PRM_M_R_XLength* PRM_M_BC_Mantle/100.0));
	} else {
		return -(p(0) + ((PRM_M_R_XLength / 2.0) - (PRM_M_R_XLength* PRM_M_BC_Mantle/100.0)));
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_Y_Distance(Point<3> &p){
	if(p(1) >0){
		return p(1) - ((PRM_M_R_YLength / 2.0) - (PRM_M_R_YLength* PRM_M_BC_Mantle/100.0));
	} else {
		return -(p(1) + ((PRM_M_R_YLength / 2.0) - (PRM_M_R_YLength* PRM_M_BC_Mantle/100.0)));
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_Z_Distance(Point<3> &p){
	if(p(2) >0){
		return p(2) - ((PRM_M_R_ZLength / 2.0) - (PRM_M_R_ZLength * PRM_M_BC_XYout/100));
	} else {
		return -(p(2) + ((PRM_M_R_ZLength / 2.0) - (PRM_M_R_ZLength * PRM_M_BC_XYin/100)));
	}
}

/**
template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::RHS_value (const Point<3> &p , const unsigned int component)
{
	if(p[2] < 0){
		if( System_Coordinate_in_Waveguide(p) ){
			//Point<3> ref(0,0,0);
			double d2 = Distance2D(p);
			if(component <= 2 ) {
				return 1.0;
				// return exp(-d2*d2/2);
			}
		}

		else return 0.0;
	}

	return 0.0;
}
**/

Point<3> Triangulation_Stretch_X (const Point<3> &p)
{
  Point<3> q = p;
  q[0] *= PRM_M_R_XLength /2.0;
  return q;
}


Point<3> Triangulation_Stretch_Y (const Point<3> &p)
{
  Point<3> q = p;
  q[1] *= PRM_M_R_YLength /2.0;
  return q;
}


Point<3> Triangulation_Stretch_Z (const Point<3> &p)
{
  Point<3> q = p;
  q[2] *= PRM_M_R_ZLength /2.0;
  return q;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::read_values() {
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

	prm.enter_subsection("Assembly");
	{
		PRM_A_Threads = prm.get_integer("Threads");
	}
	prm.leave_subsection();

	prm.enter_subsection("Solver");
	{
		PRM_S_Solver = prm.get("Solver");
		PRM_S_GMRESSteps = prm.get_integer("GMRESSteps");
		PRM_S_Preconditioner = prm.get("Preconditioner");
		PRM_S_PreconditionerBlockCount = prm.get_integer("PreconditionerBlockCount");
		PRM_S_Steps = prm.get_integer("Steps");
		PRM_S_Precision = prm.get_double("Precision");
	}
	prm.leave_subsection();

	prm.enter_subsection("Constants");
	{
		PI = prm.get_double("Pi");
		if(! prm.get_bool("AllOne")){
			Eps0 = prm.get_double("EpsilonZero");
			Mu0 = prm.get_double("MuZero");
			c = 1/sqrt(Eps0 * Mu0);
			f0 = c/0.63;
			omega = 2 * PI * f0;
		} else {
			Eps0 = 1.0;
			Mu0 = 1.0;
			c = 1.0;
			f0 = 1.0;
			omega = 1.0;
		}
	}
	prm.leave_subsection();
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::make_grid ()
{
	log_total.start();
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

	GridTools::transform(& Triangulation_Stretch_X, triangulation);
	GridTools::transform(& Triangulation_Stretch_Y, triangulation);
	GridTools::transform(& Triangulation_Stretch_Z, triangulation);


	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		if(cell->at_boundary()){
			for(int j = 0; j<6; j++){
				if(cell->face(j)->at_boundary()){
					Point<3> ctr =cell->face(j)->center(true, false);
					if(System_Coordinate_in_Waveguide(ctr)){
						if(ctr(2) < 0) cell->face(j)->set_all_boundary_indicators(1);
						else cell->face(j)->set_all_boundary_indicators(2);
					}
				}
			}
		}
	}


	if(PRM_D_Refinement == "global") triangulation.refine_global (PRM_D_XY);

	cell = triangulation.begin_active();
	double l = (double)PRM_M_R_ZLength / (PRM_A_Threads*2.0);
	for (; cell!=endc; ++cell){

		int temp  = (int) (((cell->center(true, false))[2] + (PRM_M_R_ZLength/2)) / l);
		if( temp >= 2* PRM_A_Threads) std::cout << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
		cell->set_subdomain_id(temp);
	}


	if(PRM_O_Grid) {
		if(PRM_O_VerboseOutput) std::cout<< "Writing Mesh data to file \"grid-3D.vtk\"" << std::endl;
		mesh_info(triangulation, "grid-3D.vtk");
		if(PRM_O_VerboseOutput) std::cout<< "Done" << std::endl;

	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::setup_system ()
{
	if(PRM_O_VerboseOutput && PRM_O_Dofs) {
		std::cout << "Distributing Degrees of freedom." << std::endl;
	}
	dof_handler.distribute_dofs (fe);
	if(PRM_O_VerboseOutput) {
		std::cout << "Renumbering DOFs (Cuthill_McKee...)" << std::endl;
	}

	DoFRenumbering::Cuthill_McKee (dof_handler);
	if(PRM_O_Dofs) {
		std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
	}

	if(PRM_O_VerboseOutput) {
		std::cout << "Calculating compressed Sparsity Pattern..." << std::endl;
	}

	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
	sparsity_pattern.copy_from(c_sparsity);

	system_matrix.reinit( sparsity_pattern );
	solution.reinit ( dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

	if(PRM_O_VerboseOutput) {
			std::cout << "Done." << std::endl;
	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::assemble_part ( unsigned int in_part) {
	QGauss<3>  quadrature_formula(2);
	const FEValuesExtractors::Vector real(0);
	const FEValuesExtractors::Vector imag(3);
	FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	FullMatrix<double>	cell_matrix_real (dofs_per_cell, dofs_per_cell);
	Vector<double>		cell_rhs (dofs_per_cell);
	cell_rhs = 0;
	Tensor<2,3, std::complex<double>> 		epsilon, mu;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	DoFHandler<3>::active_cell_iterator cell, endc;

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();

	for (; cell!=endc; ++cell)
	{
		if(cell->subdomain_id() == in_part) {
			fe_values.reinit (cell);
			quadrature_points = fe_values.get_quadrature_points();
			cell_matrix_real = 0;

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

						std::complex<double> x = (mu * I_Curl) * Transpose_Vector(J_Curl) * JxW - ( ( epsilon * I_Val ) * Transpose_Vector(J_Val)) *(omega * omega)*JxW ;
						if(x.real() != 0) {
							cell_matrix_real[i][j] += x.real();
						}
					}
				}
			}

			cell->get_dof_indices (local_dof_indices);

			cm.distribute_local_to_global(cell_matrix_real, cell_rhs, local_dof_indices,system_matrix, system_rhs, false );
			//cm.distribute_local_to_global(cell_matrix_imag, cell_rhs, local_dof_indices,system_matrix.block(0,1), system_rhs.block(1), true );

	    }
	}
	assembly_progress ++;
	std::cout << "Progress: " << 100 * assembly_progress/(PRM_A_Threads*2) << " %" << std::endl;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::assemble_system ()
{
	QGauss<3>  quadrature_formula(2);
	const FEValuesExtractors::Vector real(0);
	const FEValuesExtractors::Vector imag(3);
	FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	if(PRM_O_VerboseOutput) {
		std::cout << "Dofs per cell: " << dofs_per_cell << std::endl << "Quadrature Formula Size: " << n_q_points << std::endl;
		std::cout << "Dofs per face: " << fe.dofs_per_face << std::endl << "Dofs per line: " << fe.dofs_per_line << std::endl;
	}

	log_data.Dofs = dof_handler.n_dofs();
	log_constraints.start();

	//starting to calculate Constraint Matrix for boundary values;
	//VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, RightHandSide<3>() , 0 , cm , StaticMappingQ1<3>::mapping);
	VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, RightHandSide<3>() , 1 , cm , StaticMappingQ1<3>::mapping);
	//VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, RightHandSide<3>() , 2 , cm , StaticMappingQ1<3>::mapping);

	DoFHandler<3>::active_cell_iterator cell, endc;

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();

	for (; cell!=endc; ++cell)
	{
		bool At_Boundary = false;
		bool Is_At_One = false;
		for(int i = 0; i < 6; i++) {
			if(cell->face(i)->at_boundary()){
				At_Boundary = true;
				if(cell->face(i)->boundary_indicator() == 1) Is_At_One = true;
			}
		}
		if(At_Boundary && !Is_At_One) {
			std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
			cell->get_dof_indices(local_dof_indices);
			unsigned int i;
			for (i = 0; i < dofs_per_cell; ++i) {
				cm.add_line(local_dof_indices[i]);
				cm.set_inhomogeneity(local_dof_indices[i], 0);
			}
		}
	}

	cm.close();
	//cm.distribute(solution);
	log_constraints.stop();

	log_assemble.start();
	std::cout << "Starting Assemblation process" << std::endl;
	Threads::TaskGroup<void> task_group1;
	for (int i = 0; i < PRM_A_Threads; ++i) {
		task_group1 += Threads::new_task (&Waveguide<MatrixType, VectorType>::assemble_part , *this, 2*i);
	}
	task_group1.join_all ();

	Threads::TaskGroup<void> task_group2;
	for (int i = 0; i < PRM_A_Threads; ++i) {
		task_group2 += Threads::new_task (&Waveguide<MatrixType, VectorType>::assemble_part , *this, 2*i+1);
	}
	task_group2.join_all ();

	std::cout<<system_rhs.l2_norm()<<std::endl;
	log_assemble.stop();

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::estimate_solution() {
	DoFHandler<3>::active_cell_iterator cell, endc;

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if(! (cell->at_boundary())) {
			std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
			for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
				for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
					((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
					Point<3, double> p = ((cell->face(i))->line(j))->center(false, false);
					Point<3, double> direction = ((cell->face(i))->line(j))->vertex(0) - ((cell->face(i))->line(j))->vertex(1);
					const std::complex<double> z(0.0, omega * (p(2)- PRM_M_R_ZLength/2.0));
					double d2 = Distance2D(p);
					const std::complex<double> result = - exp(z) * exp(-d2*d2/2);
					solution[local_dof_indices[0]] = result.real();
					solution[local_dof_indices[1]] = result.imag();
				}
			}
		}
	}
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::timerupdate() {
	log_precondition.stop();
	log_solver.start();
}

template<>
void Waveguide<PETScWrappers::SparseMatrix, PETScWrappers::Vector>::solve () {
	SolverControl          solver_control (PRM_S_Steps, PRM_S_Precision, true );
	log_precondition.start();
	std::cout << "Framework: PETSc" << std::endl;
	std::cout << "Solver: " << PRM_S_Solver << std::endl;
	std::cout << "Preconditioner: " << PRM_S_Preconditioner << std::endl;
	system_matrix.compress(VectorOperation::add);
	std::cout << "Matrix-compression done" << std::endl;
	system_rhs.compress(VectorOperation::add);
	std::cout << "Vector-compression done" << std::endl;
	if(PRM_S_Solver == "GMRES") {
		PETScWrappers::SolverGMRES::AdditionalData add(PRM_S_GMRESSteps);
		PETScWrappers::SolverGMRES solver ( solver_control, system_matrix.get_mpi_communicator() , add);

		if(PRM_S_Preconditioner == "Jacobi"){
			PETScWrappers::PreconditionJacobi pre_jacobi ;

			pre_jacobi.initialize(system_matrix, PETScWrappers::PreconditionJacobi::AdditionalData());
			std::cout << "Preconditioner initialized! Start solving ..." << std::endl;
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, pre_jacobi);
		}

		if(PRM_S_Preconditioner == "SOR"){
			PETScWrappers::PreconditionSOR plu;
			plu.initialize(system_matrix, PETScWrappers::PreconditionSOR::AdditionalData());
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, plu);
		}

		if(PRM_S_Preconditioner == "SSOR"){
			PETScWrappers::PreconditionSSOR plu;
			plu.initialize(system_matrix, PETScWrappers::PreconditionSSOR::AdditionalData());
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, plu);
		}

		if(PRM_S_Preconditioner == "ILU"){
			PETScWrappers::PreconditionILU ilu;
			ilu.initialize(system_matrix, PETScWrappers::PreconditionILU::AdditionalData());
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, ilu);
		}

		if(PRM_S_Preconditioner == "ICC") {
			PETScWrappers::PreconditionICC icc;
			icc.initialize(system_matrix, PETScWrappers::PreconditionICC::AdditionalData());
			timerupdate();
			solver.solve(system_matrix, solution, system_rhs, icc);
		}

		if(PRM_S_Preconditioner == "LU") {
			PETScWrappers::PreconditionLU lu;
			lu.initialize(system_matrix, PETScWrappers::PreconditionLU::AdditionalData());
			timerupdate();
			solver.solve(system_matrix, solution, system_rhs, lu);
		}

		if(PRM_S_Preconditioner == "BoomerAMG") {
			PETScWrappers::PreconditionBoomerAMG bamg;
			bamg.initialize(system_matrix, PETScWrappers::PreconditionBoomerAMG::AdditionalData());
			timerupdate();
			solver.solve(system_matrix, solution, system_rhs, bamg);
		}

		if(PRM_S_Preconditioner == "ParaSails") {
			PETScWrappers::PreconditionParaSails psails;
			psails.initialize(system_matrix, PETScWrappers::PreconditionParaSails::AdditionalData(0));
			timerupdate();
			solver.solve(system_matrix, solution, system_rhs, psails);
		}

		if(PRM_S_Preconditioner == "Eisenstat") {
			PETScWrappers::PreconditionEisenstat estat;
			estat.initialize(system_matrix, PETScWrappers::PreconditionEisenstat::AdditionalData());
			timerupdate();
			solver.solve(system_matrix, solution, system_rhs, estat);
		}
	}

	log_solver.stop();
	cm.distribute(solution);
}

template<>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::Vector>::solve () {
	SolverControl          solver_control (PRM_S_Steps,  PRM_S_Precision, true);
	log_precondition.start();
	system_matrix.compress(VectorOperation::add);
	std::cout << "Framework: Trilinos" << std::endl;
	std::cout << "Solver: " << PRM_S_Solver << std::endl;
	std::cout << "Preconditioner: " << PRM_S_Preconditioner << std::endl;

	if(PRM_S_Solver == "GMRES") {
		TrilinosWrappers::SolverGMRES::AdditionalData additional (true, PRM_S_Steps);
		TrilinosWrappers::SolverGMRES solver( solver_control, additional);
		if(PRM_S_Preconditioner == "Jacobi"){
			TrilinosWrappers::PreconditionJacobi pre_jacobi;
			pre_jacobi.initialize(system_matrix, TrilinosWrappers::PreconditionJacobi::AdditionalData());
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, pre_jacobi);
		}
		if(PRM_S_Preconditioner == "SOR"){
			TrilinosWrappers::PreconditionSOR plu;
			plu.initialize(system_matrix);
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, plu);
		}
		if(PRM_S_Preconditioner == "SSOR"){
			TrilinosWrappers::PreconditionSSOR plu;
			plu.initialize(system_matrix);
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, plu);
		}
		if(PRM_S_Preconditioner == "ILU"){
			TrilinosWrappers::PreconditionILU ilu;
			ilu.initialize(system_matrix, TrilinosWrappers::PreconditionILU::AdditionalData());
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, ilu);
		}
		if(PRM_S_Preconditioner == "Identity"){
			TrilinosWrappers::PreconditionIdentity ide;
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, ide);
		}
	}

	log_solver.stop();
	cm.distribute(solution);
}

template<>
void Waveguide<SparseMatrix<double>, Vector<double> >::solve () {
	SolverControl          solver_control (PRM_S_Steps, PRM_S_Precision, true, true);
	log_precondition.start();
	std::cout << "Framework: Dealii" << std::endl;
	std::cout << "Solver: " << PRM_S_Solver << std::endl;
	std::cout << "Preconditioner: " << PRM_S_Preconditioner << std::endl;
	if(PRM_S_Solver == "Richardson") {
		SolverRichardson<Vector<double> > solver (solver_control);
		timerupdate();
		solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity());
	}

	if(PRM_S_Solver == "GMRES") {
		SolverGMRES<Vector<double> > solver (solver_control, SolverGMRES<Vector<double> >::AdditionalData(PRM_S_GMRESSteps, true));

		if(PRM_S_Preconditioner == "Block_Jacobi"){
			PreconditionBlockJacobi<SparseMatrix<double>, double> block_jacobi;
			block_jacobi.initialize(system_matrix, PreconditionBlock<SparseMatrix<double>, double>::AdditionalData(log_data.Dofs / PRM_S_PreconditionerBlockCount));
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, block_jacobi);
		}

		if(PRM_S_Preconditioner == "Identity") {
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, PreconditionIdentity());
		}

		if(PRM_S_Preconditioner == "Jacobi"){
			PreconditionJacobi<SparseMatrix<double>> pre_jacobi;
			pre_jacobi.initialize(system_matrix, PreconditionJacobi<SparseMatrix<double>>::AdditionalData());
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, pre_jacobi);
		}

		if(PRM_S_Preconditioner == "SOR"){
			PreconditionSOR<SparseMatrix<double> > plu;
			plu.initialize(system_matrix, .6);
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, plu);
		}

		if(PRM_S_Preconditioner == "SSOR"){
			PreconditionSSOR<SparseMatrix<double> > plu;
			plu.initialize(system_matrix, .6);
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, plu);
		}

		if(PRM_S_Preconditioner == "ILU"){
			SparseILU<double> ilu;
			ilu.initialize(system_matrix, SparseILU<double>::AdditionalData());
			timerupdate();
			solver.solve (system_matrix, solution, system_rhs, ilu);
		}
	}

	if(PRM_S_Solver == "UMFPACK") {
		SparseDirectUMFPACK  A_direct;
		A_direct.initialize(system_matrix);
		timerupdate();
		A_direct.vmult(solution, system_rhs);
	}

	log_solver.stop();
	cm.distribute(solution);
}

template <typename MatrixType, typename VectorType>
void Waveguide<MatrixType, VectorType>::solve ()
{

	if(typeid(VectorType) == typeid(Vector<double>)) {

	}

	if(typeid(MatrixType) == typeid(TrilinosWrappers::SparseMatrix)) {

	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::init_loggers () {
	log_data.PML_in 				= 	PRM_M_BC_XYin;
	log_data.PML_out 				=	PRM_M_BC_XYout;
	log_data.PML_mantle 			=	PRM_M_BC_Mantle;
	log_data.ParamSteps 			=	PRM_S_Steps;
	log_data.Precondition_BlockSize = 	0;
	log_data.Precondition_weight 	=	0;
	log_data.Solver_Precision 		=	PRM_S_Precision;
	log_data.XLength				=	PRM_M_R_XLength;
	log_data.YLength 				= 	PRM_M_R_YLength;
	log_data.ZLength 				= 	PRM_M_R_ZLength;
	log_data.preconditioner 		=	PRM_S_Preconditioner;
	log_data.solver 				= 	PRM_S_Solver;
	log_data.Dofs 					=	0;
	log_constraints.Dofs			=	true;
	log_constraints.PML_in			=	log_constraints.PML_mantle	= log_constraints.PML_out		= true;
	log_assemble.Dofs				=	true;
	log_precondition.Dofs			=	true;
	log_precondition.preconditioner = 	log_precondition.cputime	= true;
	log_solver.Dofs					=	true;
	log_solver.solver				=	log_solver.preconditioner	= log_solver.Solver_Precision	= log_solver.cputime	= true;
	log_total.Dofs					=	log_total.solver			= log_total.Solver_Precision	= log_total.cputime		= true;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::output_results () const
{


	DataOut<3> data_out;

	data_out.attach_dof_handler (dof_handler);
	data_out.add_data_vector (solution, "solution");

	data_out.build_patches ();

	//std::ofstream output ("solution.gpl");
	std::ofstream outputvtk (solutionpath + "/solution.vtk");
	data_out.write_vtk(outputvtk);
	//data_out.write_gnuplot (output);

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::run ()
{
	init_loggers ();
	read_values ();
	make_grid ();
	setup_system ();
	assemble_system ();
	estimate_solution();
	solve ();
	// output_results ();
	log_total.stop();
}

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	Parameters System_Parameters = GetParameters();
	// Waveguide<PETScWrappers::SparseMatrix, PETScWrappers::Vector > waveguide(prm);
	// Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::Vector > waveguide(prm);
	double r_0, r_1, deltaY, epsilon_M, epsilon_K, sectors;
	WaveguideStructure structure(System_Parameters);
	Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> > waveguide(System_Parameters, system);
	Optimization opt(System_Parameters, waveguide, structure);
	opt.run();
	return 0;
}

