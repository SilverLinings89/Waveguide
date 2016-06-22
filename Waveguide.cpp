#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "Waveguide.h"
#include "staticfunctions.cpp"
#include "WaveguideStructure.h"
#include "SolutionWeight.h"
#include "ExactSolution.h"
#include <string>
#include <sstream>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/std_cxx11/bind.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/shared_tria.h>
#include "QuadratureFormulaCircle.cpp"
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include "PreconditionerSweeping.cpp"
#include <deal.II/lac/solver.h>
using namespace dealii;

template<typename MatrixType, typename VectorType >
Waveguide<MatrixType, VectorType>::Waveguide (Parameters &param )
  :
  fe(FE_Nedelec<3> (0), 2),
  triangulation (MPI_COMM_WORLD, typename parallel::distributed::Triangulation<3>::MeshSmoothing(Triangulation<3>::none ), parallel::distributed::Triangulation<3>::Settings::no_automatic_repartitioning),
  //triangulation_real (MPI_COMM_WORLD, typename Triangulation<3>::MeshSmoothing(Triangulation<3>::smoothing_on_refinement | Triangulation<3>::smoothing_on_coarsening)),
  dof_handler (triangulation),
  //dof_handler_real(triangulation_real),
  prm(param),
  log_data(),
  log_constraints(std::string("constraints.log"), log_data),
  log_assemble(std::string("assemble.log"), log_data),
  log_precondition(std::string("precondition.log"), log_data),
  log_total(std::string("total.log"), log_data),
  log_solver(std::string("solver.log"), log_data),
  run_number(0),
  condition_file_counter(0),
  eigenvalue_file_counter(0),
  Sectors(prm.PRM_M_W_Sectors),
  Dofs_Below_Subdomain(prm.PRM_M_W_Sectors),
  Block_Sizes(prm.PRM_M_W_Sectors),
  temporary_pattern_preped(false),
  real(0),
  imag(3),
  solver_control (prm.PRM_S_Steps, prm.PRM_S_Precision, false, true),
  pout(std::cout, GlobalParams.MPI_Rank==0)
   // Sweeping_Additional_Data(1.0, 1),
  // sweep(Sweeping_Additional_Data)
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
	i = Utilities::MPI::min(i, MPI_COMM_WORLD);
	std::stringstream out;
	out << "solutions/run";
	out << i;
	solutionpath = out.str();
	Dofs_Below_Subdomain[prm.PRM_M_W_Sectors];
	mkdir(solutionpath.c_str(), ACCESSPERMS);
	pout << "Will write solutions to " << solutionpath << std::endl;
	is_stored = false;
	solver_control.log_frequency(10);
	const int number = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) -1;
	Preconditioner_Matrices = new TrilinosWrappers::SparseMatrix[number];
}

template<typename MatrixType, typename VectorType>
std::complex<double> Waveguide<MatrixType, VectorType>::evaluate_for_Position(double x, double y, double z ) {
	Point<3, double> position(x, y, z);
	Vector<double> result(6);
	Vector<double> mode(3);

	mode(0) = TEMode00( position , 0);
	mode(1) = TEMode00( position , 1);
	mode(2) = 0;

	VectorTools::point_value(dof_handler, solution, position, result);

	return std::complex<double>( mode(0)*result(0) + mode(1)*result(1) + mode(2)*result(2) , mode(0)*result(3) + mode(1)*result(4) + mode(2)*result(5) );
}

template<typename MatrixType, typename VectorType>
std::complex<double> Waveguide<MatrixType, VectorType>::gauss_product_2D_sphere(double z, int n, double R, double Xc, double Yc)
{
	double* r = NULL;
	double* t = NULL;
	double* q = NULL;
	double* A = NULL;
	double  B;
	double x, y;
	std::complex<double> s(0.0, 0.0);

	int i,j;

	/* Load appropriate predefined table */
	for (i = 0; i<GSPHERESIZE;i++)
	{
		if(n==gsphere[i].n)
		{
			r = gsphere[i].r;
			t = gsphere[i].t;
			q = gsphere[i].q;
			A = gsphere[i].A;
			B = gsphere[i].B;
			break;
		}
	}

	if (NULL==r) return -1.0;

	for (i=0;i<n;i++)
	{
		for (j=0;j<n;j++)
		{
			x = r[j]*q[i];
			y = r[j]*t[i];
			s += A[j]*evaluate_for_Position(R*x-Xc,R*y-Yc,z);
		}
	}

	s *= R*R*B;

	return s;
}

template<typename MatrixType, typename VectorType>
double Waveguide<MatrixType, VectorType>::evaluate_for_z(double z) {
	double r = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0;
	std::complex<double> exc = gauss_product_2D_sphere(z,10,r,0,0);
	return std::sqrt(exc.real()*exc.real() + exc.imag()*exc.imag());
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_out () {
	double real = 0.0;
	double imag = 0.0;
	double z = prm.PRM_M_R_ZLength/2.0;
	double r = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0;

	for (double x = -r; x < r; x += r/5) {
		for(double y = -r; y < r; y += r/5) {
			Point<3, double> position(x, y, z);
			Vector<double> result(6);
			VectorTools::point_value(dof_handler, solution, position, result);
			double Q1 = structure->getQ1(position[2] );
			double Q2 = structure->getQ2(position[2] );
			real +=  (result[0] / Q1) * ( TEMode00( position , 0) / Q1);
			imag += (result[1] / Q2) * ( TEMode00( position , 1) / Q2);
			real += (result[3] / Q1) * ( TEMode00( position , 0) / Q1);
			imag += (result[4] / Q2) * ( TEMode00( position , 1) / Q2);
		}
	}
	return sqrt(imag*imag + real*real);
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_in () {
	double real = 0.0;
	double imag = 0.0;
	double z = -prm.PRM_M_R_ZLength/2.0 ;
	double r = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0;

	for (double x = -r; x < r; x += r/5) {
		for(double y = -r; y < r; y += r/5) {
			Point<3, double> position(x, y, z);
			Vector<double> result(6);
			VectorTools::point_value(dof_handler, solution, position, result);
			double Q1 = structure->getQ1(position[2] );
			double Q2 = structure->getQ2(position[2] );
			real +=  (result[0] / Q1) * ( TEMode00( position , 0) / Q1);
			imag += (result[1] / Q2) * ( TEMode00( position , 1) / Q2);
			real += (result[3] / Q1) * ( TEMode00( position , 0) / Q1);
			imag += (result[4] / Q2) * ( TEMode00( position , 1) / Q2);

		}
	}
	return sqrt(imag*imag + real*real);
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_overall () {
	std::vector <double> qualities(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
	double lower = 0.0;
	double upper = 0.0;
	if(GlobalParams.evaluate_in) {
		std::cout << "Evaluation for input side by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << -GlobalParams.PRM_M_R_ZLength / 2.0 << std::endl;
		lower = evaluate_for_z(-GlobalParams.PRM_M_R_ZLength / 2.0 + 0.0001);
	}
	if(GlobalParams.evaluate_out) {
		std::cout << "Evaluation for output side by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << GlobalParams.PRM_M_R_ZLength / 2.0 << std::endl;
		upper = evaluate_for_z(GlobalParams.PRM_M_R_ZLength / 2.0 - 0.0001);
	}
	lower = Utilities::MPI::sum(lower, MPI_COMM_WORLD);
	upper = Utilities::MPI::sum(upper, MPI_COMM_WORLD);

	for(unsigned int i = 0; i< Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); i++) {
		double contrib = 0.0;
		if(i == GlobalParams.MPI_Rank) {
			std::cout << "Evaluation of contribution by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << GlobalParams.z_evaluate<< std::endl;
			 contrib = evaluate_for_z(GlobalParams.z_evaluate);
		}
		qualities[i] = Utilities::MPI::sum(contrib, MPI_COMM_WORLD);
	}
	double quality_in	= lower;
	double quality_out	= upper;
	pout << "Quality in: "<< quality_in << std::endl;
	pout << "Quality out: "<< quality_out << std::endl;
	/**
	differences.reinit(triangulation.n_active_cells());
	QGauss<3>  quadrature_formula(4);
	VectorTools::integrate_difference(dof_handler, solution, ExactSolution<3>(), differences, quadrature_formula, VectorTools::L2_norm, new SolutionWeight<3>(), 2.0);
	double L2error = differences.l2_norm();
	**/
	// if(!is_stored) pout << "L2 Norm of the error: " << L2error << std::endl;
	if(GlobalParams.MPI_Rank == 0) {
		result_file << "Number of Dofs: " << dof_handler.n_dofs() << std::endl;
		//result_file << "L2 Norm of the error:" << L2error << std::endl;
		result_file << "Z-Length: " << GlobalParams.PRM_M_R_ZLength << std::endl;
		result_file << "Quality in: "<< quality_in << std::endl;
		result_file << "Quality out: "<< quality_out << std::endl;
		result_file << 	100* quality_out/quality_in << "%" << std::endl;
		result_file << "Delta:" << GlobalParams.PRM_M_W_Delta << std::endl;
		result_file << "Solver: " << GlobalParams.PRM_S_Solver << std::endl;
		result_file << "Preconditioner: " << GlobalParams.PRM_S_Preconditioner << std::endl;
		result_file << "Minimal Stretch: " << structure->lowest << std::endl;
		result_file << "Maximal Stretch: " << structure->highest << std::endl;
		result_file.close();

	}
	pout << "Signal quality evolution: ";
	for(unsigned int i =0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); i++) {
			pout << 100 * qualities[i]/ lower << "% ";
			if(i != Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)-1) pout << " -> ";
	}
	pout << std::endl;
	return quality_out/quality_in;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType >::store() {
	reinit_storage();
	// storage.reinit(dof_handler.n_dofs());
	/** storage = solution;
	is_stored = true; **/
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::get_Tensor(Point<3> & position, bool inverse , bool epsilon) {
	if(PML_in_Z(position)  || PML_in_X(position) || PML_in_Y(position)) {
		Tensor<2,3, std::complex<double>> ret;
		for(int i = 0; i<3; i++ ){
			for(int j = 0; j<3; j++) {
				ret[i][j] = 0.0;
			}
		}
		std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);

		double omegaepsilon0 = GlobalParams.PRM_C_omega * ((System_Coordinate_in_Waveguide(position))?GlobalParams.PRM_M_W_EpsilonIn : GlobalParams.PRM_M_W_EpsilonOut);
		std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);
		if(PML_in_X(position)){
			double r,d, sigmax;
			r = PML_X_Distance(position);
			d = GlobalParams.PRM_M_R_XLength * 1.0 * GlobalParams.PRM_M_BC_Mantle;
			sigmax = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaXMax;
			sx.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaXMax);
			sx.imag( sigmax / ( omegaepsilon0));
			S1 /= sx;
			S2 *= sx;
			S3 *= sx;
		}
		if(PML_in_Y(position)){
			double r,d, sigmay;
			r = PML_Y_Distance(position);
			d = GlobalParams.PRM_M_R_YLength * 1.0 * GlobalParams.PRM_M_BC_Mantle;
			sigmay = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaYMax;
			sy.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaYMax);
			sy.imag( sigmay / ( omegaepsilon0));
			S1 *= sy;
			S2 /= sy;
			S3 *= sy;
		}
		if(PML_in_Z(position)){
			double r,d, sigmaz;
			r = PML_Z_Distance(position);
			d = GlobalParams.PRM_M_BC_XYout * structure->Sector_Length();
			sigmaz = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaZMax;
			sz.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaZMax);
			sz.imag( sigmaz / omegaepsilon0 );
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

		if(epsilon) {
			if(System_Coordinate_in_Waveguide(position) ) {
				ret *= GlobalParams.PRM_M_W_EpsilonIn;
			} else {
				ret *= GlobalParams.PRM_M_W_EpsilonOut;
			}
			ret *= GlobalParams.PRM_C_Eps0;
		} else {
			ret *= GlobalParams.PRM_C_Mu0;
		}

		if(inverse) ret = invert(ret);

		return ret;

	} else {

		Tensor<2,3, std::complex<double>> ret2;
		Tensor<2,3, double> transformation = structure->TransformationTensor(position[0], position[1], position[2]);
		double dist = position[0] * position[0] + position[1]*position[1];
		dist = std::sqrt(dist);
		double maxdist = GlobalParams.PRM_M_R_XLength/2.0 - GlobalParams.PRM_M_BC_Mantle * GlobalParams.PRM_M_R_XLength;
		double mindist = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0;
		double sig = sigma(dist, mindist, maxdist);
		double factor = InterpolationPolynomialZeroDerivative(sig, 1,0);
		transformation *= factor;
		for(int i = 0; i < 3; i++) {
			transformation[i][i] += 1-factor;
		}
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				ret2[i][j] = transformation[i][j]* std::complex<double>(1.0, 0.0);
			}
		}

		if(inverse) ret2 = invert(ret2);

		return ret2;
	}
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::get_Preconditioner_Tensor(Point<3> & position, bool inverse , bool epsilon, int block) {
	std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);
	Tensor<2,3, std::complex<double>> ret;

	double omegaepsilon0 = GlobalParams.PRM_C_omega * ((System_Coordinate_in_Waveguide(position))?GlobalParams.PRM_M_W_EpsilonIn : GlobalParams.PRM_M_W_EpsilonOut);
	std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);
	if(PML_in_X(position)){
		double r,d, sigmax;
		r = PML_X_Distance(position);
		d = GlobalParams.PRM_M_R_XLength * 1.0 * GlobalParams.PRM_M_BC_Mantle;
		sigmax = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaXMax;
		sx.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaXMax);
		sx.imag( sigmax / ( omegaepsilon0));
		S1 /= sx;
		S2 *= sx;
		S3 *= sx;
	}
	if(PML_in_Y(position)){
		double r,d, sigmay;
		r = PML_Y_Distance(position);
		d = GlobalParams.PRM_M_R_YLength * 1.0 * GlobalParams.PRM_M_BC_Mantle;
		sigmay = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaYMax;
		sy.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaYMax);
		sy.imag( sigmay / ( omegaepsilon0));
		S1 *= sy;
		S2 /= sy;
		S3 *= sy;
	}
	if(Preconditioner_PML_in_Z(position, block)){
		double r,d, sigmaz;
		r = Preconditioner_PML_Z_Distance(position, block);
		d = structure->Sector_Length() * 0.1;
		sigmaz = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaZMax;
		sz.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaZMax);
		sz.imag( sigmaz / omegaepsilon0 );
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


	if(epsilon) {
		if(System_Coordinate_in_Waveguide(position) ) {
			ret *= GlobalParams.PRM_M_W_EpsilonIn;
		} else {
			ret *= GlobalParams.PRM_M_W_EpsilonOut;
		}
		ret *= GlobalParams.PRM_C_Eps0;
	}

	Tensor<2,3, double> transformation = structure->TransformationTensor(position[0], position[1], position[2]);
	Tensor<2,3, std::complex<double>> ret2;

	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			ret2[i][j] = std::complex<double>(0.0, 0.0);
			for(int k = 0; k < 3; k++) {
				ret2[i][j] += ret[i][k] * transformation[k][j];
			}
		}
	}
	//pout << "get_Tensor_2" << std::endl;
	if  ( inverse ) ret2 = invert(ret2);

	return ret2;

	/**
	Tensor<2,3, double> transformation = structure->TransformationTensor(position[0], position[1], position[2]);

	Tensor<2,3, std::complex<double>>( 1.0, ret;
	for(int l = 0; l<3; l++) {
		for(int k = 0; k<3; k++) {
			std::complex<double> temp(transformation[l][k],0.0);
			ret[l][k] = temp;
		}
	}

	if(epsilon) {
		if(System_Coordinate_in_Waveguide(position) ) {
			ret *= GlobalParams.PRM_M_W_EpsilonIn;
		} else {
			ret *= GlobalParams.PRM_M_W_EpsilonOut;
		}
		ret *= GlobalParams.PRM_C_Eps0;
	} else {
		ret *= GlobalParams.PRM_C_Mu0;
	}

	ret[0][0] *= sy * sz / sx ;
	ret[0][1] *= sz ;
	ret[0][2] *= sy ;

	ret[1][0] *= sz ;
	ret[1][1] *= sx * sz / sy ;
	ret[1][2] *= sx ;

	ret[2][0] *= sy ;
	ret[2][1] *= sx ;
	ret[2][2] *= sx * sy / sz ;

	if(inverse) return invert(ret) ;

	else return ret;
	**/
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::Conjugate_Tensor(Tensor<2,3, std::complex<double>> input) {
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
Tensor<1,3, std::complex<double>> Waveguide<MatrixType, VectorType>::Conjugate_Vector(Tensor<1,3, std::complex<double>> input) {
	Tensor<1,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		ret[i].real(input[i].real());
		ret[i].imag( - input[i].imag());

	}
	return ret;
}

template<typename MatrixType, typename VectorType  >
bool Waveguide<MatrixType, VectorType>::PML_in_X(Point<3> &p) {
	double pmlboundary = GlobalParams.PRM_M_R_XLength * (1- 2.0*GlobalParams.PRM_M_BC_Mantle)*0.5;
	return p(0) < -(pmlboundary) ||p(0) > (pmlboundary);
}

template<typename MatrixType, typename VectorType>
bool Waveguide<MatrixType, VectorType>::PML_in_Y(Point<3> &p) {
	double pmlboundary = GlobalParams.PRM_M_R_YLength * (1- 2.0*GlobalParams.PRM_M_BC_Mantle) * 0.5;
	return p(1) < -(pmlboundary) ||p(1) > (pmlboundary);
}

template<typename MatrixType, typename VectorType>
bool Waveguide<MatrixType, VectorType>::PML_in_Z(Point<3> &p) {
	return p(2) > (GlobalParams.PRM_M_R_ZLength / 2.0 );
}

template<typename MatrixType, typename VectorType>
bool Waveguide<MatrixType, VectorType>::Preconditioner_PML_in_Z(Point<3> &p, unsigned int block) {
	double l = structure->Sector_Length();
	double width = l * 0.3;
	bool up =    (( p(2) + GlobalParams.PRM_M_R_ZLength/2.0 ) - ((double)block+1.0) * l + width) > 0;
	bool down =  -(( p(2) + GlobalParams.PRM_M_R_ZLength/2.0 ) - ((double)block-1.0) * l - width) > 0;
	//pout <<std::endl<< p(2) << ":" << block << ":" << up << " " << down <<std::endl;
	return up || down;
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block ){
	double l = structure->Sector_Length();
	double width = l * 0.3;
	if( ( p(2) +GlobalParams.PRM_M_R_ZLength/2.0 )-  ((double)block) * l < 0){
		return -(( p(2) + GlobalParams.PRM_M_R_ZLength/2.0  ) - ((double)block-1.0) * l - width);
	} else {
		return  (( p(2) + GlobalParams.PRM_M_R_ZLength/2.0  ) - ((double)block+1.0) * l + width);
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_X_Distance(Point<3> &p){
	//double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	double pmlboundary = GlobalParams.PRM_M_R_YLength * (1- 2.0*GlobalParams.PRM_M_BC_Mantle) * 0.5;
	if(p(0) >0){
		return p(0) - (pmlboundary) ;
	} else {
		return -p(0) - (pmlboundary);
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_Y_Distance(Point<3> &p){
	double pmlboundary = GlobalParams.PRM_M_R_YLength * (1- 2.0*GlobalParams.PRM_M_BC_Mantle) * 0.5;
	if(p(1) >0){
		return p(1) - (pmlboundary);
	} else {
		return -p(1) - (pmlboundary);
	}
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::PML_Z_Distance(Point<3> &p){
	return p(2) - (GlobalParams.PRM_M_R_ZLength / 2.0) ;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::set_boundary_ids (parallel::distributed::Triangulation<3> &tria) const
{

	int counter = 0;
	parallel::shared::Triangulation<3>::active_cell_iterator cell = tria.begin_active();
	parallel::shared::Triangulation<3>::active_cell_iterator endc = tria.end();
	tria.set_all_manifold_ids(0);
	for (; cell!=endc; ++cell){
		if (Distance2D(cell->center() ) < 0.25 ) {
			cell->set_all_manifold_ids(1);
			cell->set_manifold_id(1);
		}
	}
	unsigned int man = 1;

	tria.set_manifold (man, round_description);

//	cell = tria.begin_active(),
//	endc = tria.end();
//	for (; cell!=endc; ++cell){
//		int temp  = std::floor((cell->center(true, false)[2] + 1.0)/len);
//		cell->set_subdomain_id(temp);
//	}
//
	cell = tria.begin_active();

	for (; cell!=endc; ++cell){
		if(cell->at_boundary()){
			for(int j = 0; j<6; j++){
				if(cell->face(j)->at_boundary()){
					Point<3> ctr =cell->face(j)->center(true, false);
					if(System_Coordinate_in_Waveguide(ctr)){
						if(ctr(2) < 0) {

							cell->face(j)->set_all_boundary_ids(11);
							counter ++;
						}

						else {
							cell->face(j)->set_all_boundary_ids(2);
						}
					}
				}
			}
		}
	}




//	cell = tria.begin_active();
//	for (; cell!=endc; ++cell){
//		double distance_from_center = 0;
//		for( int j = 0; j<4; j++) distance_from_center += Distance2D(Point<3> (cell->vertex(j)));
//		if (distance_from_center < 1.2) {
//			cell->set_all_manifold_ids(0);
//		}
//	}


}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::make_grid ()
{
	Point<3> origin(-1,-1,-1);
	std_cxx11::array< Tensor< 1, 3 >, 3 > edges;
	edges[0][0] = 2;
	edges[0][1] = 0;
	edges[0][2] = 0;

	edges[1][0] = 0;
	edges[1][1] = 2;
	edges[1][2] = 0;

	edges[2][0] = 0;
	edges[2][1] = 0;
	edges[2][2] = 2;

	const std_cxx11::array< Tensor< 1, 3 >, 3 > edges2(edges);

	std::vector<unsigned int> subs(3);
	subs[0] = 1;
	subs[1] = 1;
//	subs[2] = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);;
	subs[2] = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
	GridGenerator::subdivided_parallelepiped<3,3>(triangulation,origin, edges2, subs, false);


	triangulation.repartition();
	//	parallel::shared::Triangulation<3>::active_cell_iterator cell, endc;
	triangulation.refine_global(3);

	triangulation.signals.post_refinement.connect
			    (std_cxx11::bind (&Waveguide<MatrixType, VectorType>::set_boundary_ids,
			                      std_cxx11::cref(*this),
			                      std_cxx11::ref(triangulation)));


	triangulation.set_all_manifold_ids(0);

	GridTools::transform( &Triangulation_Stretch_to_circle , triangulation);

	unsigned int man = 1;

	triangulation.set_manifold (man, round_description);

	triangulation.set_all_manifold_ids(0);
	cell = triangulation.begin_active();
	endc = triangulation.end();
	for (; cell!=endc; ++cell){
		if (Distance2D(cell->center() ) < 0.25 ) {
			cell->set_all_manifold_ids(1);
			cell->set_manifold_id(1);
		}
	}


	triangulation.set_manifold (man, round_description);

	//triangulation.refine_global(1);

	triangulation.set_all_manifold_ids(0);
	cell = triangulation.begin_active();
	endc = triangulation.end();
	for (; cell!=endc; ++cell){
		if (Distance2D(cell->center() ) < 0.25 ) {
			cell->set_all_manifold_ids(1);
			cell->set_manifold_id(1);
		}
	}


	triangulation.set_manifold (man, round_description);

	//triangulation.refine_global(0);

//	cell = triangulation.begin_active();
//	endc = triangulation.end();
//
//	for ( ; cell!=endc; ++cell) {
//		if(Distance2D(cell->center()) < 0.3) {
//			cell->set_coarsen_flag();
//		}
//	}
//
//	triangulation.execute_coarsening_and_refinement();

	//triangulation.refine_global(1);

	// mesh_info(triangulation, solutionpath + "/grid" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".vtk");

	// std::cout << "done" << std::endl;

	parallel::shared::Triangulation<3>::active_cell_iterator

	cell = triangulation.begin_active(),
	endc = triangulation.end();
	for ( ; cell!=endc; ++cell) {
		//cell->set_subdomain_id(0);
	}

	int layers_per_sector = 4;
	layers_per_sector /= GlobalParams.PRM_R_Global;
	int reps = log2(layers_per_sector);
	if( layers_per_sector > 0 && pow(2,reps) != layers_per_sector) {
		pout << "The number of layers per sector has to be a power of 2. At least 2 layers are recommended for neccessary sparsity in the pattern for preconditioner to work." << std::endl;
		exit(0);
	}
//	for ( int j = 0; j < reps; j++) {
//		cell = triangulation.begin_active(),
//		endc = triangulation.end();
//		for (; cell!=endc; ++cell){
//			cell->set_refine_flag(dealii::RefinementPossibilities<3>::cut_z);
//		}
//		triangulation.execute_coarsening_and_refinement();
//	}

	double len = 2.0 / Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		int temp  = (int) std::floor((cell->center(true, false)[2] + 1.0)/len);
		if( temp >=  Sectors || temp < 0) pout << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
		//cell->set_subdomain_id(temp);
	}



	GridTools::transform(& Triangulation_Stretch_X, triangulation);
	GridTools::transform(& Triangulation_Stretch_Y, triangulation);
	GridTools::transform(& Triangulation_Stretch_Computational_Radius, triangulation);

	if(prm.PRM_D_Refinement == "global"){
		triangulation.refine_global (prm.PRM_D_XY);
	} else {

		//triangulation.refine_global (GlobalParams.PRM_R_Global);
//		double MaxDistFromBoundary = (GlobalParams.PRM_M_C_RadiusOut + GlobalParams.PRM_M_C_RadiusIn)*1.4/2.0;
		double MaxDistFromBoundary = (GlobalParams.PRM_M_C_RadiusOut + GlobalParams.PRM_M_C_RadiusIn)*1.4/2.0;
		for(int i = 0; i < GlobalParams.PRM_R_Semi; i++) {
			cell = triangulation.begin_active();
			for (; cell!=endc; ++cell){
				if(std::abs(Distance2D(cell->center(true, false)) - (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0 ) < MaxDistFromBoundary) {
					cell->set_refine_flag();
				}
			}
			//triangulation.execute_coarsening_and_refinement();
			//MaxDistFromBoundary *= 0.7 ;
		}
//		MaxDistFromBoundary = (GlobalParams.PRM_M_C_RadiusOut + GlobalParams.PRM_M_C_RadiusIn)*1.4/2.0;
//		for(int i = 0; i < GlobalParams.PRM_R_Internal; i++) {
//			cell = triangulation.begin_active();
//			for (; cell!=endc; ++cell){
//				if( Distance2D(cell->center(true, false))< (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0)  {
//					cell->set_refine_flag();
//				}
//			}
//			triangulation.execute_coarsening_and_refinement();
//		}
	}

	mesh_info(triangulation, solutionpath + "/grid" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".vtk");

	GridTools::transform(& Triangulation_Stretch_Z, triangulation);


	GridTools::transform(& Triangulation_Shift_Z , triangulation);

	GlobalParams.z_min = 10000000.0;
	GlobalParams.z_max = -10000000.0;
	cell = triangulation.begin_active();
	endc = triangulation.end();

	for (; cell!=endc; ++cell){
		if(cell->is_locally_owned()){
			for(int face = 0; face < 6; face++) {
				GlobalParams.z_min = std::min(GlobalParams.z_min, cell->face(face)->center()[2]);
				GlobalParams.z_max = std::max(GlobalParams.z_max, cell->face(face)->center()[2]);
			}
		}
	}

	if(GlobalParams.z_min < (-GlobalParams.PRM_M_R_ZLength/2.0 + 0.00001) && GlobalParams.z_max >= -GlobalParams.PRM_M_R_ZLength/2.0 ) {
		GlobalParams.evaluate_in = true;
	} else {
		GlobalParams.evaluate_in = false;
	}

	if(GlobalParams.z_min <= GlobalParams.PRM_M_R_ZLength/(2.0) && GlobalParams.z_max >= GlobalParams.PRM_M_R_ZLength/2.0 ) {
		GlobalParams.evaluate_out = true;
	} else {
		GlobalParams.evaluate_out = false;
	}

	GlobalParams.z_evaluate = (GlobalParams.z_min + GlobalParams.z_max)/2.0;
	// mesh_info(triangulation, solutionpath + "/grid" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".vtk");

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::Do_Refined_Reordering() {
	std::vector<types::global_dof_index> dof_indices (fe.dofs_per_face);
	std::vector<types::global_dof_index> DofsPerSubdomain(Sectors);
	std::vector<int> InternalBoundaryDofs(Sectors);

	DofsPerSubdomain = dof_handler.n_locally_owned_dofs_per_processor();
	for( int i = 0; i < Sectors; i++) {
		Block_Sizes[i] = DofsPerSubdomain[i];
	}

	Dofs_Below_Subdomain[0] = 0;

	for(int i = 1; i  < Sectors; i++) {
		Dofs_Below_Subdomain[i] = Dofs_Below_Subdomain[i-1] + Block_Sizes[i-1];
	}
	for(int i = 0; i < Sectors; i++) {
		IndexSet temp (dof_handler.n_dofs());
		temp.clear();
		pout << "Adding Block "<< i +1 << " from " << Dofs_Below_Subdomain[i] << " to " << Dofs_Below_Subdomain[i]+ Block_Sizes[i] -1<<std::endl;
		temp.add_range(Dofs_Below_Subdomain[i],Dofs_Below_Subdomain[i]+Block_Sizes[i] );
		set.push_back(temp);
	}
	pout << "Storing details in Waveguidestructure->case_sectors..." <<std::endl;
	for(int i=0; i  < Sectors; i++) {
		structure->case_sectors[i].setLowestDof( Dofs_Below_Subdomain[i] );
		structure->case_sectors[i].setNActiveCells( GridTools::count_cells_with_subdomain_association(triangulation,i) );
		structure->case_sectors[i].setNDofs( Block_Sizes[i] );
		//structure->case_sectors[i].setNInternalBoundaryDofs(InternalBoundaryDofs[i]);
	}
	if(GlobalParams.MPI_Rank == 0 ) {
		GlobalParams.sub_block_lowest =0;
		GlobalParams.block_lowest =0;
		GlobalParams.block_highest = Block_Sizes[0] -1;
	} else {
		GlobalParams.sub_block_lowest = structure->case_sectors[GlobalParams.MPI_Rank-1].LowestDof;
		GlobalParams.block_lowest = structure->case_sectors[GlobalParams.MPI_Rank].LowestDof;
		GlobalParams.block_highest = structure->case_sectors[GlobalParams.MPI_Rank].LowestDof + Block_Sizes[GlobalParams.MPI_Rank]-1;
	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::setup_system ()
{

	if(prm.PRM_O_VerboseOutput && prm.PRM_O_Dofs) {
		pout << "Distributing Degrees of freedom." << std::endl;
	}
	dof_handler.distribute_dofs (fe);
	//dof_handler_real.distribute_dofs (fe);

	if(prm.PRM_O_VerboseOutput) {
		pout << "Renumbering DOFs (Downstream...)" << std::endl;
	}

	// const Point<3> direction(0,0,1);
	// std::vector<unsigned int, std::allocator<unsigned int>> new_dofs(dof_handler.n_dofs());
	// DoFRenumbering::compute_subdomain_wise( new_dofs , dof_handler);

	// IndexSet current = dof_handler.locally_owned_dofs();
	//std::vector<unsigned int, std::allocator<unsigned int>> new_dofs_ordered(dof_handler.n_locally_owned_dofs());
	// pout << "locally owned: " << dof_handler.n_locally_owned_dofs() << " and set size: " << current.size() <<std::endl;
	//for(unsigned int i = 0;i < dof_handler.n_locally_owned_dofs(); i++) {
	// 	new_dofs_ordered[i] = new_dofs[current.nth_index_in_set(i)];
	//}
	//pout << "done" <<std::endl;
	// dof_handler.renumber_dofs(new_dofs);
	//DoFRenumbering::downstream(dof_handler_real, direction, false);

	if(prm.PRM_O_Dofs) {
		pout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
	}

	if(prm.PRM_O_VerboseOutput) {
			pout << "Renumbering DOFs (Custom...)" << std::endl;
	}

	Do_Refined_Reordering();

	pout << "Reordering done." << std::endl;

	locally_owned_dofs = dof_handler.locally_owned_dofs ();
	DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);
	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
	std::vector<unsigned int> n_neighboring = dof_handler.n_locally_owned_dofs_per_processor();
	//locally_relevant_dofs_per_subdomain = DoFTools::locally_relevant_dofs_per_subdomain(dof_handler);
	extended_relevant_dofs = locally_relevant_dofs;
	if(GlobalParams.MPI_Rank > 0) {
		extended_relevant_dofs.add_range(locally_owned_dofs.nth_index_in_set(0) - n_neighboring[GlobalParams.MPI_Rank-1], locally_owned_dofs.nth_index_in_set(0));
	}

	pout << "Constructing Sparsity Patterns and Constrain Matrices ... ";
	// std::cout << GlobalParams.MPI_Rank << ": "<< locally_owned_dofs.is_contiguous() << " , " << locally_owned_dofs.n_elements() << std::endl;
	// std::cout << GlobalParams.MPI_Rank << ": "<< locally_active_dofs.is_contiguous() << " , " << locally_active_dofs.n_elements() << std::endl;
	// std::cout << GlobalParams.MPI_Rank << ": "<< locally_relevant_dofs.is_contiguous() << " , " << locally_relevant_dofs.n_elements() << std::endl;
	// std::cout << GlobalParams.MPI_Rank << ": "<< extended_relevant_dofs.is_contiguous() << " , " << extended_relevant_dofs.n_elements() << std::endl;
	cm.clear();
	cm.reinit(locally_relevant_dofs);

	cm_prec1.clear();
	cm_prec2.clear();
	cm_prec1.reinit(locally_relevant_dofs);
	cm_prec2.reinit(locally_relevant_dofs);
	// std::cout << "Size: " << locally_relevant_dofs.size() << std::endl;

	system_pattern.reinit(locally_owned_dofs, locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
	pout << "done" << std::endl;


//	dynamic_preconditioner_pattern_even.reinit(n_neighboring, n_neighboring);
	preconditioner_pattern.reinit(locally_owned_dofs, locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

//	pout << "Dofs Per CPU: ";
//	for(int i = 0 ; i < n_neighboring.size(); i++) {
//		pout << n_neighboring[i] << " " <<std::endl;
//		for (int j = 0; j < n_neighboring.size(); j++) {
//			IndexSet is(n_neighboring[i]);
//			if(i==GlobalParams.MPI_Rank && i == j) {
//				is.add_range(0,n_neighboring[i]-1);
//			}
//			dynamic_preconditioner_pattern_even.block(i,j).reinit(n_neighboring[i], n_neighboring[j], is);
//		}
//	}
//
//	dynamic_preconditioner_pattern_even.collect_sizes();

	DoFTools::make_hanging_node_constraints(dof_handler, cm);
	DoFTools::make_hanging_node_constraints(dof_handler, cm_prec1);
	DoFTools::make_hanging_node_constraints(dof_handler, cm_prec2);

	MakeBoundaryConditions();
	MakePreconditionerBoundaryConditions();

	cm.close();
	cm_prec1.close();
	cm_prec2.close();

	DoFTools::make_sparsity_pattern(dof_handler, system_pattern, cm, true , Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
	DoFTools::make_sparsity_pattern(dof_handler, preconditioner_pattern, cm_prec1, true , Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
	DoFTools::make_sparsity_pattern(dof_handler, preconditioner_pattern, cm_prec2, true , Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

	//SparsityTools::distribute_sparsity_pattern(dynamic_system_pattern,n_neighboring, MPI_COMM_WORLD, locally_owned_dofs);
	//SparsityTools::distribute_sparsity_pattern(dynamic_preconditioner_pattern_even,n_neighboring, MPI_COMM_WORLD, locally_owned_dofs);

	pout << "done" << std::endl;
	preconditioner_pattern.compress();
	system_pattern.compress();


	pout << "Initialization done." << std::endl;
	// cm.distribute(solution);

	if(prm.PRM_O_VerboseOutput) {
		if(!is_stored) 	pout << "Done." << std::endl;
	}

	// locally_relevant_dofs_all_processors(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

	std::ostringstream set_string;

	locally_owned_dofs.write(set_string);

	std::string local_set = set_string.str();

	const char * test = local_set.c_str();

	char * text_local_set = const_cast<char*> (test);

	// std::cout << "Process " << GlobalParams.MPI_Rank << " has " << text_local_set <<std::endl << " wich has length " << strlen(text_local_set) << std::endl;

	unsigned int text_local_length = strlen( text_local_set) ;

	const int mpi_size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

	int * all_lens = new int[mpi_size];
	int * displs = new int[mpi_size];

	// std::cout << "MaxLength in process " << GlobalParams.MPI_Rank << " before sync is " << text_local_length;

	MPI_Allgather(& text_local_length, 1, MPI_INT, all_lens, 1, MPI_INT, MPI_COMM_WORLD);

	int totlen = all_lens[mpi_size-1];
	displs[0] = 0;
	for (int i=0; i<mpi_size-1; i++) {
		displs[i+1] = displs[i] + all_lens[i];
		totlen += all_lens[i];
	}
	char * all_names = (char *)malloc( totlen );
	if (!all_names) MPI_Abort( MPI_COMM_WORLD, 1 );

	MPI_Allgatherv( text_local_set, text_local_length, MPI_CHAR, all_names, all_lens, displs, MPI_CHAR,	MPI_COMM_WORLD );

	pout << "-------------------------------------" << std::endl;

	locally_relevant_dofs_all_processors.resize(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));


	for(unsigned int i= 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); i++ ) {
		// pout << &all_names[displs[i]]<< std::endl << "++++++++++++++++++++" <<std::endl;
	}

	for(unsigned int i= 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); i++ ) {
		std::istringstream ss;
		char *temp = &all_names[displs[i]] ;
		ss.rdbuf()->pubsetbuf(temp,strlen(temp));
		locally_relevant_dofs_all_processors[i].clear();
		locally_relevant_dofs_all_processors[i].set_size(dof_handler.n_dofs());
		locally_relevant_dofs_all_processors[i].read(ss);
	}

	// std::cout<< "Reading worked in process number " << GlobalParams.MPI_Rank << std::endl;

	UpperDofs = locally_owned_dofs;

	LowerDofs = locally_owned_dofs;


	if(GlobalParams.MPI_Rank != 0 ) {
		LowerDofs.add_indices(locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank-1], 0);
	}

	if(GlobalParams.MPI_Rank != Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) -1 ) {
		UpperDofs.add_indices(locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank+1], 0);
	}

	prec_patterns = new TrilinosWrappers::SparsityPattern[mpi_size-1];

	// std::cout << "Stage 1 for processor " << GlobalParams.MPI_Rank << std::endl;
	// MPI_Comm_split(MPI_COMM_WORLD, GlobalParams.MPI_Rank/2, GlobalParams.MPI_Rank, &comm_even );
	// IndexSet none(dof_handler.n_dofs());

	IndexSet all(dof_handler.n_dofs());
	all.add_range(0, dof_handler.n_dofs());

	IndexSet owned(dof_handler.n_dofs());

	IndexSet writable(dof_handler.n_dofs());

	IndexSet none(dof_handler.n_dofs());

	for(int i =0; i< mpi_size-1; i++) {
		owned.clear();
		writable.clear();
		bool spec = false ;
		bool upper = false;
		bool lower = false;
		int dofs = 0;

		if ( GlobalParams.MPI_Rank -i == 0 ) {
			spec = true;
			lower = true;
		}

		if ( GlobalParams.MPI_Rank - i == 1) {
			upper = true;
			spec = true;
		}


		if(!spec) {
			owned.add_indices(locally_owned_dofs);
			writable.add_indices(locally_relevant_dofs);

			// is2.add_indices(is1);
		} else {
			if(upper) {
				owned = LowerDofs;
				writable = LowerDofs;
				writable.add_indices(locally_relevant_dofs);
				dofs = dof_handler.max_couplings_between_dofs();
			}
			if(lower) {
				owned = none;
				writable = UpperDofs;
				writable.add_indices(locally_relevant_dofs);
			}
		}

		// std::cout << "Stage 4 for processor " << GlobalParams.MPI_Rank << std::endl;


		MPI_Barrier(MPI_COMM_WORLD);

		// prec_patterns[i].reinit(owned, owned, writable, MPI_COMM_WORLD, dofs);

		prec_patterns[i].reinit(owned, owned, writable, MPI_COMM_WORLD, dofs);

		DoFTools::make_sparsity_pattern(dof_handler, prec_patterns[i], cm_prec1, true , Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));


		// prec_patterns[i].reinit(locally_owned_dofs, MPI_COMM_WORLD, 0);
		// std::cout << GlobalParams.MPI_Rank << " has reached the end of loop " << i << std::endl;

	}


	/*
	if(GlobalParams.MPI_Rank != 1) {
		std::cout << GlobalParams.MPI_Rank << ": ( " << Utilities::MPI::this_mpi_process(split_comms[0])<< ","<<Utilities::MPI::this_mpi_process(split_comms[1]) << "," <<Utilities::MPI::this_mpi_process(split_comms[2])<<" - " << Utilities::MPI::n_mpi_processes(split_comms[0])<< ","<<Utilities::MPI::n_mpi_processes(split_comms[1]) << "," <<Utilities::MPI::n_mpi_processes(split_comms[2]) << ")" << std::endl;
	} else {
		std::cout << GlobalParams.MPI_Rank << ": ( " << Utilities::MPI::this_mpi_process(split_comms[0])<< ","<<Utilities::MPI::this_mpi_process(split_comms[1]) << " - " << Utilities::MPI::n_mpi_processes(split_comms[0])<< ","<<Utilities::MPI::n_mpi_processes(split_comms[1]) << ")" << std::endl;
	}


	//MPI_Comm_split(MPI_COMM_WORLD,  GlobalParams.MPI_Rank   /3, GlobalParams.MPI_Rank%3		, &split_comms[GlobalParams.MPI_Rank%3	] );
	//MPI_Comm_split(MPI_COMM_WORLD, ((GlobalParams.MPI_Rank-1)%mpi_size)/3, (GlobalParams.MPI_Rank+1)%3 , &split_comms[(GlobalParams.MPI_Rank+1)%3] );
	//MPI_Comm_split(MPI_COMM_WORLD, ((GlobalParams.MPI_Rank-2)%mpi_size)/3, (GlobalParams.MPI_Rank+2)%3	, &split_comms[(GlobalParams.MPI_Rank+2)%3] );

	//std::cout << "Stage 2 for processor " << GlobalParams.MPI_Rank << std::endl;
	// MPI_Comm_split(MPI_COMM_WORLD, (GlobalParams.MPI_Rank-1)/2  , GlobalParams.MPI_Rank, &comm_odd );
	//std::cout << "Stage 3 for processor " << GlobalParams.MPI_Rank << std::endl;
	IndexSet none(dof_handler.n_dofs());
	IndexSet extend(dof_handler.n_dofs());
	extend.add_range(0,dof_handler.n_dofs());
	int min1 = (GlobalParams.MPI_Rank-1);
	int min2 = (GlobalParams.MPI_Rank-2);
	if (min1 < 0) min1 += mpi_size;
	if (min2 < 0) min2 += mpi_size;
	extend.subtract_set(locally_relevant_dofs_all_processors[min1]);
	extend.subtract_set(locally_relevant_dofs_all_processors[min2]);
	extend.compress();
	none.clear();
	none.add_index(0);
	IndexSet singleIndex(dof_handler.n_dofs());
	singleIndex.add_index(0);
	UpperDofs.subtract_set(singleIndex);
	LowerDofs.subtract_set(singleIndex);
	UpperDofs.compress();
	LowerDofs.compress();
	prec_patterns = new TrilinosWrappers::SparsityPattern[mpi_size-1];
	std::cout << "Process " << GlobalParams.MPI_Rank << " - Lower: " << LowerDofs.n_elements() << " | Upper: " << UpperDofs.n_elements() << " | Extension: " << extend.n_elements()<< std::endl;

	IndexSet * sets = new IndexSet[3];
	sets[0] = none;
	sets[1] = UpperDofs;
	sets[2] = LowerDofs;

	for( int i = 0; i < mpi_size-1; i++) {

		//prec_pattern[i].reinit()
		if( i ==(int) GlobalParams.MPI_Rank) {
			if( GlobalParams.MPI_Rank != mpi_size-1) {
				std::cout << "Process " << GlobalParams.MPI_Rank << " other for Block " << i << "( " << none.n_elements()<< ","<<UpperDofs.n_elements()<<")"<<std::endl;
				prec_patterns[i].reinit(none, none, UpperDofs, 			split_comms[0], dof_handler.max_couplings_between_dofs());
			}
		}
		if(i+1 == (int)GlobalParams.MPI_Rank) {
			if(GlobalParams.MPI_Rank !=0) {
				std::cout << "Process " << GlobalParams.MPI_Rank << " self for Block " << i << "( " << LowerDofs.n_elements()<< ")"<< std::endl;
				prec_patterns[i].reinit(LowerDofs, LowerDofs, LowerDofs, 	split_comms[1], dof_handler.max_couplings_between_dofs());
			}
		}
		int temp = i+2;
		if (temp == mpi_size) temp =0;
		if(temp == (int)GlobalParams.MPI_Rank) {
			if(GlobalParams.MPI_Rank != 1 ) {
				std::cout << "Process " << GlobalParams.MPI_Rank << " extension for Block " << i << "( " << extend.n_elements()<< ")"<< i << std::endl;
				prec_patterns[i].reinit(extend, extend, extend, 			split_comms[2], 1);
			}
		}

		std::cout << "Stage 5 for processor " << GlobalParams.MPI_Rank << " with i=" <<i<<std::endl;
	}
	*/

	std::cout << "Stage 6 for processor " << GlobalParams.MPI_Rank << std::endl;
	reinit_all();

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_all () {
	// pout << "0-";
	reinit_rhs();
	// pout << "1-";
	reinit_solution();
	// pout << "2-";
	reinit_preconditioner();
	// pout << "3-";
	reinit_systemmatrix();
	// pout << "4";
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_preconditioner () {
	/**
	if(!temporary_pattern_preped) {
		preconditioner_pattern.copy_from(prec_pattern);

	}
	preconditioner_matrix_large.reinit( dof_handler.n_dofs(), dof_handler.n_dofs(), dof_handler.max_couplings_between_dofs(), false);
	preconditioner_matrix_small.reinit( GlobalParams.block_highest - GlobalParams.sub_block_lowest + 1, GlobalParams.block_highest - GlobalParams.sub_block_lowest + 1,dof_handler.max_couplings_between_dofs(), false);
	**/
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_rhs () {

	}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_systemmatrix() {
	TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
	                                       locally_owned_dofs,
	                                       locally_relevant_dofs,
	                                       MPI_COMM_WORLD);
	DoFTools::make_sparsity_pattern (dof_handler, sp,
	                                   cm, false,
	                                   Utilities::MPI::
	                                   this_mpi_process(MPI_COMM_WORLD));
	sp.compress();
	system_matrix.reinit( sp);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_solution() {
	/**
	solution.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) solution.block(i).reinit(Block_Sizes[i]);
	solution.collect_sizes();

	temp_storage.reinit(Sectors);
	for (int i = 0; i < Sectors; i++) temp_storage.block(i).reinit(Block_Sizes[i]);
	temp_storage.collect_sizes();
	**/
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_storage() {
	//storage.reinit(Sectors);
	//for (int i = 0; i < Sectors; i++) storage.block(i).reinit(Block_Sizes[i]);
	//storage.collect_sizes();
}

template <>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::reinit_systemmatrix() {

//	IndexSet active(dof_handler.n_dofs());
//	DoFTools::extract_locally_active_dofs(dof_handler, active);
//	TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
//		                                       locally_owned_dofs,
//		                                       active,
//		                                       MPI_COMM_WORLD, dof_handler.max_couplings_between_dofs());
//	DoFTools::make_sparsity_pattern (dof_handler, sp,
//		                                   cm, true,
//		                                   Utilities::MPI::
//		                                   this_mpi_process(MPI_COMM_WORLD));
//	sp.compress();
//
	system_matrix.reinit( system_pattern);
}

template <>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::reinit_rhs () {
	// std::cout << "Reinit rhs for p " << GlobalParams.MPI_Rank << std::endl;

	system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

	preconditioner_rhs.reinit(dof_handler.n_dofs());

}

template <>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::reinit_solution() {
	solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
}

template<>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::Vector>::reinit_storage() {
	storage.reinit(locally_owned_dofs,  MPI_COMM_WORLD);
}

template<>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::reinit_preconditioner () {
	// if(!temporary_pattern_preped) {
	//	preconditioner_pattern.copy_from(prec_pattern);
	// }
//	IndexSet large(dof_handler.n_dofs());
//	large.add_range(0, dof_handler.n_dofs());
	// preconditioner_pattern.copy_from(dynamic_preconditioner_pattern_even);
	// preconditioner_matrix_even.reinit(preconditioner_pattern);

	std::cout << "Reinit precond for p " << GlobalParams.MPI_Rank << std::endl;
	IndexSet all(dof_handler.n_dofs());
	all.add_range(0, dof_handler.n_dofs());

	IndexSet owned(dof_handler.n_dofs());

	IndexSet writable(dof_handler.n_dofs());

	IndexSet none(dof_handler.n_dofs());

	for(int i = 0; i < (int)Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)-1; i++) {
		owned.clear();
		writable.clear();
		bool spec = false ;
		bool upper = false;
		bool lower = false;
		// std::cout << "Stage 2 for processor " << GlobalParams.MPI_Rank << std::endl;


		if ( GlobalParams.MPI_Rank -i == 0 ) {
			spec = true;
			lower = true;
		}

		if ( GlobalParams.MPI_Rank - i == 1) {
			upper = true;
			spec = true;
		}


		if(!spec) {
			owned.add_indices(locally_owned_dofs);
			writable.add_indices(locally_owned_dofs);

			// is2.add_indices(is1);
		} else {
			if(upper) {
				owned = LowerDofs;
				writable = LowerDofs;
				writable.add_indices(locally_relevant_dofs);
			}
			if(lower) {
				owned = none;
				writable = UpperDofs;
				writable.add_indices(locally_relevant_dofs);
			}
		}
		TrilinosWrappers::SparseMatrix temporary;
		prec_patterns[i].compress();
		// std::cout << GlobalParams.MPI_Rank << " compressed" <<std::endl;
		MPI_Barrier(MPI_COMM_WORLD);
		Preconditioner_Matrices[i].reinit(prec_patterns[i]);


	}
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::assemble_part ( ) {
	QGauss<3>  			 quadrature_formula(2);
	FEValues<3> 		fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	FullMatrix<double>	cell_matrix_real (dofs_per_cell, dofs_per_cell);
	FullMatrix<double>	cell_matrix_prec1 (dofs_per_cell, dofs_per_cell);
	FullMatrix<double>	cell_matrix_prec2 (dofs_per_cell, dofs_per_cell);

	Vector<double>		cell_rhs (dofs_per_cell);
	cell_rhs = 0;
	Tensor<2,3, std::complex<double>> 		epsilon, epsilon_pre1, epsilon_pre2, mu, mu_prec1, mu_prec2;
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	DoFHandler<3>::active_cell_iterator cell, endc;
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();

	for (; cell!=endc; ++cell)
	{
		unsigned int subdomain_id = cell->subdomain_id();
		if( subdomain_id == GlobalParams.MPI_Rank) {
			fe_values.reinit (cell);
			quadrature_points = fe_values.get_quadrature_points();

			cell_matrix_real = 0;
			cell_matrix_prec1 = 0;
			cell_matrix_prec2 = 0;
			for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
			{
				epsilon = get_Tensor(quadrature_points[q_index],  false, true);
				mu = get_Tensor(quadrature_points[q_index], true, false);
				epsilon_pre1 = get_Preconditioner_Tensor(quadrature_points[q_index],false, true, subdomain_id);
				mu_prec1 = get_Preconditioner_Tensor(quadrature_points[q_index],true, false, subdomain_id);

				epsilon_pre2 = get_Preconditioner_Tensor(quadrature_points[q_index],false, true, subdomain_id + 1);
				mu_prec2 = get_Preconditioner_Tensor(quadrature_points[q_index],true, false, subdomain_id + 1);

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

						std::complex<double> x = (mu * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.PRM_C_omega*GlobalParams.PRM_C_omega;
						cell_matrix_real[i][j] += x.real();

						std::complex<double> pre1 = (mu_prec1 * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon_pre1 * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.PRM_C_omega*GlobalParams.PRM_C_omega;
						cell_matrix_prec1[i][j] += pre1.real();

						std::complex<double> pre2 = (mu_prec2 * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon_pre2 * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.PRM_C_omega*GlobalParams.PRM_C_omega;
						cell_matrix_prec2[i][j] += pre2.real();
					}
				}
			}
			cell->get_dof_indices (local_dof_indices);
			// pout << "Starting distribution"<<std::endl;
			cm.distribute_local_to_global     (cell_matrix_real, cell_rhs, local_dof_indices,system_matrix, system_rhs, false);
			// pout << "P1 done"<<std::endl;

			if(GlobalParams.MPI_Rank != 0 ) {
				// std::cout << GlobalParams.MPI_Rank << ": pre  1" << std::endl;
				cm_prec1.distribute_local_to_global(cell_matrix_prec1, cell_rhs, local_dof_indices,Preconditioner_Matrices[GlobalParams.MPI_Rank-1], preconditioner_rhs, false);
				// std::cout << GlobalParams.MPI_Rank << ": post 1" << std::endl;
			}
			if(GlobalParams.MPI_Rank != Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)-1) {
				// std::cout << GlobalParams.MPI_Rank << ": pre  2" << std::endl;
				cm_prec2.distribute_local_to_global(cell_matrix_prec2, cell_rhs, local_dof_indices,Preconditioner_Matrices[GlobalParams.MPI_Rank], preconditioner_rhs, false);
				// std::cout << GlobalParams.MPI_Rank << ": post 2" << std::endl;
			}
			// pout << "P2 done"<<std::endl;
	    }
	}
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::assemble_system ()
{

	reinit_rhs();
	// system_rhs.reinit(dof_handler.n_dofs());

	QGauss<3>  quadrature_formula(2);
	const FEValuesExtractors::Vector real(0);
	const FEValuesExtractors::Vector imag(3);
	FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	if(prm.PRM_O_VerboseOutput) {
		if(!is_stored) {
		pout << "Dofs per cell: " << dofs_per_cell << std::endl << "Quadrature Formula Size: " << n_q_points << std::endl;
		pout << "Dofs per face: " << fe.dofs_per_face << std::endl << "Dofs per line: " << fe.dofs_per_line << std::endl;
		}
	}

	log_assemble.start();
	if(!is_stored) pout << "Starting Assemblation process" << std::endl;

	assemble_part( );

	if(!is_stored)  pout<<"Assembling done. L2-Norm of RHS: "<< system_rhs.l2_norm()<<std::endl;
	log_assemble.stop();
	system_matrix.compress(VectorOperation::add);
	system_rhs.compress(VectorOperation::add);

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::MakeBoundaryConditions (){
	DoFHandler<3>::active_cell_iterator cell, endc;
	double sector_length = structure->Sector_Length();

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if(cell->is_locally_owned()) {
			for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
				Point<3, double> center =(cell->face(i))->center(true, false);
				if( center[0] < 0) center[0] *= (-1.0);
				if( center[1] < 0) center[1] *= (-1.0);

				if ( std::abs( center[0] - GlobalParams.PRM_M_R_XLength/2.0) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						if(locally_owned_dofs.is_element(local_dof_indices[0])) {
							cm.add_line(local_dof_indices[0]);
							cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
						}
						if(locally_owned_dofs.is_element(local_dof_indices[1])) {
							cm.add_line(local_dof_indices[1]);
							cm.set_inhomogeneity(local_dof_indices[1], 0.0);
						}
					}
				}
				if ( std::abs( center[1] - GlobalParams.PRM_M_R_YLength/2.0) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						if(locally_owned_dofs.is_element(local_dof_indices[0])) {
							cm.add_line(local_dof_indices[0]);
							cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
						}
						if(locally_owned_dofs.is_element(local_dof_indices[1])) {
							cm.add_line(local_dof_indices[1]);
							cm.set_inhomogeneity(local_dof_indices[1], 0.0);
						}
					}
				}
				if( std::abs(center[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						if((cell->face(i))->line(j)->at_boundary()) {
							((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
							Tensor<1,3,double> ptemp = ((cell->face(i))->line(j))->center(true, false);
							Point<3, double> p (ptemp[0], ptemp[1], ptemp[2]);
							Tensor<1,3,double> dtemp = ((cell->face(i))->line(j))->vertex(0) - ((cell->face(i))->line(j))->vertex(1);
							Point<3, double> direction (dtemp[0], dtemp[1], dtemp[2]);

							double result = TEMode00(p,0);
							if(PML_in_X(p) || PML_in_Y(p)) result = 0.0;
							if(locally_owned_dofs.is_element(local_dof_indices[0])) {
								cm.add_line(local_dof_indices[0]);
								cm.set_inhomogeneity(local_dof_indices[0], direction[0] * result );
							}
							if(locally_owned_dofs.is_element(local_dof_indices[1])) {
								cm.add_line(local_dof_indices[1]);
								cm.set_inhomogeneity(local_dof_indices[1], 0.0);
							}

						}
					}
				}
				if( std::abs(center[2] - GlobalParams.PRM_M_R_ZLength/2.0  - GlobalParams.PRM_M_BC_XYout *sector_length) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						if(locally_owned_dofs.is_element(local_dof_indices[0])) {
							cm.add_line(local_dof_indices[0]);
							cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
						}
						if(locally_owned_dofs.is_element(local_dof_indices[1])) {
							cm.add_line(local_dof_indices[1]);
							cm.set_inhomogeneity(local_dof_indices[1], 0.0);
						}

					}
				}
			}
		}
	}
}

template<typename MatrixType, typename VectorType>
void Waveguide<MatrixType, VectorType>::MakePreconditionerBoundaryConditions (  ){
	DoFHandler<3>::active_cell_iterator cell, endc;
	// cm_prec.clear();
	double l = structure->Sector_Length();
	double width = l * 0.3;
	double sector_length = structure->Sector_Length();
	cell = dof_handler.begin_active();
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if(cell->subdomain_id() == GlobalParams.MPI_Rank || cell->subdomain_id() == GlobalParams.MPI_Rank -1 ) {
			for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
				Point<3, double> center =(cell->face(i))->center(true, false);
				if( center[0] < 0) center[0] *= (-1.0);
				if( center[1] < 0) center[1] *= (-1.0);

				// Set x-boundary values
				if ( std::abs( center[0] - GlobalParams.PRM_M_R_XLength/2.0) < 0.0001){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						for(unsigned int k = 0; k < 2; k++) {
							cm_prec1.add_line(local_dof_indices[k]);
							cm_prec1.set_inhomogeneity(local_dof_indices[k], 0.0 );
							cm_prec2.add_line(local_dof_indices[k]);
							cm_prec2.set_inhomogeneity(local_dof_indices[k], 0.0 );
						}
					}
				}

				// Set y-boundary values
				if ( std::abs( center[1] - GlobalParams.PRM_M_R_YLength/2.0) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						for(unsigned int k = 0; k < 2; k++) {
							cm_prec1.add_line(local_dof_indices[k]);
							cm_prec1.set_inhomogeneity(local_dof_indices[k], 0.0 );
							cm_prec2.add_line(local_dof_indices[k]);
							cm_prec2.set_inhomogeneity(local_dof_indices[k], 0.0 );
						}
					}
				}

				//lower boundary both
				if( GlobalParams.MPI_Rank < 2 ) {
					if( std::abs(center[2] + GlobalParams.PRM_M_R_ZLength/2.0  ) < 0.0001 ){
						std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
						for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
							if((cell->face(i))->line(j)->at_boundary()) {
								((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
								Tensor<1,3,double> ptemp = ((cell->face(i))->line(j))->center(true, false);
								Point<3, double> p (ptemp[0], ptemp[1], ptemp[2]);
								Tensor<1,3,double> dtemp = ((cell->face(i))->line(j))->vertex(0) - ((cell->face(i))->line(j))->vertex(1);
								Point<3, double> direction (dtemp[0], dtemp[1], dtemp[2]);

								double result = TEMode00(p,0);
								if(PML_in_X(p) || PML_in_Y(p)) result = 0.0;
								cm_prec1.add_line(local_dof_indices[0]);
								cm_prec1.set_inhomogeneity(local_dof_indices[0], direction[0] * result);
								cm_prec1.add_line(local_dof_indices[1]);
								cm_prec1.set_inhomogeneity(local_dof_indices[1], 0.0);
								cm_prec2.add_line(local_dof_indices[0]);
								cm_prec2.set_inhomogeneity(local_dof_indices[0], direction[0] * result);
								cm_prec2.add_line(local_dof_indices[1]);
								cm_prec2.set_inhomogeneity(local_dof_indices[1], 0.0);
							}
						}
					}
				}

				//upper boundary both
				if((int)GlobalParams.MPI_Rank >= Sectors-2 ) {
					if( std::abs(center[2] - GlobalParams.PRM_M_R_ZLength/2.0  - GlobalParams.PRM_M_BC_XYout * sector_length) < 0.0001 ){
						std::vector<types::global_dof_index> local_dof_indices ( fe.dofs_per_line);
						for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
							((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
							for(unsigned int k = 0; k < 2; k++) {
								cm_prec1.add_line(local_dof_indices[k]);
								cm_prec1.set_inhomogeneity(local_dof_indices[k], 0.0 );
								cm_prec2.add_line(local_dof_indices[k]);
								cm_prec2.set_inhomogeneity(local_dof_indices[k], 0.0 );
							}
						}
					}
				}
				// in between below
				if(GlobalParams.MPI_Rank >  1) {
					// if( std::abs( (center[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) - ((double)(GlobalParams.MPI_Rank-1))*sector_length ) < 0.0001 ){
					if( std::abs( (center[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) - (GlobalParams.MPI_Rank -1 )*sector_length ) < width/3.0 ){
						std::vector<types::global_dof_index> local_dof_indices ( fe.dofs_per_line);
						for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
							((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
							for(unsigned int k = 0; k < 2; k++) {
								cm_prec1.add_line(local_dof_indices[k]);
								cm_prec1.set_inhomogeneity(local_dof_indices[k], 0.0 );

							}
						}
					}
				}

				if(GlobalParams.MPI_Rank >  1) {
					// if( std::abs( (center[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) - ((double)(GlobalParams.MPI_Rank-1))*sector_length ) < 0.0001 ){
					if( std::abs( (center[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) - ( GlobalParams.MPI_Rank )*sector_length ) <  width/3.0 ){
						std::vector<types::global_dof_index> local_dof_indices ( fe.dofs_per_line);
						for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
							((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
							for(unsigned int k = 0; k < 2; k++) {
								cm_prec2.add_line(local_dof_indices[k]);
								cm_prec2.set_inhomogeneity(local_dof_indices[k], 0.0 );

							}
						}
					}
				}

				if((int)GlobalParams.MPI_Rank < Sectors -1) {
					if( std::abs( (center[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) - (GlobalParams.MPI_Rank +1)*sector_length ) <  width/3.0 ){
						std::vector<types::global_dof_index> local_dof_indices ( fe.dofs_per_line);
						for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
							((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
							for(unsigned int k = 0; k < 2; k++) {
								cm_prec1.add_line(local_dof_indices[k]);
								cm_prec1.set_inhomogeneity(local_dof_indices[k], 0.0 );

							}
						}
					}
				}

				if((int)GlobalParams.MPI_Rank < Sectors -1) {
					if( std::abs( (center[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) - (GlobalParams.MPI_Rank + 2)*sector_length ) <  width/3.0 ){
						std::vector<types::global_dof_index> local_dof_indices ( fe.dofs_per_line);
						for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
							((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
							for(unsigned int k = 0; k < 2; k++) {
								cm_prec2.add_line(local_dof_indices[k]);
								cm_prec2.set_inhomogeneity(local_dof_indices[k], 0.0 );

							}
						}
					}
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

template<typename MatrixType, typename VectorType >
SolverControl::State  Waveguide<MatrixType, VectorType>::check_iteration_state (const unsigned int iteration, const double check_value, const VectorType & ){
	SolverControl::State ret = SolverControl::State::iterate;
	if(iteration > GlobalParams.PRM_S_Steps){
		// pout << std::endl;
		return SolverControl::State::failure;
	}
	if(check_value < GlobalParams.PRM_S_Precision){
		// pout << std::endl;
		return SolverControl::State::success;
	}
	pout << '\r';
	pout << "Iteration: " << iteration << "\t Precision: " << check_value;

	iteration_file << iteration << "\t" << check_value << "    " << std::endl;
	iteration_file.flush();
	return ret;
}

template< >
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector >::solve () {

	log_precondition.start();
	result_file.open((solutionpath + "/solution_of_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() + ".dat").c_str());

	if(GlobalParams.MPI_Rank != 0) {
		std::ofstream pattern (solutionpath + "/pattern" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".gnu");
		prec_patterns[GlobalParams.MPI_Rank -1].print_gnuplot(pattern);
	}

	if(prm.PRM_S_Solver == "GMRES") {

		dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control , dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData( prm.PRM_S_GMRESSteps) );
		timerupdate();
		if(prm.PRM_S_Preconditioner == "Sweeping"){
			// std::cout << GlobalParams.MPI_Rank << " prep dofs." <<std::endl;
			IndexSet own (dof_handler.n_dofs());
			own.add_indices(locally_owned_dofs);

			if(GlobalParams.MPI_Rank == 0 ){
				// own.add_indices(locally_owned_dofs);
			} else {
				// own.add_indices(LowerDofs);
			}

			// std::cout << GlobalParams.MPI_Rank << " prep matrix." <<std::endl;
			dealii::SparsityPattern temp_pattern;
			temp_pattern.reinit(own.n_elements(),own.n_elements(), dof_handler.max_couplings_between_dofs());

			if(GlobalParams.MPI_Rank == 0 ){

				for (unsigned int current_row = 0; current_row < own.n_elements(); current_row++  ) {
					for(TrilinosWrappers::SparseMatrix::iterator row = system_matrix.begin(own.nth_index_in_set(current_row)); row != system_matrix.end(own.nth_index_in_set(current_row)); row++) {
						if(own.is_element(row->column())) {
							temp_pattern.add(current_row, own.index_within_set(row->column()));
						}
					}
				}

			} else {

				for (unsigned int current_row = 0; current_row < own.n_elements(); current_row++  ) {
					for(TrilinosWrappers::SparseMatrix::iterator row = Preconditioner_Matrices[GlobalParams.MPI_Rank-1].begin(own.nth_index_in_set(current_row)); row != Preconditioner_Matrices[GlobalParams.MPI_Rank-1].end(own.nth_index_in_set(current_row)); row++) {
						if(own.is_element(row->column())) {
							temp_pattern.add(current_row, own.index_within_set(row->column()));
						}
					}
				}

			}
			temp_pattern.compress();
			std::ofstream pattern (solutionpath + "/pattern" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".gnu");
			temp_pattern.print_gnuplot(pattern);
			dealii::SparseMatrix<double> prec_matrix(temp_pattern);

			// std::cout << GlobalParams.MPI_Rank << " build matrix." <<std::endl;
			if(GlobalParams.MPI_Rank == 0 ){
				for (unsigned int current_row = 0; current_row < own.n_elements(); current_row++  ) {
					for(TrilinosWrappers::SparseMatrix::iterator row = system_matrix.begin(own.nth_index_in_set(current_row)); row != system_matrix.end(own.nth_index_in_set(current_row)); row++) {
						if(own.is_element(row->column())) {
							prec_matrix.set(current_row, own.index_within_set(row->column()), row->value());
						}
					}
				}
			} else {
				for (unsigned int current_row = 0; current_row < own.n_elements(); current_row++  ) {
					for(TrilinosWrappers::SparseMatrix::iterator row = Preconditioner_Matrices[GlobalParams.MPI_Rank-1].begin(own.nth_index_in_set(current_row)); row != Preconditioner_Matrices[GlobalParams.MPI_Rank-1].end(own.nth_index_in_set(current_row)); row++) {
						if(own.is_element(row->column())) {
							prec_matrix.set(current_row, own.index_within_set(row->column()), row->value());
						}
					}
				}
			}


			// prec_matrix.compress(VectorOperation::insert);
			int empty_rows = 0;
			double average = 0.0;
			const unsigned int max_row = own.n_elements();
			// for ( unsigned int i = 0; i < max_row; i++) {
				// const unsigned int temp = prec_matrix.row_length(i);
				// if (temp == 0) {
				//	empty_rows++;
				// }
				//average += (double)temp / (double)max_row;
			// }

			// std::cout << GlobalParams.MPI_Rank << " has " << empty_rows << " empty rows and an average of " << average << " Elements."<<std::endl;

			std::cout << GlobalParams.MPI_Rank << " done building matrix. Init Sweep." <<std::endl;
			int below = 0;
			if (GlobalParams.MPI_Rank != 0 ) {
				//below = locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank-1].n_elements();
			}

			dealii::SparseDirectUMFPACK preconditioner_solver;
			preconditioner_solver.initialize(prec_matrix, dealii::SparseDirectUMFPACK::AdditionalData());
			TrilinosWrappers::SolverDirect prec_sol(solver_control, TrilinosWrappers::SolverDirect::AdditionalData(true, "Amesos_Mumps"));


			PreconditionerSweeping sweep( &preconditioner_solver,  locally_owned_dofs.n_elements(), below);
			std::cout << GlobalParams.MPI_Rank << " ready to solve" <<std::endl;
			MPI_Barrier(MPI_COMM_WORLD);
			solver.solve(system_matrix,solution, system_rhs, sweep);

		}


		pout << "A Solution was calculated!" <<std::endl;
		log_solver.stop();
	}


 /**
	SolverControl cn;
	PETScWrappers::SparseDirectMUMPS solver(cn, MPI_COMM_WORLD);
	//solver.set_symmetric_mode(true);
	solver.solve(system_matrix, solution, system_rhs);
	**/
	//solution.compress(VectorOperation::insert);

	cm.distribute(solution);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::Analyse() {
	unsigned int dofs_equal = 0;
	unsigned int dofs_prec_missing = 0;
	unsigned int dofs_system_missing = 0;
	unsigned int dofs_differen_interior = 0;
	int below = 0;
	IndexSet relevant = locally_owned_dofs;

	if (GlobalParams.MPI_Rank != 0 ) {
		below = locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank-1].n_elements();
		relevant.add_indices(locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank-1]);
	}


	for (unsigned int current_row = 0; current_row < relevant.n_elements(); current_row++  ) {
		for(TrilinosWrappers::SparseMatrix::iterator row = Preconditioner_Matrices[GlobalParams.MPI_Rank-1].begin(relevant.nth_index_in_set(current_row)); row != Preconditioner_Matrices[GlobalParams.MPI_Rank-1].end(relevant.nth_index_in_set(current_row)); row++) {
			if(relevant.is_element(row->column())) {
				// prec_matrix.set(current_row, relevant.index_within_set(row->column()), row->value());
			}
		}
	}

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::solve ()
{


}

template< >
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::Vector>::store() {
	reinit_storage();
	// storage.reinit(dof_handler.n_dofs());
	storage = solution;
	is_stored = true;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::init_loggers () {
	log_data.PML_in 				= 	prm.PRM_M_BC_XYin;
	log_data.PML_out 				=	prm.PRM_M_BC_XYout;
	log_data.PML_mantle 			=	prm.PRM_M_BC_Mantle;
	log_data.ParamSteps 			=	prm.PRM_S_Steps;
	log_data.Precondition_BlockSize = 	0;
	log_data.Precondition_weight 	=	0;
	log_data.Solver_Precision 		=	prm.PRM_S_Precision;
	log_data.XLength				=	prm.PRM_M_R_XLength;
	log_data.YLength 				= 	prm.PRM_M_R_YLength;
	log_data.ZLength 				= 	prm.PRM_M_R_ZLength;
	log_data.preconditioner 		=	prm.PRM_S_Preconditioner;
	log_data.solver 				= 	prm.PRM_S_Solver;
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
void Waveguide<MatrixType, VectorType>::output_results ()
{

	// evaluate_overall();
	if(GlobalParams.MPI_Rank == 0) {
		DataOut<3> data_out;

		data_out.attach_dof_handler (dof_handler);
		data_out.add_data_vector (solution, "solution");
		// data_out.add_data_vector(differences, "L2error");

		data_out.build_patches ();

		std::ofstream outputvtk (solutionpath + "/solution-run" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() +".vtk");
		data_out.write_vtk(outputvtk);

		/**
		DataOut<3> data_out_real;

		//data_out_real.attach_dof_handler(dof_handler_real);
		data_out_real.add_data_vector (solution, "solution");
		// data_out.add_data_vector(differences, "L2error");

		data_out_real.build_patches ();

		std::ofstream outputvtk2 (solutionpath + "/solution-real" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() +".vtk");
		data_out_real.write_vtk(outputvtk2);
		 **/
		std::ofstream pattern (solutionpath + "/pattern.gnu");
		system_pattern.print_gnuplot(pattern);


		std::ofstream patternscript (solutionpath + "/displaypattern.gnu");
		patternscript << "set style line 1000 lw 1 lc \"black\"" <<std::endl;
		for(int i = 0; i < prm.PRM_M_W_Sectors; i++) {
			patternscript << "set arrow " << 1000 + 2*i << " from 0,-" << Dofs_Below_Subdomain[i] << " to "<<dof_handler.n_dofs()<<",-"<<Dofs_Below_Subdomain[i]<<" nohead ls 1000 front"<<std::endl;
			patternscript << "set arrow " << 1001 + 2*i  << " from " << Dofs_Below_Subdomain[i] << ",0 to " << Dofs_Below_Subdomain[i] << ", -"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;
		}
		patternscript << "set arrow " << 1000 + 2*prm.PRM_M_W_Sectors << " from 0,-" << dof_handler.n_dofs() << " to "<<dof_handler.n_dofs()<<",-"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;
		patternscript << "set arrow " << 1001 + 2*prm.PRM_M_W_Sectors << " from " << dof_handler.n_dofs() << ",0 to " << dof_handler.n_dofs() << ", -"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;

		patternscript << "plot \"pattern.gnu\" with dots" <<std::endl;
		patternscript.flush();

		std::ifstream source("Parameters.xml", std::ios::binary);
		std::ofstream dest(solutionpath +"/Parameters.xml", std::ios::binary);

		dest << source.rdbuf();

		source.close();
		dest.close();
	}
}

template<typename MatrixType, typename VectorType>
void Waveguide<MatrixType, VectorType>::run ()
{
	init_loggers ();
	make_grid ();
	setup_system ();
	assemble_system ();
	solve ();
	output_results ();
	log_total.stop();
	run_number++;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::print_eigenvalues(const std::vector<std::complex<double>> &input) {
	for (unsigned int i = 0; i < input.size(); i++){
		eigenvalue_file << input.at(i).real() << "\t" << input.at(i).imag() << std::endl;
	}
	eigenvalue_file << std::endl;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::print_condition(double condition) {
	condition_file << condition << std::endl;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reset_changes ()
{
	reinit_all();

	// cm.distribute(solution);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::rerun ()
{
	reset_changes();
	assemble_system ();
	solve ();
	output_results ();
	log_total.stop();

	run_number++;
}

#endif
