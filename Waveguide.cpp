#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "Waveguide.h"
#include "staticfunctions.cpp"
#include "WaveguideStructure.h"
#include "SolutionWeight.h"
#include "ExactSolution.h"
#include <string>
#include <sstream>
#include <deal.II/base/timer.h>
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
#include <deal.II/numerics/data_out_dof_data.h>
using namespace dealii;

template<typename MatrixType, typename VectorType >
Waveguide<MatrixType, VectorType>::Waveguide (Parameters &param )
  :
  fe(FE_Nedelec<3> (0), 2),
  triangulation (MPI_COMM_WORLD, typename parallel::distributed::Triangulation<3>::MeshSmoothing(Triangulation<3>::none ), parallel::distributed::Triangulation<3>::Settings::no_automatic_repartitioning),
  dof_handler (triangulation),
  prm(param),
  run_number(0),
  condition_file_counter(0),
  eigenvalue_file_counter(0),
  Sectors(prm.PRM_M_W_Sectors),
  Layers(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
  Dofs_Below_Subdomain(Layers),
  Block_Sizes(Layers),
  temporary_pattern_preped(false),
  real(0),
  imag(3),
  solver_control (prm.PRM_S_Steps, prm.PRM_S_Precision, (GlobalParams.MPI_Rank == 0), true),
  pout(std::cout, GlobalParams.MPI_Rank==0),
  timer(MPI_COMM_WORLD, pout, TimerOutput::OutputFrequency::summary, TimerOutput::wall_times)


{
	prec_patterns = new TrilinosWrappers::SparsityPattern[Layers-1];
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
	Dofs_Below_Subdomain[Layers];
	mkdir(solutionpath.c_str(), ACCESSPERMS);
	pout << "Will write solutions to " << solutionpath << std::endl;

	if(GlobalParams.MPI_Rank == 0) {
		std::ifstream source("Parameters.xml", std::ios::binary);
		std::ofstream dest(solutionpath +"/Parameters.xml", std::ios::binary);
		dest << source.rdbuf();
		source.close();
		dest.close();
	}

	is_stored = false;
	solver_control.log_frequency(10);
	const int number = Layers -1;
	Preconditioner_Matrices = new TrilinosWrappers::SparseMatrix[number];
	deallog.attach( std::cout );
	qualities = new double[number];
	execute_recomputation = false;
}

template<typename MatrixType, typename VectorType>
Waveguide<MatrixType, VectorType>::~Waveguide() {

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
	std::complex<double> c1(result(0), - result(3));
	std::complex<double> c2(result(1), - result(4));
	std::complex<double> c3(result(2), - result(5));

	return mode(0) * c1 + mode(1)*c2 + mode(2)*c3;

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

	std::complex<double> res = gauss_product_2D_sphere(z,10,r,0,0);
	return std::sqrt(std::norm(res));
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
void Waveguide<MatrixType, VectorType>::mark_changed() {
	execute_recomputation = true;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::mark_unchanged() {
	execute_recomputation = false;
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::evaluate() {
	pout << "Starting Evaluation" << std::endl;
	double z_for_evaluation = (double)(0.5+GlobalParams.MPI_Rank)*structure->Layer_Length() - GlobalParams.PRM_M_R_ZLength/2.0 ;
	double local_value = evaluate_for_z(z_for_evaluation) ;
	MPI_Allgather( & local_value, 1, MPI_DOUBLE, qualities, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	pout << "Done Gathering Qualities!"<< std::endl;
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::evaluate_overall () {
	std::vector <double> qualities(Layers);
	double lower = 0.0;
	double upper = 0.0;
	if(GlobalParams.evaluate_in) {
		pout << "Evaluation for input side by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << -GlobalParams.PRM_M_R_ZLength / 2.0 << std::endl;
		lower = evaluate_for_z(-GlobalParams.PRM_M_R_ZLength / 2.0 + 0.0001);
	}
	if(GlobalParams.evaluate_out) {
		pout << "Evaluation for output side by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << GlobalParams.PRM_M_R_ZLength / 2.0 << std::endl;
		upper = evaluate_for_z(GlobalParams.PRM_M_R_ZLength / 2.0 - 0.0001);
	}
	lower = Utilities::MPI::sum(lower, GlobalParams.MPI_Communicator);
	upper = Utilities::MPI::sum(upper, GlobalParams.MPI_Communicator);

	for(unsigned int i = 0; i< Layers; i++) {
		double contrib = 0.0;
		if(i == GlobalParams.MPI_Rank) {
			 std::cout << "Evaluation of contribution by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << GlobalParams.z_evaluate<<":" ;
			 contrib = evaluate_for_z(GlobalParams.z_evaluate);
			 std::cout << contrib << std::endl;
		}
		qualities[i] = Utilities::MPI::max(contrib, GlobalParams.MPI_Communicator);
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
	for(unsigned int i =0; i < Layers; i++) {
			pout << 100 * qualities[i]/ lower << "% ";
			if(i != Layers-1) pout << " -> ";
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

template<>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector >::estimate_solution() {
	MPI_Barrier(MPI_COMM_WORLD);
	pout << "Starting solution estimation..." << std::endl;
	DoFHandler<3>::active_cell_iterator cell, endc;
	pout << "Lambda: " << GlobalParams.PRM_M_W_Lambda << std::endl;
	unsigned int min_dof = locally_owned_dofs.nth_index_in_set(0);
	unsigned int max_dof = locally_owned_dofs.nth_index_in_set(locally_owned_dofs.n_elements()-1 );

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if(cell->is_locally_owned()) {
			for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
				std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
				for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {

					((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
					Tensor<1,3,double> ptemp = ((cell->face(i))->line(j))->center(true, false);
					if( std::abs(ptemp[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) > 0.0001 ){
						Point<3, double> p (ptemp[0], ptemp[1], ptemp[2]);
						Tensor<1,3,double> dtemp = ((cell->face(i))->line(j))->vertex(0) - ((cell->face(i))->line(j))->vertex(1);
						dtemp = dtemp / dtemp.norm();
						Point<3, double> direction (dtemp[0], dtemp[1], dtemp[2]);


						//double phi = (ptemp[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) *2 * GlobalParams.PRM_C_PI / (GlobalParams.PRM_M_W_Lambda / GlobalParams.PRM_M_W_EpsilonIn);
						double phi = (ptemp[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) * 2* GlobalParams.PRM_C_PI / (GlobalParams.PRM_M_W_Lambda / std::sqrt(GlobalParams.PRM_M_W_EpsilonIn));
						double result_real = TEMode00(p,0) * std::cos(phi) ;
						double result_imag = - TEMode00(p,0) * std::sin(phi) ;
						if(PML_in_X(p) || PML_in_Y(p)) result_real = 0.0;
						if(PML_in_X(p) || PML_in_Y(p)) result_imag = 0.0;

						if(local_dof_indices[0] >= min_dof && local_dof_indices[0] < max_dof) {
							EstimatedSolution[local_dof_indices[0]] = direction[0] * result_real ;
						}
						if(local_dof_indices[1] >= min_dof && local_dof_indices[1] < max_dof) {
							EstimatedSolution[local_dof_indices[1]] = direction[0] * result_imag ;
						}
					}
				}
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	EstimatedSolution.compress(VectorOperation::insert);
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::get_Tensor(Point<3> & position, bool inverse , bool epsilon) {
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

	

	Tensor<2,3, std::complex<double>> ret3;

	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			ret3[i][j] = std::complex<double>(0.0, 0.0);
			for(int k = 0; k < 3; k++) {
				ret3[i][j] += ret[i][k] * ret2[k][j];
			}
		}
	}

	if  ( inverse ) ret3 = invert(ret3);

	return ret3;
}

template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> Waveguide<MatrixType, VectorType>::get_Preconditioner_Tensor(Point<3> & position, bool inverse , bool epsilon, int block) {
	std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);
	Tensor<2,3, std::complex<double>> ret;

	Tensor<2,3, std::complex<double>> MaterialTensor;
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
			MaterialTensor[i][j] = transformation[i][j]* std::complex<double>(1.0, 0.0);
		}
	}

	double omegaepsilon0 = GlobalParams.PRM_C_omega * ((System_Coordinate_in_Waveguide(position))?GlobalParams.PRM_M_W_EpsilonIn : GlobalParams.PRM_M_W_EpsilonOut);
	std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0),sz_p(0.0,0.0);
	if(PML_in_X(position)){
		double r,d, sigmax;
		r = PML_X_Distance(position);
		d = GlobalParams.PRM_M_R_XLength * 1.0 * GlobalParams.PRM_M_BC_Mantle;
		sigmax = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaXMax;
		sx.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaXMax);
		sx.imag( sigmax / ( omegaepsilon0));
	}
	if(PML_in_Y(position)){
		double r,d, sigmay;
		r = PML_Y_Distance(position);
		d = GlobalParams.PRM_M_R_YLength * 1.0 * GlobalParams.PRM_M_BC_Mantle;
		sigmay = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaYMax;
		sy.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaYMax);
		sy.imag( sigmay / ( omegaepsilon0));
	}
	if(Preconditioner_PML_in_Z(position, block)){
		double r,d, sigmaz;
		r = Preconditioner_PML_Z_Distance(position, block);
		d = structure->Layer_Length() * 1.0;
		sigmaz = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaZMax;
		sz_p.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaZMax);
		sz_p.imag( sigmaz / omegaepsilon0 );
	}

	if(PML_in_Z(position)){
		double r,d, sigmaz;
		r = PML_Z_Distance(position);
		d = GlobalParams.PRM_M_BC_XYout * structure->Sector_Length();
		sigmaz = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaZMax;

		sz.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaZMax);
		sz.imag( sigmaz / omegaepsilon0 );
	}

	sz += sz_p;

	MaterialTensor[0][0] *= sy*sz/sx;
	MaterialTensor[0][1] *= sz;
	MaterialTensor[0][2] *= sy;

	MaterialTensor[1][0] *= sz;
	MaterialTensor[1][1] *= sx*sz/sy;
	MaterialTensor[1][2] *= sx;

	MaterialTensor[2][0] *= sy;
	MaterialTensor[2][1] *= sx;
	MaterialTensor[2][2] *= sx*sy/sz;

    if(epsilon) {
		if(System_Coordinate_in_Waveguide(position) ) {
			MaterialTensor *= GlobalParams.PRM_M_W_EpsilonIn;
		} else {
			MaterialTensor *= GlobalParams.PRM_M_W_EpsilonOut;
		}
		MaterialTensor *= GlobalParams.PRM_C_Eps0;
	}
    

	//pout << "get_Tensor_2" << std::endl;
	if  ( inverse ) MaterialTensor = invert(MaterialTensor);

	return MaterialTensor;

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
	double l = structure->Layer_Length();
	double width = l * 1.0;
	if( block == GlobalParams.MPI_Size-2) return false;
	if ( block == GlobalParams.MPI_Rank-1){
		return true;
	} else {
		return false;
	}
	// bool up =    (( p(2) + GlobalParams.PRM_M_R_ZLength/2.0 ) - ((double)block+1.0) * l + width) > 0;
	// bool down =  -(( p(2) + GlobalParams.PRM_M_R_ZLength/2.0 ) - ((double)block-1.0) * l - width) > 0;
	// pout <<std::endl<< p(2) << ":" << block << ":" << up << " " << down <<std::endl;
	// return up || down;
	// return down;
}

template<typename MatrixType, typename VectorType >
double Waveguide<MatrixType, VectorType>::Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block ){
	double l = structure->Layer_Length();
	double width = l * 1.0;

	return p(2) +GlobalParams.PRM_M_R_ZLength/2.0 - ((double)block +1)*l;
	/**
	if( ( p(2) +GlobalParams.PRM_M_R_ZLength/2.0 )-  ((double)block) * l < 0){
		return -(( p(2) + GlobalParams.PRM_M_R_ZLength/2.0  ) - ((double)block-1.0) * l - width);
	}

	else {
		return  (( p(2) + GlobalParams.PRM_M_R_ZLength/2.0  ) - ((double)block+1.0) * l + width);
	}**/
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
//	subs[2] = Layers;;
	subs[2] = Layers;
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

	double len = 2.0 / Layers;

	cell = triangulation.begin_active();
	for (; cell!=endc; ++cell){
		unsigned int temp  = (int) std::floor((cell->center(true, false)[2] + 1.0)/len);
		if( temp >=  Layers || temp < 0) pout << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
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
			triangulation.execute_coarsening_and_refinement();
			MaxDistFromBoundary = (MaxDistFromBoundary + ((GlobalParams.PRM_M_C_RadiusOut + GlobalParams.PRM_M_C_RadiusIn)/2.0))/2.0 ;
		}

		for(int i = 0; i < GlobalParams.PRM_R_Internal; i++) {
			cell = triangulation.begin_active();
			for (; cell!=endc; ++cell){
				if( Distance2D(cell->center(true, false))< (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0)  {
					cell->set_refine_flag();
				}
			}
			triangulation.execute_coarsening_and_refinement();
		}
	}

	// mesh_info(triangulation, solutionpath + "/grid" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".vtk");

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

    cell = triangulation.begin_active();
	endc = triangulation.end();

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::Do_Refined_Reordering() {
	std::vector<types::global_dof_index> dof_indices (fe.dofs_per_face);
	std::vector<types::global_dof_index> DofsPerSubdomain(Layers);
	std::vector<int> InternalBoundaryDofs(Layers);

	DofsPerSubdomain = dof_handler.n_locally_owned_dofs_per_processor();
	for( unsigned int i = 0; i < Layers; i++) {
		Block_Sizes[i] = DofsPerSubdomain[i];
	}

	Dofs_Below_Subdomain[0] = 0;

	for(unsigned int i = 1; i  < Layers; i++) {
		Dofs_Below_Subdomain[i] = Dofs_Below_Subdomain[i-1] + Block_Sizes[i-1];
	}
	for(unsigned int i = 0; i < Layers; i++) {
		IndexSet temp (dof_handler.n_dofs());
		temp.clear();
		pout << "Adding Block "<< i +1 << " from " << Dofs_Below_Subdomain[i] << " to " << Dofs_Below_Subdomain[i]+ Block_Sizes[i] -1<<std::endl;
		temp.add_range(Dofs_Below_Subdomain[i],Dofs_Below_Subdomain[i]+Block_Sizes[i] );
		set.push_back(temp);
	}
	pout << "Storing details in Waveguidestructure->case_sectors..." <<std::endl;
	/**
	for(unsigned int i=0; i  < Layers; i++) {
		structure->case_sectors[i].setLowestDof( Dofs_Below_Subdomain[i] );
		structure->case_sectors[i].setNActiveCells( GridTools::count_cells_with_subdomain_association(triangulation,i) );
		structure->case_sectors[i].setNDofs( Block_Sizes[i] );
		//structure->case_sectors[i].setNInternalBoundaryDofs(InternalBoundaryDofs[i]);
	}
	**/

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

	if(prm.PRM_O_Dofs) {
		pout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
	}

	if(prm.PRM_O_VerboseOutput) {
			pout << "Renumbering DOFs (Custom...)" << std::endl;
	}

	// Do_Refined_Reordering();

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
	pout << "Done" << std::endl;


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

	DoFTools::make_sparsity_pattern(dof_handler, system_pattern, cm, true , GlobalParams.MPI_Rank);
	DoFTools::make_sparsity_pattern(dof_handler, preconditioner_pattern, cm_prec1, true , GlobalParams.MPI_Rank);
	DoFTools::make_sparsity_pattern(dof_handler, preconditioner_pattern, cm_prec2, true , GlobalParams.MPI_Rank);

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

	// locally_relevant_dofs_all_processors(Layers);

	std::ostringstream set_string;

	locally_owned_dofs.write(set_string);

	std::string local_set = set_string.str();

	const char * test = local_set.c_str();

	char * text_local_set = const_cast<char*> (test);

	// std::cout << "Process " << GlobalParams.MPI_Rank << " has " << text_local_set <<std::endl << " wich has length " << strlen(text_local_set) << std::endl;

	unsigned int text_local_length = strlen( text_local_set) ;

	const int mpi_size = Layers;

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

	locally_relevant_dofs_all_processors.resize(Layers);


	for(unsigned int i= 0; i < Layers; i++ ) {
		// pout << &all_names[displs[i]]<< std::endl << "++++++++++++++++++++" <<std::endl;
	}

	for(unsigned int i= 0; i < Layers; i++ ) {
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

	if(GlobalParams.MPI_Rank != Layers -1 ) {
		UpperDofs.add_indices(locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank+1], 0);
	}



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
				owned = none;
				// owned = LowerDofs;
				writable = locally_relevant_dofs;
			}
			if(lower) {
				// owned = none;
				owned = UpperDofs;
				writable = locally_relevant_dofs;
				writable.add_indices(UpperDofs);
				dofs =  2* dof_handler.max_couplings_between_dofs();
				// UpperDofs;
				// writable.add_indices(locally_relevant_dofs);
			}
		}

		// std::cout << "Stage 4 for processor " << GlobalParams.MPI_Rank << std::endl;


		MPI_Barrier(MPI_COMM_WORLD);

		// prec_patterns[i].reinit(owned, owned, writable, MPI_COMM_WORLD, dofs);

		prec_patterns[i].reinit(owned, owned, writable, MPI_COMM_WORLD, dofs);

		if( lower ){
			DoFTools::make_sparsity_pattern(dof_handler, prec_patterns[i], cm_prec2, true , GlobalParams.MPI_Rank);
		}else {
			if(upper) {
				DoFTools::make_sparsity_pattern(dof_handler, prec_patterns[i], cm_prec1, true , GlobalParams.MPI_Rank);
			} else {
				DoFTools::make_sparsity_pattern(dof_handler, prec_patterns[i], cm_prec1, true , GlobalParams.MPI_Rank);
			}
		}

		// prec_patterns[i].reinit(locally_owned_dofs, MPI_COMM_WORLD, 0);
		// std::cout << GlobalParams.MPI_Rank << " has reached the end of loop " << i << std::endl;

		prec_patterns[i].compress();
	}

	reinit_all();
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::calculate_cell_weights () {
    cell = triangulation.begin_active();
	endc = triangulation.end();

	for (; cell!=endc; ++cell){
		if(cell->is_locally_owned()) {
            Tensor<2,3, std::complex<double>> tens;
            Point<3> pos = cell->center();
            // tens = get_Tensor(pos, false, true);
            tens = get_Tensor(pos,false, true);
            cell_weights(cell->active_cell_index()) = tens.norm();
            tens = get_Preconditioner_Tensor(pos,false, true, GlobalParams.MPI_Rank);
            cell_weights_prec_1(cell->active_cell_index()) = tens.norm();
            tens = get_Preconditioner_Tensor(pos,false, true, GlobalParams.MPI_Rank+1);
            cell_weights_prec_2(cell->active_cell_index()) = tens.norm();
        }
	}

	DataOut<3> data_out_cells;
	data_out_cells.attach_dof_handler (dof_handler);
	//data_out_real.attach_dof_handler(dof_handler_real);
	data_out_cells.add_data_vector (cell_weights, "Material_Tensor_Norm",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_cell_data );
	data_out_cells.add_data_vector (cell_weights_prec_1, "Material_Tensor_Prec_Low",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_cell_data );
	data_out_cells.add_data_vector (cell_weights_prec_2, "Material_Tensor_Prec_Up",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_cell_data );
	// data_out.add_data_vector(differences, "L2error");

	data_out_cells.build_patches ();

	std::ofstream outputvtu2 (solutionpath + "/cell-weights" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() +"-"+static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str()+".vtu");
	data_out_cells.write_vtu(outputvtu2);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_all () {
	// pout << "0-";
	reinit_rhs();
    
    reinit_cell_weights();
	// pout << "1-";
	reinit_solution();
	// pout << "2-";
	reinit_preconditioner();
	// pout << "3-";
	reinit_systemmatrix();
	// pout << "4";
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_for_rerun () {
	// pout << "0-";
	reinit_rhs();
	// pout << "1-";
	reinit_preconditioner_fast();
	// pout << "2-";
	reinit_systemmatrix();
	// pout << "3";
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
									   GlobalParams.MPI_Rank);
	sp.compress();
	system_matrix.reinit( sp);
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_solution() {

}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::reinit_storage() {

}

template <>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::reinit_systemmatrix() {
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
	EstimatedSolution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
	ErrorOfSolution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
}

template <>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::reinit_cell_weights() {
	cell_weights.reinit(triangulation.n_active_cells());
	cell_weights_prec_1.reinit(triangulation.n_active_cells());
	cell_weights_prec_2.reinit(triangulation.n_active_cells());
	calculate_cell_weights();
}

template<>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::Vector>::reinit_storage() {
	storage.reinit(locally_owned_dofs,  MPI_COMM_WORLD);
}

template<>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::reinit_preconditioner () {

	for(unsigned int i = 0; i < Layers -1; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		Preconditioner_Matrices[i].reinit(prec_patterns[i]);
	}
}

template<>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::reinit_preconditioner_fast () {
	for(int i = 0; i < (int)Layers-1; i++) {
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

				epsilon_pre1 = get_Preconditioner_Tensor(quadrature_points[q_index],false, true, subdomain_id-1);
				mu_prec1 = get_Preconditioner_Tensor(quadrature_points[q_index],true, false, subdomain_id-1);

				epsilon_pre2 = get_Preconditioner_Tensor(quadrature_points[q_index],false, true, subdomain_id);
				mu_prec2 = get_Preconditioner_Tensor(quadrature_points[q_index],true, false, subdomain_id);

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
			cm.distribute_local_to_global     (cell_matrix_real, cell_rhs, local_dof_indices,system_matrix, system_rhs, true);
			// pout << "P1 done"<<std::endl;

			if(GlobalParams.MPI_Rank != 0 ) {
				cm_prec1.distribute_local_to_global(cell_matrix_prec1, cell_rhs, local_dof_indices,Preconditioner_Matrices[GlobalParams.MPI_Rank-1], preconditioner_rhs, true);
			}
			if(GlobalParams.MPI_Rank != Layers-1) {
				cm_prec2.distribute_local_to_global(cell_matrix_prec2, cell_rhs, local_dof_indices, Preconditioner_Matrices[GlobalParams.MPI_Rank], preconditioner_rhs, true);
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

	if(prm.PRM_O_VerboseOutput && run_number == 0) {
		if(!is_stored) {
		pout << "Dofs per cell: " << dofs_per_cell << std::endl << "Quadrature Formula Size: " << n_q_points << std::endl;
		pout << "Dofs per face: " << fe.dofs_per_face << std::endl << "Dofs per line: " << fe.dofs_per_line << std::endl;
		}
	}

	if(!is_stored) pout << "Starting Assemblation process" << std::endl;

	assemble_part( );

	MPI_Barrier(MPI_COMM_WORLD);
	if(!is_stored)  pout<<"Assembling done. L2-Norm of RHS: "<< system_rhs.l2_norm()<<std::endl;

	system_matrix.compress(VectorOperation::add);
	system_rhs.compress(VectorOperation::add);

	for(unsigned int i = 0; i < Layers-1; i++) {
		// if(i == GlobalParams.MPI_Rank || i+1 == GlobalParams.MPI_Rank) {
			Preconditioner_Matrices[i].compress(VectorOperation::add);
		//}
	}

	cm.distribute(solution);
	cm.distribute(EstimatedSolution);
	cm.distribute(ErrorOfSolution);
	MPI_Barrier(MPI_COMM_WORLD);
	pout << "Distributing solution done." <<std::endl;
	estimate_solution();

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
							dtemp = dtemp / dtemp.norm();
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
	double layer_length = structure->Layer_Length();
	double sector_length = structure->Sector_Length();
	cell = dof_handler.begin_active();
	endc = dof_handler.end();
	IndexSet own (dof_handler.n_dofs());
	own.add_indices(locally_owned_dofs);
	sweepable.set_size(dof_handler.n_dofs());
	sweepable.add_indices(locally_owned_dofs);
	if(GlobalParams.MPI_Rank == 0 ){
		// own.add_indices(locally_owned_dofs);
	} else {
		own.add_indices(LowerDofs);
	}
	for (; cell!=endc; ++cell)
	{
		if(std::abs(cell->subdomain_id() - GlobalParams.MPI_Rank) < 2 ) {
			Point<3, double> cell_center =cell->center(true, false);
			/**
			if( PML_in_X(cell_center) || PML_in_Y(cell_center) || PML_in_Z(cell_center)) {
				std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);
				cell->get_dof_indices(local_dof_indices);
				IndexSet localdofs(dof_handler.n_dofs());
				for(unsigned int k =0 ; k< local_dof_indices.size(); k++) {
					localdofs.add_index(local_dof_indices[k]);
				}
				sweepable.subtract_set(localdofs);
			}
			**/
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
							if(own.is_element(local_dof_indices[k])) {
								cm_prec1.add_line(local_dof_indices[k]);
								cm_prec1.set_inhomogeneity(local_dof_indices[k], 0.0 );
								cm_prec2.add_line(local_dof_indices[k]);
								cm_prec2.set_inhomogeneity(local_dof_indices[k], 0.0 );
							}
						}
					}
				}

				// Set y-boundary values
				if ( std::abs( center[1] - GlobalParams.PRM_M_R_YLength/2.0) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						for(unsigned int k = 0; k < 2; k++) {
							if(own.is_element(local_dof_indices[k])) {
								cm_prec1.add_line(local_dof_indices[k]);
								cm_prec1.set_inhomogeneity(local_dof_indices[k], 0.0 );
								cm_prec2.add_line(local_dof_indices[k]);
								cm_prec2.set_inhomogeneity(local_dof_indices[k], 0.0 );
							}
						}
					}
				}

				if ( std::abs( center[2] - GlobalParams.PRM_M_R_YLength/2.0 - (1 + GlobalParams.MPI_Rank)*layer_length) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						for(unsigned int k = 0; k < 2; k++) {
							if(own.is_element(local_dof_indices[k])) {
								cm_prec1.add_line(local_dof_indices[k]);
								cm_prec1.set_inhomogeneity(local_dof_indices[k], 0.0 );
							}
						}
					}
				}


				if( GlobalParams.MPI_Rank == 0 && std::abs(center[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						if((cell->face(i))->line(j)->at_boundary()) {
							((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
							Tensor<1,3,double> ptemp = ((cell->face(i))->line(j))->center(true, false);
							Point<3, double> p (ptemp[0], ptemp[1], ptemp[2]);
							Tensor<1,3,double> dtemp = ((cell->face(i))->line(j))->vertex(0) - ((cell->face(i))->line(j))->vertex(1);
							dtemp = dtemp / dtemp.norm();
							Point<3, double> direction (dtemp[0], dtemp[1], dtemp[2]);

							double result = TEMode00(p,0);
							if(PML_in_X(p) || PML_in_Y(p)) result = 0.0;
							if(locally_owned_dofs.is_element(local_dof_indices[0])) {
								cm_prec2.add_line(local_dof_indices[0]);
								cm_prec2.set_inhomogeneity(local_dof_indices[0], direction[0] * result );
							}
							if(locally_owned_dofs.is_element(local_dof_indices[1])) {
								cm_prec2.add_line(local_dof_indices[1]);
								cm_prec2.set_inhomogeneity(local_dof_indices[1], 0.0);
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

	solver_control.log_frequency(1);
	result_file.open((solutionpath + "/solution_of_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() + ".dat").c_str());

	if(GlobalParams.PRM_S_Solver == "GMRES") {
		int mindof = locally_owned_dofs.nth_index_in_set(0);
		for(unsigned int i = 0; i < locally_owned_dofs.n_elements(); i++ ) {
			solution[mindof + i] = EstimatedSolution[mindof + i];
		}


		dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control , dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData( GlobalParams.PRM_S_GMRESSteps) );

		// std::cout << GlobalParams.MPI_Rank << " prep dofs." <<std::endl;

		int above = 0;
		if (GlobalParams.MPI_Rank != GlobalParams.MPI_Size -1 ) {
			above = locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank+1].n_elements();
		}

		int own = locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank].n_elements();

		int t_upper = 0;
		if (GlobalParams.MPI_Rank != GlobalParams.MPI_Size-1) {
			t_upper = locally_relevant_dofs_all_processors[GlobalParams.MPI_Rank +1].n_elements();
		}
		
		// PreconditionerSweeping sweep( locally_owned_dofs.n_elements(), below, dof_handler.max_couplings_between_dofs(), locally_owned_dofs, t_upper, & cm);
		PreconditionerSweeping sweep( locally_owned_dofs.n_elements(), above, dof_handler.max_couplings_between_dofs(), sweepable, locally_owned_dofs, t_upper, & cm);



		unsigned int dofs_below = mindof;
		unsigned int total_dofs_local = UpperDofs.n_elements();

		if(GlobalParams.MPI_Rank == GlobalParams.MPI_Size-1 ){
			for (unsigned int current_row = mindof; current_row < dof_handler.n_dofs(); current_row++ ) {
				for(TrilinosWrappers::SparseMatrix::iterator row = system_matrix.begin(current_row); row != system_matrix.end(current_row); row++) {
					if(row->column() >= mindof){
						sweep.matrix.set(current_row - mindof, row->column() -mindof, row->value());
					}
					
				}
			}
		} else {
			for (unsigned int current_row = 0; current_row < UpperDofs.n_elements(); current_row++ ) {
				for(TrilinosWrappers::SparseMatrix::iterator row = Preconditioner_Matrices[GlobalParams.MPI_Rank].begin(mindof + current_row); row != Preconditioner_Matrices[GlobalParams.MPI_Rank].end(current_row + mindof); row++) {
					if(row->column() >= dofs_below && row->column() < dofs_below + total_dofs_local){
						sweep.matrix.set(current_row, row->column() - dofs_below, row->value());
					}
				}
			}
			unsigned int lower_entries, upper_entries;
			lower_entries = 0;
			upper_entries = 0;
			for (unsigned int current_row = 0; current_row < own; current_row++ ) {
				for(TrilinosWrappers::SparseMatrix::iterator row = system_matrix.begin(dofs_below + current_row); row != system_matrix.end(dofs_below + current_row); row++) {
					if(row->column() >= own + dofs_below && row->column() - dofs_below < own  + above){
						sweep.prec_matrix_lower.set(current_row , row->column() - dofs_below-own, row->value());
						sweep.prec_matrix_upper.set(row->column() - dofs_below -own , current_row ,  row->value());
						lower_entries ++;
					}
				}
			}


			std::cout << "Copy in proc " << GlobalParams.MPI_Rank << " gave " << lower_entries << " lower entries" << std::endl;
		}

		sweep.matrix.compress(VectorOperation::insert);

		sweep.prec_matrix_lower.compress(VectorOperation::insert);
		sweep.prec_matrix_upper.compress(VectorOperation::insert);

		if(GlobalParams.MPI_Rank == 0) {
			sweep.Prepare(solution);
		}

		MPI_Barrier(MPI_COMM_WORLD);


		pout << "All preconditioner matrices built. Solving..." <<std::endl;



		solver.solve(system_matrix,solution, system_rhs, sweep);

		pout << "Done." << std::endl;

		pout << "Norm of the solution: " << solution.l2_norm() << std::endl;
	}

	if(GlobalParams.PRM_S_Solver == "UMFPACK") {
		SolverControl sc2(2,false,false);
		TrilinosWrappers::SolverDirect temp_s(sc2, TrilinosWrappers::SolverDirect::AdditionalData(false, GlobalParams.PRM_S_Preconditioner));
		temp_s.solve(system_matrix, solution, system_rhs);
	}

 
	// cm.distribute(solution);
	// cm.distribute(EstimatedSolution);
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

}

template<>
void Waveguide<TrilinosWrappers::SparseMatrix, TrilinosWrappers::MPI::Vector>::output_results ( bool details )
{

	for(unsigned int i = 0; i < locally_owned_dofs.n_elements(); i++) {
		unsigned int index = locally_owned_dofs.nth_index_in_set(i);
		ErrorOfSolution[index] = solution[index] - EstimatedSolution[index];
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// ErrorOfSolution.compress(VectorOperation::insert);

	//solution.compress(VectorOperation::unknown);



	TrilinosWrappers::MPI::Vector solution_output(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
	solution_output = solution;

	TrilinosWrappers::MPI::Vector estimate_output(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
	estimate_output = EstimatedSolution;

	TrilinosWrappers::MPI::Vector error_output(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
	error_output = ErrorOfSolution;


	std::cout << GlobalParams.MPI_Rank << ": " <<locally_owned_dofs.n_elements()<< "," <<locally_owned_dofs.nth_index_in_set(0) << "," << locally_owned_dofs.nth_index_in_set(locally_owned_dofs.n_elements()-1) <<std::endl;
	// evaluate_overall();
	if(true) {
		DataOut<3> data_out;

		data_out.attach_dof_handler (dof_handler);
		data_out.add_data_vector (solution_output, "Solution",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_dof_data);
		data_out.add_data_vector (error_output, "Error_Of_Estimated_Solution",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_dof_data);
		data_out.add_data_vector (estimate_output, "Estimated_Solution",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_dof_data);
		// data_out.add_data_vector(differences, "L2error");

		data_out.build_patches ();

		std::ofstream outputvtk (solutionpath + "/solution-run" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() + "-P" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() +".vtk");
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
		if ( false ) {
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
		}


	}
}

template<typename MatrixType, typename VectorType>
void Waveguide<MatrixType, VectorType>::run ()
{

	timer.enter_subsection ("Setup Mesh");
	make_grid ();
	timer.leave_subsection();

	timer.enter_subsection ("Setup FEM");
	setup_system ();
	timer.leave_subsection();

	timer.enter_subsection ("Reset");
	timer.leave_subsection();

	timer.enter_subsection ("Assemble");
	assemble_system ();
	timer.leave_subsection();

	timer.enter_subsection ("Solve");
	solve ();
	timer.leave_subsection();

	timer.enter_subsection ("Evaluate");
	evaluate ();
	timer.leave_subsection();

	timer.print_summary();
	timer.reset();
    
    output_results(false);
    
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
}

template<typename MatrixType, typename VectorType >
void Waveguide<MatrixType, VectorType>::rerun ()
{
	timer.enter_subsection ("Setup Mesh");

	timer.leave_subsection();

	timer.enter_subsection ("Setup FEM");
	structure->Print();

	timer.leave_subsection();
	pout << "Reinit for rerun..." ;
	pout.get_stream().flush();
	timer.enter_subsection ("Reset");
	reinit_for_rerun();
	timer.leave_subsection();

	pout << " Assemble for rerun... " ;
	pout.get_stream().flush();
	timer.enter_subsection ("Assemble");
	assemble_system ();
	timer.leave_subsection();

	pout << " Solve for rerun..." << std::endl;
	timer.enter_subsection ("Solve");
	solve ();
	timer.leave_subsection();

	pout << "Evaluate for rerun." << std::endl;
	timer.enter_subsection ("Evaluate");
	evaluate ();
	timer.leave_subsection();

	timer.print_summary();
	timer.reset();

	run_number++;

}

#endif
