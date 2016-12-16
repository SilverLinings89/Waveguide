#ifndef WaveguideCppFlag
#define WaveguideCppFlag


#include <sys/time.h>
#include "Waveguide.h"
#include "WaveguideStructure.h"
#include "SolutionWeight.h"
#include "../Helpers/staticfunctions.cpp"
#include "../Helpers/ExactSolution.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "PreconditionerSweeping.cpp"
#include <deal.II/base/timer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/std_cxx11/bind.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/solver.h>
#include <deal.II/numerics/data_out_dof_data.h>

using namespace dealii;

Waveguide::Waveguide (MPI_Comm in_mpi_comm, MeshGenerator * in_mg, SpaceTransformation * in_st):
    fe(FE_Nedelec<3> (0), 2),
    triangulation (mpi_comm, parallel::distributed::Triangulation<3>::MeshSmoothing(Triangulation<3>::none ), parallel::distributed::Triangulation<3>::Settings::no_automatic_repartitioning),
    dof_handler (triangulation),
    run_number(0),
    condition_file_counter(0),
    eigenvalue_file_counter(0),
    Sectors(GlobalParams.M_W_Sectors),
    Layers(GlobalParams.NumberProcesses),
    Dofs_Below_Subdomain(Layers),
    Block_Sizes(Layers),
    temporary_pattern_preped(false),
    real(0),
    imag(3),
    solver_control (GlobalParams.So_TotalSteps, GlobalParams.So_Precision, (GlobalParams.MPI_Rank == 0), true),
    pout(std::cout, GlobalParams.MPI_Rank==0),
    timer(mpi_comm, pout, TimerOutput::OutputFrequency::summary, TimerOutput::wall_times),
    is_stored(false),
    even(Utilities::MPI::this_mpi_process(in_mpi_comm)%2 == 0),
    rank(Utilities::MPI::this_mpi_process(in_mpi_comm))
{
  mg = in_mg;
  st = in_st;
  mpi_comm = in_mpi_comm;
  int i = 0;
  bool dir_exists = true;
  while(dir_exists) {
    std::stringstream out;
    out << "Solutions/run";
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
  i = Utilities::MPI::max(i, mpi_comm);
  std::stringstream out;
  out << "Solutions/run";

  // TODO check if this directory is really available for all processes and throw an error otherwise.

  out << i;
  solutionpath = out.str();
  Dofs_Below_Subdomain[Layers];
  mkdir(solutionpath.c_str(), ACCESSPERMS);
  pout << "Will write solutions to " << solutionpath << std::endl;

  // Copy Parameter file to the output directory in processor 0. This should be replaced with an output generator eventually.
  if(GlobalParams.MPI_Rank == 0) {
    std::ifstream source("Parameters/Parameters.xml", std::ios::binary);
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
  start_solver_milis = 0;

}

Waveguide::~Waveguide() {
  delete Preconditioner_Matrices;
}

std::complex<double> Waveguide::evaluate_for_Position(double x, double y, double z ) {
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

std::complex<double> Waveguide::gauss_product_2D_sphere(double z, int n, double R, double Xc, double Yc)
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

double Waveguide::evaluate_for_z(double z) {
	double r = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0;

	std::complex<double> res = gauss_product_2D_sphere(z,10,r,0,0);
	return std::sqrt(std::norm(res));
}

void Waveguide::evaluate() {
	pout << "Starting Evaluation" << std::endl;
	double z_for_evaluation = (double)(0.5+GlobalParams.MPI_Rank)*structure->Layer_Length() - GlobalParams.PRM_M_R_ZLength/2.0 ;
	double local_value = evaluate_for_z(z_for_evaluation) ;
	MPI_Allgather( & local_value, 1, MPI_DOUBLE, qualities, 1, MPI_DOUBLE, mpi_comm);
	pout << "Done Gathering Qualities!"<< std::endl;
}

double Waveguide::evaluate_overall () {
	std::vector <double> qualities(Layers);
	double lower = 0.0;
	double upper = 0.0;
	if(GlobalParams.evaluate_in) {
		// pout << "Evaluation for input side by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << -GlobalParams.PRM_M_R_ZLength / 2.0 << std::endl;
		lower = evaluate_for_z(-GlobalParams.M_R_ZLength / 2.0 + 0.0000001);
	}
	if(GlobalParams.evaluate_out) {
		// pout << "Evaluation for output side by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << GlobalParams.PRM_M_R_ZLength / 2.0 << std::endl;
		upper = evaluate_for_z(GlobalParams.M_R_ZLength / 2.0 -  0.0000001);
	}
	lower = Utilities::MPI::sum(lower, mpi_comm);
	upper = Utilities::MPI::sum(upper, mpi_comm);

	for(unsigned int i = 0; i< Layers; i++) {
		double contrib = 0.0;
		if(i == GlobalParams.MPI_Rank) {
			 std::cout << "Evaluation of contribution by Task " << GlobalParams.MPI_Rank <<" with lower " << GlobalParams.z_min << " and upper " << GlobalParams.z_max << " at " << GlobalParams.z_evaluate<<":" ;
			 contrib = evaluate_for_z(GlobalParams.z_evaluate);
			 std::cout << contrib << std::endl;
		}
		qualities[i] = Utilities::MPI::max(contrib, mpi_comm);
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

void Waveguide::estimate_solution() {
	MPI_Barrier(mpi_comm);
	pout << "Starting solution estimation..." << std::endl;
	DoFHandler<3>::active_cell_iterator cell, endc;
	pout << "Lambda: " << GlobalParams.M_W_Lambda << std::endl;
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
					if( std::abs(ptemp[2] + GlobalParams.M_R_ZLength/2.0 ) > 0.0001 ){
						Point<3, double> p (ptemp[0], ptemp[1], ptemp[2]);
						Tensor<1,3,double> dtemp = ((cell->face(i))->line(j))->vertex(0) - ((cell->face(i))->line(j))->vertex(1);
						dtemp = dtemp / dtemp.norm();
						Point<3, double> direction (dtemp[0], dtemp[1], dtemp[2]);


						//double phi = (ptemp[2] + GlobalParams.PRM_M_R_ZLength/2.0 ) *2 * GlobalParams.PRM_C_PI / (GlobalParams.PRM_M_W_Lambda / GlobalParams.PRM_M_W_EpsilonIn);
						double phi = (ptemp[2] + GlobalParams.M_R_ZLength/2.0 ) * 2* GlobalParams.C_Pi / (GlobalParams.M_W_Lambda / std::sqrt(GlobalParams.M_W_epsilonin));
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
	MPI_Barrier(mpi_comm);
	EstimatedSolution.compress(VectorOperation::insert);
}



Tensor<2,3, std::complex<double>> Waveguide::Conjugate_Tensor(Tensor<2,3, std::complex<double>> input) {
	Tensor<2,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		for(int j = 0; j<3; j++){
			ret[i][j].real(input[i][j].real());
			ret[i][j].imag( - input[i][j].imag());
		}
	}
	return ret;
}

Tensor<1,3, std::complex<double>> Waveguide::Conjugate_Vector(Tensor<1,3, std::complex<double>> input) {
	Tensor<1,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		ret[i].real(input[i].real());
		ret[i].imag( - input[i].imag());

	}
	return ret;
}

void Waveguide::make_grid ()
{
  mg->prepare_triangulation(& triangulation);
}

void Waveguide::Compute_Dof_Numbers() {
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



}

void Waveguide::setup_system ()
{

	dof_handler.distribute_dofs (fe);

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

	system_pattern.reinit(locally_owned_dofs, locally_owned_dofs, locally_relevant_dofs, mpi_comm);
	pout << "Done" << std::endl;


//	dynamic_preconditioner_pattern_even.reinit(n_neighboring, n_neighboring);
	preconditioner_pattern.reinit(locally_owned_dofs, locally_owned_dofs, locally_relevant_dofs, mpi_comm);

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

	//SparsityTools::distribute_sparsity_pattern(dynamic_system_pattern,n_neighboring, mpi_comm, locally_owned_dofs);
	//SparsityTools::distribute_sparsity_pattern(dynamic_preconditioner_pattern_even,n_neighboring, mpi_comm, locally_owned_dofs);

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

	MPI_Allgather(& text_local_length, 1, MPI_INT, all_lens, 1, MPI_INT, mpi_comm);

	int totlen = all_lens[mpi_size-1];
	displs[0] = 0;
	for (int i=0; i<mpi_size-1; i++) {
		displs[i+1] = displs[i] + all_lens[i];
		totlen += all_lens[i];
	}
	char * all_names = (char *)malloc( totlen );
	if (!all_names) MPI_Abort( mpi_comm, 1 );

	MPI_Allgatherv( text_local_set, text_local_length, MPI_CHAR, all_names, all_lens, displs, MPI_CHAR,	mpi_comm );

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
	// MPI_Comm_split(mpi_comm, GlobalParams.MPI_Rank/2, GlobalParams.MPI_Rank, &comm_even );
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


		MPI_Barrier(mpi_comm);

		// prec_patterns[i].reinit(owned, owned, writable, mpi_comm, dofs);

		prec_patterns[i].reinit(owned, owned, writable, mpi_comm, dofs);

		if( lower ){
			DoFTools::make_sparsity_pattern(dof_handler, prec_patterns[i], cm_prec2, true , GlobalParams.MPI_Rank);
		}else {
			if(upper) {
				DoFTools::make_sparsity_pattern(dof_handler, prec_patterns[i], cm_prec1, true , GlobalParams.MPI_Rank);
			} else {
				DoFTools::make_sparsity_pattern(dof_handler, prec_patterns[i], cm_prec1, true , GlobalParams.MPI_Rank);
			}
		}

		// prec_patterns[i].reinit(locally_owned_dofs, mpi_comm, 0);
		// std::cout << GlobalParams.MPI_Rank << " has reached the end of loop " << i << std::endl;

		prec_patterns[i].compress();
	}

	reinit_all();
}

void Waveguide::calculate_cell_weights () {
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

void Waveguide::reinit_all () {
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

void Waveguide::reinit_for_rerun () {
	// pout << "0-";
	reinit_rhs();
	// pout << "1-";
	reinit_preconditioner_fast();
	// pout << "2-";
	reinit_systemmatrix();
	// pout << "3";
}

void Waveguide::reinit_systemmatrix() {
  /**
   * What has to be done?
   * 1. Compute which indices should be written (and owned) for the system matrix.
   * 2. Compute which indices block structures of all 3 matrices.
   * 3. Compute which indices can be written to in which process.
   *
   */

  int prec_even_block_count = Utilities::MPI::n_mpi_processes(mpi_comm) /2;
  if(Utilities::MPI::n_mpi_processes(mpi_comm)%2 == 1) {
    prec_even_block_count++;
  }

  int prec_odd_block_count = Utilities::MPI::n_mpi_processes(mpi_comm) /2 +1;

  std::vector<IndexSet> i_sys_owned(Layers);

  for(int i = 0; i < Layers; i++) {
    int size = Block_Sizes[i];
    bool local = (i == rank);
    IndexSet temp(size);
    if(local) {
      temp.add_range(0,size);
    }
    i_sys_owned[i] = temp;
  }

  std::vector<IndexSet> i_prec_even_owned_row(Layers);
  std::vector<IndexSet> i_prec_even_owned_col(Layers);
  std::vector<IndexSet> i_prec_even_writable(Layers);
  std::vector<IndexSet> i_prec_odd_owned_row(Layers);
  std::vector<IndexSet> i_prec_odd_owned_col(Layers);
  std::vector<IndexSet> i_prec_odd_writable(Layers);


  for(int i = 0; i < Layers; i++) {
    int size = Block_Sizes[i];
    bool even_row_owned = false;
    bool even_row_writable = false;
    bool even_col_owned = false;
    if(even){
      if (i == rank || i == rank +1) {
        even_row_owned = true;
        even_row_writable = true;
        even_col_owned = true;
      }
    } else {
      if (i == rank || i == rank -1) {
        even_row_writable = true;
      }
    }
    IndexSet ero(size);
    IndexSet erw(size);
    IndexSet eco(size);
    if(even_row_owned){
      ero.add_range(0,size);
    }
    if(even_row_writable){
      erw.add_range(0,size);
    }
    if(even_col_owned){
      eco.add_range(0,size);
    }

    i_prec_even_owned_row[i] = ero;
    i_prec_even_owned_col[i] = eco;
    i_prec_even_writable[i] = erw;
  }

  for(int i = 0; i < Layers; i++) {
    int size = Block_Sizes[i];
    bool odd_row_owned = false;
    bool odd_row_writable = false;
    bool odd_col_owned = false;
    if(!even){
      if (i == rank || i == rank +1) {
        odd_row_owned = true;
        odd_row_writable = true;
        odd_col_owned = true;
      }
    } else {
      if ( i == rank || i == rank -1) {
        odd_row_writable = true;
      }
    }
    IndexSet oro(size);
    IndexSet orw(size);
    IndexSet oco(size);
    if(odd_row_owned){
      oro.add_range(0,size);
    }
    if(odd_row_writable){
      orw.add_range(0,size);
    }
    if(odd_col_owned){
      oco.add_range(0,size);
    }

    i_prec_odd_owned_row[i] = oro;
    i_prec_odd_owned_col[i] = oco;
    i_prec_odd_writable[i] = orw;
  }

  TrilinosWrappers::BlockSparsityPattern epsp(i_prec_even_owned_row, i_prec_even_owned_col, i_prec_even_writable, mpi_comm);
  TrilinosWrappers::BlockSparsityPattern opsp(i_prec_odd_owned_row, i_prec_odd_owned_col, i_prec_odd_writable, mpi_comm);
  TrilinosWrappers::BlockSparsityPattern sp(i_sys_owned, mpi_comm);

  epsp.collect_sizes();
  opsp.collect_sizes();
  sp.collect_sizes();

  DoFTools::make_sparsity_pattern (dof_handler, epsp,
	                                   cm, false,
									   GlobalParams.MPI_Rank);
	epsp.compress();
	DoFTools::make_sparsity_pattern (dof_handler, opsp,
	                                     cm, false,
	                     GlobalParams.MPI_Rank);
	opsp.compress();

  DoFTools::make_sparsity_pattern (dof_handler, sp,
                                       cm, false,
                       GlobalParams.MPI_Rank);
  sp.compress();

  // const TrilinosWrappers::BlockSparsityPattern  cepsp(epsp);

  // const std::vector<IndexSet> ciso(i_sys_owned);
	system_matrix.reinit(sp);
	prec_matrix_even.reinit(epsp);
	prec_matrix_odd.reinit(opsp);
}

void Waveguide::reinit_rhs () {
	// std::cout << "Reinit rhs for p " << GlobalParams.MPI_Rank << std::endl;

	system_rhs.reinit(locally_owned_dofs, mpi_comm);

	preconditioner_rhs.reinit(dof_handler.n_dofs());

}

void Waveguide::reinit_solution() {
	solution.reinit(locally_owned_dofs, mpi_comm);
	EstimatedSolution.reinit(locally_owned_dofs, mpi_comm);
	ErrorOfSolution.reinit(locally_owned_dofs, mpi_comm);
}

void Waveguide::reinit_cell_weights() {
	cell_weights.reinit(triangulation.n_active_cells());
	cell_weights_prec_1.reinit(triangulation.n_active_cells());
	cell_weights_prec_2.reinit(triangulation.n_active_cells());
	calculate_cell_weights();
}

void Waveguide::reinit_storage() {
	storage.reinit(locally_owned_dofs,  mpi_comm);
}

void Waveguide::reinit_preconditioner () {

	for(unsigned int i = 0; i < Layers -1; i++) {
		MPI_Barrier(mpi_comm);
		Preconditioner_Matrices[i].reinit(prec_patterns[i]);
	}
}

void Waveguide::reinit_preconditioner_fast () {
	for(int i = 0; i < (int)Layers-1; i++) {
		Preconditioner_Matrices[i].reinit(prec_patterns[i]);
	}
}

void Waveguide::assemble_system ()
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

	if(!is_stored) pout << "Starting Assemblation process" << std::endl;

	FullMatrix<double>  cell_matrix_real (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  cell_matrix_prec1 (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  cell_matrix_prec2 (dofs_per_cell, dofs_per_cell);

  double e_temp = 1.0;
  if(!GlobalParams.C_AllOne){
    e_temp *= GlobalParams.C_Epsilon;
  }
  double mu_temp = 1.0;
  if(!GlobalParams.C_AllOne){
    mu_temp *= GlobalParams.C_Mu;
  }

  const double eps_in= GlobalParams.M_W_epsilonin * e_temp;
  const double eps_out= GlobalParams.M_W_epsilonout * e_temp;
  const double mu_zero = mu_temp;

  Vector<double>    cell_rhs (dofs_per_cell);
  cell_rhs = 0;
  Tensor<2,3, std::complex<double>>     transformation, epsilon, epsilon_pre1, epsilon_pre2, mu, mu_prec1, mu_prec2;
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
        transformation = st->get_Tensor(quadrature_points[q_index]);

        if( mg->math_coordinate_in_waveguide(quadrature_points[q_index])) {
          epsilon = transformation * eps_in;
        } else {
          epsilon = transformation * eps_out;
        }

        mu = invert(transformation) * mu_zero;

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

            std::complex<double> x = (mu * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.C_omega*GlobalParams.C_omega;
            cell_matrix_real[i][j] += x.real();


            std::complex<double> pre1 = (mu_prec1 * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon_pre1 * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.C_omega*GlobalParams.C_omega;
            cell_matrix_prec1[i][j] += pre1.real();

            std::complex<double> pre2 = (mu_prec2 * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon_pre2 * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.C_omega*GlobalParams.C_omega;
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


	MPI_Barrier(mpi_comm);
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
	MPI_Barrier(mpi_comm);
	pout << "Distributing solution done." <<std::endl;
	estimate_solution();

}

void Waveguide::MakeBoundaryConditions (){
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

void Waveguide::MakePreconditionerBoundaryConditions (  ){
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

SolverControl::State  Waveguide::check_iteration_state (const unsigned int iteration, const double check_value, const dealii::TrilinosWrappers::MPI::Vector & ){
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

void Waveguide::solve () {

	solver_control.log_frequency(1);
	result_file.open((solutionpath + "/solution_of_run_" + static_cast<std::ostringstream*>( &(std::ostringstream() << run_number) )->str() + ".dat").c_str());

	if(GlobalParams.PRM_S_Solver == "GMRES") {
		int mindof = locally_owned_dofs.nth_index_in_set(0);
		for(unsigned int i = 0; i < locally_owned_dofs.n_elements(); i++ ) {
			solution[mindof + i] = EstimatedSolution[mindof + i];
		}


		// dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control , dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData( GlobalParams.PRM_S_GMRESSteps) );

		dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control , dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData(30) );
		
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

		MPI_Barrier(mpi_comm);


		pout << "All preconditioner matrices built. Solving..." <<std::endl;
        
        struct timeval tp;
        gettimeofday(&tp, NULL);
        long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
        start_solver_milis = tp.tv_sec * 1000 + tp.tv_usec / 1000;    
        
		solver.connect(std_cxx11::bind (&Waveguide::residual_tracker,
                                   this,
                                   std_cxx11::_1,
                                   std_cxx11::_2,
                                   std_cxx11::_3));
        
	 solver.solve(system_matrix,solution, system_rhs, sweep);

	 /**
	 PreconditionJacobi<TrilinosWrappers::SparseMatrix> precondition;
        precondition.initialize (system_matrix, .6);
        solver.solve(system_matrix,solution, system_rhs, precondition);
    **/

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

void Waveguide::store() {
	reinit_storage();
	// storage.reinit(dof_handler.n_dofs());
	storage = solution;
	is_stored = true;
}


void Waveguide::output_results ( bool details )
{

	for(unsigned int i = 0; i < locally_owned_dofs.n_elements(); i++) {
		unsigned int index = locally_owned_dofs.nth_index_in_set(i);
		ErrorOfSolution[index] = solution[index] - EstimatedSolution[index];
	}

	MPI_Barrier(mpi_comm);

	// ErrorOfSolution.compress(VectorOperation::insert);

	//solution.compress(VectorOperation::unknown);



	TrilinosWrappers::MPI::Vector solution_output(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
	solution_output = solution;

	TrilinosWrappers::MPI::Vector estimate_output(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
	estimate_output = EstimatedSolution;

	TrilinosWrappers::MPI::Vector error_output(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
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

void Waveguide::run ()
{

	timer.enter_subsection ("Setup Mesh");
	make_grid ();
	timer.leave_subsection();

	Compute_Dof_Numbers();

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


void Waveguide::print_eigenvalues(const std::vector<std::complex<double>> &input) {
	for (unsigned int i = 0; i < input.size(); i++){
		eigenvalue_file << input.at(i).real() << "\t" << input.at(i).imag() << std::endl;
	}
	eigenvalue_file << std::endl;
}


void Waveguide::print_condition(double condition) {
	condition_file << condition << std::endl;
}


void Waveguide::reset_changes ()
{
	reinit_all();
}


void Waveguide::rerun ()
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


SolverControl::State Waveguide::residual_tracker(unsigned int Iteration, double residual, dealii::TrilinosWrappers::MPI::Vector vec) {
    
    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
        
    result_file << "" << Iteration << "\t" << residual << "\t" << ms <<std::endl;
    return SolverControl::success;
}

#endif
