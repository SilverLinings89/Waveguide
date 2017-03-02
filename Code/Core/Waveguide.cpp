#ifndef WaveguideCppFlag
#define WaveguideCppFlag


#include <sys/time.h>
#include "../Core/Waveguide.h"
#include "SolutionWeight.h"
#include "../Helpers/staticfunctions.cpp"
#include "../Helpers/ExactSolution.cpp"
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
#include <deal.II/lac/block_matrix_array.h>

using namespace dealii;

int steps = 0;

inline bool PML_blocked_self() {
  return (int)GlobalParams.MPI_Rank >= GlobalParams.NumberProcesses - GlobalParams.M_BC_Zplus;
}

Waveguide::Waveguide (MPI_Comm in_mpi_comm, MeshGenerator * in_mg, SpaceTransformation * in_st, std::string path_part):
    fe(FE_Nedelec<3> (0), 2),
    triangulation (in_mpi_comm,parallel::distributed::Triangulation<3>::MeshSmoothing(parallel::distributed::Triangulation<3>::none ), parallel::distributed::Triangulation<3>::Settings::no_automatic_repartitioning),
    even(Utilities::MPI::this_mpi_process(in_mpi_comm)%2 == 0),
    rank(Utilities::MPI::this_mpi_process(in_mpi_comm)),
    real(0),
    imag(3),
    solver_control (GlobalParams.So_TotalSteps, GlobalParams.So_Precision, false, true),
    dof_handler (triangulation),
    run_number(0),
    condition_file_counter(0),
    eigenvalue_file_counter(0),
    Layers(GlobalParams.NumberProcesses),
    Dofs_Below_Subdomain(Layers),
    Block_Sizes(Layers),
    pout(std::cout, rank==0),
    is_stored(false),
    timer(in_mpi_comm, pout, TimerOutput::OutputFrequency::summary, TimerOutput::wall_times),
    Sectors(GlobalParams.M_W_Sectors),
    minimum_local_z(2.0 * GlobalParams.M_R_ZLength),
    maximum_local_z(- 2.0 * GlobalParams.M_R_ZLength)
{
  mg = in_mg;
  st = in_st;
  mpi_comm = in_mpi_comm;
  std::cout << "(" << path_part << "): " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " as " << rank <<std::endl;
  path_prefix = path_part;

  is_stored = false;
  solver_control.log_frequency(10);
  const int number = Layers -1;
  qualities = new double[number];
  execute_recomputation = false;
  mkdir((solutionpath + "/" +path_prefix).c_str(), ACCESSPERMS);
  deallog << "This is process " << GlobalParams.MPI_Rank << " as " << rank <<std::endl;
}

Waveguide::~Waveguide() {
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

void Waveguide::estimate_solution() {
	MPI_Barrier(mpi_comm);
	deallog.push("estimate_solution");
	deallog << "Starting solution estimation..." << std::endl;
	DoFHandler<3>::active_cell_iterator cell, endc;
	deallog << "Lambda: " << GlobalParams.M_W_Lambda << std::endl;
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
						double result_real = -TEMode00(p,0) * std::cos(phi) ;
						double result_imag = - TEMode00(p,0) * std::sin(phi) ;
						if(st->PML_in_X(p) || st->PML_in_Y(p)) result_real = 0.0;
						if(st->PML_in_X(p) || st->PML_in_Y(p)) result_imag = 0.0;

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
	deallog<<"Done."<<std::endl;
	deallog.pop();
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
  dof_handler.distribute_dofs (fe);
  // std::cout << "Waveguide current active cells:" << triangulation.n_active_cells() << std::endl;

  parallel::distributed::Triangulation<3>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
  minimum_local_z = GlobalParams.M_R_ZLength;
  maximum_local_z = - GlobalParams.M_R_ZLength;
  for (; cell!=endc; ++cell){
    if(cell->is_locally_owned()) {
        for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
          double temp = (cell->face(i)->center())[2];
          if(temp < minimum_local_z) minimum_local_z = temp;
          if(temp > maximum_local_z) maximum_local_z = temp;
        }
    }
  }

  deallog << "Process " << GlobalParams.MPI_Rank << " as " << rank << ". The local range is ["<< minimum_local_z<<","<<maximum_local_z<<"]"<< std::endl;
}

void Waveguide::Compute_Dof_Numbers() {
	std::vector<types::global_dof_index> dof_indices (fe.dofs_per_face);
	std::vector<types::global_dof_index> DofsPerSubdomain(Layers);
	std::vector<int> InternalBoundaryDofs(Layers);

	DofsPerSubdomain = dof_handler.n_locally_owned_dofs_per_processor();
	for( unsigned int i = 0; i < Layers; i++) {
		Block_Sizes[i] = DofsPerSubdomain[i];
	}
	deallog << "Layers: " << Layers <<std::endl;
	for (unsigned int i = 0; i < Layers; i++) {
	  deallog << Block_Sizes[i]<< std::endl;
	}


	Dofs_Below_Subdomain[0] = 0;

	for(unsigned int i = 1; i  < Layers; i++) {
		Dofs_Below_Subdomain[i] = Dofs_Below_Subdomain[i-1] + Block_Sizes[i-1];
	}

	for(unsigned int i = 0; i < Layers; i++) {
		IndexSet temp (dof_handler.n_dofs());
		temp.clear();
		deallog << "Adding Block "<< i +1 << " from " << Dofs_Below_Subdomain[i] << " to " << Dofs_Below_Subdomain[i]+ Block_Sizes[i] -1<<std::endl;
		temp.add_range(Dofs_Below_Subdomain[i],Dofs_Below_Subdomain[i]+Block_Sizes[i] );
		set.push_back(temp);
	}



}

IndexSet Waveguide::combine_indexes(IndexSet lower, IndexSet upper) const {
  IndexSet ret(lower.size() + upper.size());
  ret.add_indices(lower);
  ret.add_indices(upper, lower.size());
  return ret;
}

void Waveguide::setup_system ()
{
  deallog.push("setup_system");
  deallog << "Assembling IndexSets" <<std::endl;
  locally_owned_dofs = dof_handler.locally_owned_dofs ();
	DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);
	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
	std::vector<unsigned int> n_neighboring = dof_handler.n_locally_owned_dofs_per_processor();
	//locally_relevant_dofs_per_subdomain = DoFTools::locally_relevant_dofs_per_subdomain(dof_handler);
	extended_relevant_dofs = locally_relevant_dofs;
	if(rank > 0) {
		extended_relevant_dofs.add_range(locally_owned_dofs.nth_index_in_set(0) - n_neighboring[rank-1], locally_owned_dofs.nth_index_in_set(0));
	}

	deallog << "Computing block counts" <<std::endl;
	// Here we start computing the distribution of entries (indices thereof) to the specific blocks of the 3 matrices (system matrix and the 2 preconditioner matrices.)
  int prec_even_block_count = Utilities::MPI::n_mpi_processes(mpi_comm) /2;
  if(Utilities::MPI::n_mpi_processes(mpi_comm)%2 == 1) {
    prec_even_block_count++;
  }

  //int prec_odd_block_count = Utilities::MPI::n_mpi_processes(mpi_comm) /2 +1;

  i_sys_owned.resize(Layers);

  i_sys_readable.resize(Layers);

  for(unsigned int i = 0; i < Layers; i++) {
    int size = Block_Sizes[i];
    bool local = (i == rank);
    bool readable = (i == rank) || (i == rank+1);
    IndexSet temp(size);
    if(local) {
      temp.add_range(0,size);
    }
    IndexSet temp2(size);
    if(readable){
      temp2.add_range(0,size);
    }

    i_sys_owned[i] = temp;
    i_sys_readable[i] = temp2;

  }




  i_prec_even_owned_row.resize(Layers);
  i_prec_even_owned_col.resize(Layers);
  i_prec_even_writable.resize(Layers);
  i_prec_odd_owned_row.resize(Layers);
  i_prec_odd_owned_col.resize(Layers);
  i_prec_odd_writable.resize(Layers);


  for(unsigned  int i = 0; i < Layers; i++) {
    int size = Block_Sizes[i];
    bool even_row_owned = false;
    bool even_row_writable = false;
    bool even_col_owned = false;
    if(even){
      if (i == rank || i == rank +1) {
        even_row_owned = true;
        even_row_writable = true;
        even_col_owned = true;
      } else {
        if(i == rank -1) {
          even_row_writable = true;
        }
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

  for(unsigned  int i = 0; i < Layers; i++) {
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
      if ( i == rank -1) {
        odd_row_writable = true;
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

    if( rank == 0 && i == 0) {
      oro.add_range(0,size);
      orw.add_range(0,size);
      oco.add_range(0,size);
    }

    if(rank == Layers-1 && i == Layers-1) {
      oro.add_range(0,size);
      orw.add_range(0,size);
      oco.add_range(0,size);
    }

    i_prec_odd_owned_row[i] = oro;
    i_prec_odd_owned_col[i] = oco;
    i_prec_odd_writable[i] = orw;
  }

  int even_blocks = GlobalParams.NumberProcesses /2;
  int odd_blocks = GlobalParams.NumberProcesses / 2;

  if (GlobalParams.NumberProcesses % 2 == 1) {
    even_blocks ++;
  } else {
    odd_blocks ++;
  }

  std::vector<IndexSet> temp0 = i_prec_odd_owned_row;
  std::vector<IndexSet> temp1 = i_prec_odd_owned_col;
  std::vector<IndexSet> temp2 = i_prec_odd_writable;
  i_prec_odd_owned_row.clear();
  i_prec_odd_owned_col.clear();
  i_prec_odd_writable.clear();
  i_prec_odd_owned_row.push_back(temp0[0]);
  i_prec_odd_owned_col.push_back(temp1[0]);
  i_prec_odd_writable.push_back(temp2[0]);

  for( int i = 2; i < (int)Layers; i+=2) {
    i_prec_odd_owned_row.push_back(combine_indexes(temp0[i-1], temp0[i]));
    i_prec_odd_owned_col.push_back(combine_indexes(temp1[i-1], temp1[i]));
    i_prec_odd_writable.push_back(combine_indexes(temp2[i-1], temp2[i]));
  }

  if (GlobalParams.NumberProcesses % 2 == 0) {
    i_prec_odd_owned_row.push_back(temp0[GlobalParams.NumberProcesses-1]);
    i_prec_odd_owned_col.push_back(temp1[GlobalParams.NumberProcesses-1]);
    i_prec_odd_writable.push_back(temp2[GlobalParams.NumberProcesses-1]);
  }

  temp0 = i_prec_even_owned_row;
  temp1 = i_prec_even_owned_col;
  temp2 = i_prec_even_writable;

  i_prec_even_owned_row.clear();
  i_prec_even_owned_col.clear();
  i_prec_even_writable.clear();

  for(int i = 1; i < (int)Layers; i+=2) {
    i_prec_even_owned_row.push_back(combine_indexes(temp0[i-1], temp0[i]));
    i_prec_even_owned_col.push_back(combine_indexes(temp1[i-1], temp1[i]));
    i_prec_even_writable.push_back(combine_indexes(temp2[i-1], temp2[i]));
  }

  if (GlobalParams.NumberProcesses % 2 == 1) {
    i_prec_even_owned_row.push_back(temp0[GlobalParams.NumberProcesses-1]);
    i_prec_even_owned_col.push_back(temp1[GlobalParams.NumberProcesses-1]);
    i_prec_even_writable.push_back(temp2[GlobalParams.NumberProcesses-1]);
  }

  deallog << "Constructing Sparsity Patterns and Constrain Matrices ... " <<std::endl;

	cm.clear();
	cm.reinit(locally_relevant_dofs);

	cm_prec_even.clear();
	cm_prec_odd.clear();
	cm_prec_even.reinit(locally_relevant_dofs);
	cm_prec_odd.reinit(locally_relevant_dofs);

	DoFTools::make_hanging_node_constraints(dof_handler, cm);
	DoFTools::make_hanging_node_constraints(dof_handler, cm_prec_even);
	DoFTools::make_hanging_node_constraints(dof_handler, cm_prec_odd);

	MakeBoundaryConditions();
	MakePreconditionerBoundaryConditions();

	cm.close();
	cm_prec_even.close();
	cm_prec_odd.close();


	deallog << "Initialization done." << std::endl;



	std::ostringstream set_string;

	locally_owned_dofs.write(set_string);

	std::string local_set = set_string.str();

	const char * test = local_set.c_str();

	char * text_local_set = const_cast<char*> (test);

	unsigned int text_local_length = strlen( text_local_set) ;

	const int mpi_size = Layers;

	int * all_lens = new int[mpi_size];
	int * displs = new int[mpi_size];

	deallog << "Communicating the Index Sets via MPI" <<std::endl;

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

	deallog << "Updating local structures with information from the other processes" <<std::endl;

	locally_relevant_dofs_all_processors.resize(Layers);


	for(unsigned int i= 0; i < Layers; i++ ) {
		std::istringstream ss;
		char *temp = &all_names[displs[i]] ;
		ss.rdbuf()->pubsetbuf(temp,strlen(temp));
		locally_relevant_dofs_all_processors[i].clear();
		locally_relevant_dofs_all_processors[i].set_size(dof_handler.n_dofs());
		locally_relevant_dofs_all_processors[i].read(ss);
	}

	UpperDofs = locally_owned_dofs;

	LowerDofs = locally_owned_dofs;


	if(rank != 0 ) {
		LowerDofs.add_indices(locally_relevant_dofs_all_processors[rank-1], 0);
	}

	if(rank != Layers -1 ) {
		UpperDofs.add_indices(locally_relevant_dofs_all_processors[rank+1], 0);
	}

	deallog << "Done computing Index Sets. Calling for reinit now." <<std::endl;

	reinit_all();

	deallog<< "Done" << std::endl;
	deallog.pop();
}

void Waveguide::calculate_cell_weights () {
  deallog.push("Computing cell weights");
  deallog << "Iterating cells and computing local norm of material tensor."<<std::endl;
  cell = triangulation.begin_active();
	endc = triangulation.end();

	for (; cell!=endc; ++cell){
		if(cell->is_locally_owned()) {
            Tensor<2,3, std::complex<double>> tens, epsilon_pre2, epsilon_pre1;
            Point<3> pos = cell->center();
            if(even) {
              epsilon_pre1 = st->get_Tensor(pos);
              epsilon_pre2 = st->get_Preconditioner_Tensor(pos, rank);

            } else {
              epsilon_pre2 = st->get_Tensor(pos);
              epsilon_pre1 = st->get_Preconditioner_Tensor(pos, rank);

            }
            tens = st->get_Tensor(pos);

            cell_weights(cell->active_cell_index()) = tens.norm();

            cell_weights_prec_1(cell->active_cell_index()) = epsilon_pre1.norm();

            cell_weights_prec_2(cell->active_cell_index()) = epsilon_pre2.norm();
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

	std::string path = solutionpath + "/" + path_prefix +"/cell-weights" + std::to_string(run_number) +"-" + std::to_string(rank) +".vtu";
	deallog<<"Writing vtu file: "<< path <<std::endl;

	std::ofstream outputvtu2 (path);
	data_out_cells.write_vtu(outputvtu2);
	deallog << "Done." <<std::endl;
	deallog.pop();
}

void Waveguide::reinit_all () {
  deallog.push("reinit_all");

  deallog<<"reinitializing right-hand side" <<std::endl;
	reinit_rhs();

	if(GlobalParams.O_O_V_T_TransformationWeightsAll) {
	  deallog<<"reinitializing cell weights" <<std::endl;
	  reinit_cell_weights();
	}
	if(GlobalParams.O_O_V_T_TransformationWeightsFirst && run_number == 0) {
	    deallog<<"reinitializing cell weights" <<std::endl;
	    reinit_cell_weights();
	  }

  deallog<<"reinitializing solutiuon" <<std::endl;
	reinit_solution();

	deallog<<"reinitializing preconditioner" <<std::endl;
	reinit_preconditioner();

	deallog<<"reinitializing system matrix" <<std::endl;
	reinit_systemmatrix();

	deallog << "Done" <<std::endl;
	deallog.pop();
}

void Waveguide::reinit_for_rerun () {
	reinit_rhs();
	reinit_preconditioner_fast();
	reinit_systemmatrix();

}

void Waveguide::reinit_systemmatrix() {

  deallog.push("reinit_systemmatrix");

  deallog << "Generating BSP" <<std::endl;

  TrilinosWrappers::BlockSparsityPattern sp(i_sys_owned, mpi_comm);

  deallog << "Collecting sizes ..." <<std::endl;

  sp.collect_sizes();

  deallog << "Making BSP ..." <<std::endl;
  DoFTools::make_sparsity_pattern (dof_handler, sp, cm, false, rank);
  sp.compress();

  deallog << "Initializing system_matrix ..." <<std::endl;
	system_matrix.reinit(sp);
	deallog.pop();
}

void Waveguide::reinit_rhs () {
	// std::cout << "Reinit rhs for p " << rank << std::endl;

	system_rhs.reinit(i_sys_owned, mpi_comm);

	preconditioner_rhs.reinit(dof_handler.n_dofs());

}

void Waveguide::reinit_solution() {
	solution.reinit(i_sys_owned,i_sys_readable, mpi_comm, true);
	EstimatedSolution.reinit(i_sys_owned, mpi_comm);
	ErrorOfSolution.reinit(i_sys_owned, mpi_comm);

}

void Waveguide::reinit_cell_weights() {
	cell_weights.reinit(triangulation.n_active_cells());
	cell_weights_prec_1.reinit(triangulation.n_active_cells());
	cell_weights_prec_2.reinit(triangulation.n_active_cells());
	calculate_cell_weights();
}

void Waveguide::reinit_storage() {
	storage.reinit(i_sys_owned,  mpi_comm);
}

void Waveguide::reinit_preconditioner () {
  deallog.push("reinit_preconditioner");

  deallog.push("Generating BSP");

  deallog << "Started" <<std::endl;

  TrilinosWrappers::BlockSparsityPattern epsp(i_prec_even_owned_row, i_prec_even_owned_col, i_prec_even_writable, mpi_comm);

  deallog << "Even worked. Continuing Odd." <<std::endl;

  TrilinosWrappers::BlockSparsityPattern opsp(i_prec_odd_owned_row, i_prec_odd_owned_col, i_prec_odd_writable, mpi_comm);

  deallog << "Odd worked. Done" <<std::endl;

  deallog.pop();

  deallog << "Collecting sizes ..." <<std::endl;
  epsp.collect_sizes();
  opsp.collect_sizes();

  deallog.push("Making BSP");

  deallog << "Even Preconditioner Matrices ..." <<std::endl;
  DoFTools::make_sparsity_pattern (dof_handler, epsp,
                                     cm_prec_even, false,
                     rank);
  epsp.compress();
  deallog << "Odd Preconditioner Matrices ..." <<std::endl;
  DoFTools::make_sparsity_pattern (dof_handler, opsp,
                                       cm_prec_odd, false,
                       rank);
  opsp.compress();
  deallog << "Done" <<std::endl;

  deallog.pop();

  deallog.push("Initializing matrices");
  deallog<<"Even ..."<<std::endl;
  prec_matrix_even.reinit(epsp);
  deallog<<"Odd ..."<<std::endl;
  prec_matrix_odd.reinit(opsp);
  deallog << "Done." <<std::endl;
  deallog.pop();

  deallog.pop();

}

void Waveguide::reinit_preconditioner_fast () {


}

void Waveguide::assemble_system ()
{

	reinit_rhs();

	QGauss<3>  quadrature_formula(2);
	const FEValuesExtractors::Vector real(0);
	const FEValuesExtractors::Vector imag(3);
	FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
	std::vector<Point<3> > quadrature_points;
	const unsigned int   dofs_per_cell	= fe.dofs_per_cell;
	const unsigned int   n_q_points		= quadrature_formula.size();

	deallog << "Starting Assemblation process" << std::endl;

	FullMatrix<double>  cell_matrix_real (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  cell_matrix_prec_odd (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  cell_matrix_prec_even (dofs_per_cell, dofs_per_cell);

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
    if( subdomain_id == rank) {
      fe_values.reinit (cell);
      quadrature_points = fe_values.get_quadrature_points();

      cell_matrix_real = 0;
      cell_matrix_prec_odd = 0;
      cell_matrix_prec_even = 0;
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
        if(!locals_set) {
          if(quadrature_points[q_index][2] < minimum_local_z) {
            minimum_local_z =quadrature_points[q_index][2];
          }
          if(quadrature_points[q_index][2] > maximum_local_z) {
            maximum_local_z =quadrature_points[q_index][2];
          }
        }
        transformation = st->get_Tensor(quadrature_points[q_index]);

        if( mg->math_coordinate_in_waveguide(quadrature_points[q_index])) {
          epsilon = transformation * eps_in;
        } else {
          epsilon = transformation * eps_out;
        }

        mu = invert(transformation) / mu_zero;


        if(even) {
          epsilon_pre1 = st->get_Tensor(quadrature_points[q_index]);
          mu_prec1 = st->get_Tensor(quadrature_points[q_index]);
          epsilon_pre2 = st->get_Preconditioner_Tensor(quadrature_points[q_index], rank);
          mu_prec2 = st->get_Preconditioner_Tensor(quadrature_points[q_index], rank);
        } else {
          epsilon_pre2 = st->get_Tensor(quadrature_points[q_index]);
          mu_prec2 = st->get_Tensor(quadrature_points[q_index]);
          epsilon_pre1 = st->get_Preconditioner_Tensor(quadrature_points[q_index], rank);
          mu_prec1 = st->get_Preconditioner_Tensor(quadrature_points[q_index], rank);
        }

        mu_prec1 = invert(mu_prec1) / mu_zero;
        mu_prec2 = invert(mu_prec2) / mu_zero;


        if( mg->math_coordinate_in_waveguide(quadrature_points[q_index])) {
          epsilon_pre1 *= eps_in;
          epsilon_pre2 *= eps_in;
        } else {
          epsilon_pre1 *= eps_out;
          epsilon_pre2 *= eps_out;
        }

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
            cell_matrix_prec_even[i][j] += pre1.real();

            std::complex<double> pre2 = (mu_prec2 * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon_pre2 * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.C_omega*GlobalParams.C_omega;
            cell_matrix_prec_odd[i][j] += pre2.real();

          }
        }
      }
      cell->get_dof_indices (local_dof_indices);
      // pout << "Starting distribution"<<std::endl;
      cm.distribute_local_to_global     (cell_matrix_real, cell_rhs, local_dof_indices,system_matrix, system_rhs, true);
      // pout << "P1 done"<<std::endl;

      cm_prec_odd.distribute_local_to_global(cell_matrix_prec_odd, cell_rhs, local_dof_indices,prec_matrix_odd, preconditioner_rhs, true);

      cm_prec_even.distribute_local_to_global(cell_matrix_prec_even, cell_rhs, local_dof_indices,prec_matrix_even, preconditioner_rhs, true);

      // pout << "P2 done"<<std::endl;
    }
  }

  locals_set = true;

	MPI_Barrier(mpi_comm);
	if(!is_stored)  deallog<<"Assembling done. L2-Norm of RHS: "<< system_rhs.l2_norm()<<std::endl;

	system_matrix.compress(VectorOperation::add);
	system_rhs.compress(VectorOperation::add);

	prec_matrix_even.compress(VectorOperation::add);
	prec_matrix_odd.compress(VectorOperation::add);

	cm.distribute(solution);
	cm.distribute(EstimatedSolution);
	cm.distribute(ErrorOfSolution);
	MPI_Barrier(mpi_comm);
	deallog << "Distributing solution done." <<std::endl;
	estimate_solution();

}

void Waveguide::MakeBoundaryConditions (){
	DoFHandler<3>::active_cell_iterator cell, endc;

	ComponentMask comp_mask_real = fe.component_mask(real);
	ComponentMask comp_mask_imag = fe.component_mask(imag);


	DoFTools::make_zero_boundary_constraints(dof_handler,cm, comp_mask_imag );
	DoFTools::make_zero_boundary_constraints(dof_handler,cm, comp_mask_real );

	ExactSolution es;

	VectorTools::project_boundary_values_curl_conforming_l2( dof_handler, 0, es, 0, cm);

	fixed_dofs.set_size(dof_handler.n_dofs());

	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if(cell->is_locally_owned()) {
			for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
				Point<3, double> center =(cell->face(i))->center(true, false);
				if( center[0] < 0) center[0] *= (-1.0);
				if( center[1] < 0) center[1] *= (-1.0);

				if ( std::abs( center[0] - GlobalParams.M_R_XLength/2.0) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						if(locally_owned_dofs.is_element(local_dof_indices[0])) {
							cm.add_line(local_dof_indices[0]);
							cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
							fixed_dofs.add_index(local_dof_indices[0]);
						}
						/**
						if(locally_owned_dofs.is_element(local_dof_indices[1])) {
							cm.add_line(local_dof_indices[1]);
							cm.set_inhomogeneity(local_dof_indices[1], 0.0);
							fixed_dofs.add_index(local_dof_indices[1]);
						}
						**/
					}
				}
				if ( std::abs( center[1] - GlobalParams.M_R_YLength/2.0) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						if(locally_owned_dofs.is_element(local_dof_indices[0])) {
							cm.add_line(local_dof_indices[0]);
							cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
							fixed_dofs.add_index(local_dof_indices[0]);
						}
						/**
						if(locally_owned_dofs.is_element(local_dof_indices[1])) {
							cm.add_line(local_dof_indices[1]);
							cm.set_inhomogeneity(local_dof_indices[1], 0.0);
							fixed_dofs.add_index(local_dof_indices[1]);
						}
						**/
					}
				}
				if( std::abs(center[2] + GlobalParams.M_R_ZLength/2.0 ) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						if((cell->face(i))->line(j)->at_boundary()) {
							((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
							Tensor<1,3,double> ptemp = ((cell->face(i))->line(j))->center(true, false);
							Point<3, double> p (ptemp[0], ptemp[1], ptemp[2]);
							Tensor<1,3,double> dtemp = ((cell->face(i))->line(j))->vertex(1) - ((cell->face(i))->line(j))->vertex(0);
							double norm = dtemp.norm();
							Point<3, double> direction (dtemp[0]/norm, dtemp[1]/norm, dtemp[2]/norm);

							double result = TEMode00(p,0);
							if(st->PML_in_X(p) || st->PML_in_Y(p)) result = 0.0;
							if(locally_owned_dofs.is_element(local_dof_indices[0])) {
								cm.add_line(local_dof_indices[0]);
								cm.set_inhomogeneity(local_dof_indices[0], direction[0] * result );
								fixed_dofs.add_index(local_dof_indices[0]);
							}
							/**
							if(locally_owned_dofs.is_element(local_dof_indices[1])) {
								cm.add_line(local_dof_indices[1]);
								cm.set_inhomogeneity(local_dof_indices[1], 0.0);
								fixed_dofs.add_index(local_dof_indices[1]);
							}
							**/
						}
					}
				}
				if( std::abs(center[2] - GlobalParams.Maximum_Z) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						if(locally_owned_dofs.is_element(local_dof_indices[0])) {
							cm.add_line(local_dof_indices[0]);
							cm.set_inhomogeneity(local_dof_indices[0], 0.0 );
							fixed_dofs.add_index(local_dof_indices[0]);
						}
						/**
						if(locally_owned_dofs.is_element(local_dof_indices[1])) {
							cm.add_line(local_dof_indices[1]);
							cm.set_inhomogeneity(local_dof_indices[1], 0.0);
							fixed_dofs.add_index(local_dof_indices[1]);
						}
						 **/
					}
				}
			}
		}
	}
}

void Waveguide::MakePreconditionerBoundaryConditions (  ){
	DoFHandler<3>::active_cell_iterator cell, endc;

	double layer_length = GlobalParams.LayerThickness;

	cell = dof_handler.begin_active();
	endc = dof_handler.end();
	IndexSet own (dof_handler.n_dofs());
	own.add_indices(locally_owned_dofs);
	sweepable.set_size(dof_handler.n_dofs());
	sweepable.add_indices(locally_owned_dofs);
	if(rank != 0 ){
		own.add_indices(LowerDofs);
	}

	for (; cell!=endc; ++cell)
	{
		if(std::abs(cell->subdomain_id() - rank) < 3 ) {

			for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
				Point<3, double> center =(cell->face(i))->center(true, false);
				if( center[0] < 0) center[0] *= (-1.0);
				if( center[1] < 0) center[1] *= (-1.0);

				// Set x-boundary values
				if ( std::abs( center[0] - GlobalParams.M_R_XLength/2.0) < 0.0001){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						for(unsigned int k = 0; k < 2; k++) {
							if(own.is_element(local_dof_indices[k])) {
								cm_prec_odd.add_line(local_dof_indices[k]);
								cm_prec_odd.set_inhomogeneity(local_dof_indices[k], 0.0 );
								cm_prec_even.add_line(local_dof_indices[k]);
								cm_prec_even.set_inhomogeneity(local_dof_indices[k], 0.0 );
							}
						}
					}
				}

				// Set y-boundary values
				if ( std::abs( center[1] - GlobalParams.M_R_YLength/2.0) < 0.0001 ){
					std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
					for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
						((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
						for(unsigned int k = 0; k < 2; k++) {
							if(own.is_element(local_dof_indices[k])) {
								cm_prec_odd.add_line(local_dof_indices[k]);
								cm_prec_odd.set_inhomogeneity(local_dof_indices[k], 0.0 );
								cm_prec_even.add_line(local_dof_indices[k]);
								cm_prec_even.set_inhomogeneity(local_dof_indices[k], 0.0 );
							}
						}
					}
				}

				if(even){
				  if(rank != 0){
            if ( std::abs( center[2] + GlobalParams.M_R_ZLength/2.0 - (rank * layer_length)) < 0.0001){
              std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
              for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
                ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
                for(unsigned int k = 0; k < 2; k++) {
                  if(own.is_element(local_dof_indices[k])) {
                    cm_prec_even.add_line(local_dof_indices[k]);
                    cm_prec_even.set_inhomogeneity(local_dof_indices[k], 0.0 );
                  }
                }
              }
            }
				  }

          if ( std::abs( center[2] + GlobalParams.M_R_ZLength/2.0 - ((rank+2) * layer_length)) < 0.0001){
            std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
            for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
              ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
              for(unsigned int k = 0; k < 2; k++) {
                if(own.is_element(local_dof_indices[k])) {
                  cm_prec_even.add_line(local_dof_indices[k]);
                  cm_prec_even.set_inhomogeneity(local_dof_indices[k], 0.0 );
                }
              }
            }
          }

          if ( std::abs( center[2] + GlobalParams.M_R_ZLength/2.0 - ((rank+1) * layer_length)) < 0.0001){
            std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
            for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
              ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
              for(unsigned int k = 0; k < 2; k++) {
                if(own.is_element(local_dof_indices[k])) {
                  cm_prec_odd.add_line(local_dof_indices[k]);
                  cm_prec_odd.set_inhomogeneity(local_dof_indices[k], 0.0 );
                }
              }
            }
          }

          if ( std::abs( center[2] + GlobalParams.M_R_ZLength/2.0 - ((rank-1) * layer_length)) < 0.0001){
            std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
            for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
              ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
              for(unsigned int k = 0; k < 2; k++) {
                if(own.is_element(local_dof_indices[k])) {
                  cm_prec_odd.add_line(local_dof_indices[k]);
                  cm_prec_odd.set_inhomogeneity(local_dof_indices[k], 0.0 );
                }
              }
            }
          }
				} else {
				  if ( std::abs( center[2] + GlobalParams.M_R_ZLength/2.0 - (rank * layer_length)) < 0.0001){
            std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
            for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
              ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
              for(unsigned int k = 0; k < 2; k++) {
                if(own.is_element(local_dof_indices[k])) {
                  cm_prec_odd.add_line(local_dof_indices[k]);
                  cm_prec_odd.set_inhomogeneity(local_dof_indices[k], 0.0 );
                }
              }
            }
          }

          if ( std::abs( center[2] + GlobalParams.M_R_ZLength/2.0 - ((rank+2) * layer_length)) < 0.0001){
            std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
            for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
              ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
              for(unsigned int k = 0; k < 2; k++) {
                if(own.is_element(local_dof_indices[k])) {
                  cm_prec_odd.add_line(local_dof_indices[k]);
                  cm_prec_odd.set_inhomogeneity(local_dof_indices[k], 0.0 );
                }
              }
            }
          }

          if ( std::abs( center[2] + GlobalParams.M_R_ZLength/2.0 - ((rank+1) * layer_length)) < 0.0001){
            std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
            for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
              ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
              for(unsigned int k = 0; k < 2; k++) {
                if(own.is_element(local_dof_indices[k])) {
                  cm_prec_even.add_line(local_dof_indices[k]);
                  cm_prec_even.set_inhomogeneity(local_dof_indices[k], 0.0 );
                }
              }
            }
          }

          if ( std::abs( center[2] + GlobalParams.M_R_ZLength/2.0 - ((rank-1) * layer_length)) < 0.0001){
            std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_line);
            for(unsigned int j = 0; j< GeometryInfo<3>::lines_per_face; j++) {
              ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
              for(unsigned int k = 0; k < 2; k++) {
                if(own.is_element(local_dof_indices[k])) {
                  // cm_prec_even.add_line(local_dof_indices[k]);
                  // cm_prec_even.set_inhomogeneity(local_dof_indices[k], 0.0 );
                }
              }
            }
          }
				}


				if( rank <= 1 && std::abs(center[2] + GlobalParams.M_R_ZLength/2.0 ) < 0.0001 ){
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
							if(st->PML_in_X(p) || st->PML_in_Y(p)) result = 0.0;
							if(locally_owned_dofs.is_element(local_dof_indices[0])) {
								cm_prec_even.add_line(local_dof_indices[0]);
								cm_prec_even.set_inhomogeneity(local_dof_indices[0], direction[0] * result );
							}
							if(locally_owned_dofs.is_element(local_dof_indices[1])) {
								cm_prec_even.add_line(local_dof_indices[1]);
								cm_prec_even.set_inhomogeneity(local_dof_indices[1], 0.0);
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
	if((int)iteration > GlobalParams.So_TotalSteps){
		// pout << std::endl;
		return SolverControl::State::failure;
	}
	if(check_value < GlobalParams.So_Precision){
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

	// Convergence_Table.set_auto_fill_mode(true);

	if(run_number != 0) {
	  result_file.close();
	}

	result_file.open((solutionpath + "/" + path_prefix + "/solution_of_run_" + std::to_string(run_number)  + ".dat").c_str());

	if(GlobalParams.So_Solver == SolverOptions::GMRES) {

	  if(run_number > 0) {
      for(unsigned int i = 0; i < locally_owned_dofs.n_elements(); i++) {
        int index = locally_owned_dofs.nth_index_in_set(i);
        solution[index] = solution_for_computations[index];
      }
	  }

		dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::BlockVector> solver(solver_control , dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::BlockVector>::AdditionalData(GlobalParams.So_RestartSteps) );
		
		int above = 0;
		if ((int)rank != GlobalParams.NumberProcesses - 1) {
			above = locally_relevant_dofs_all_processors[rank+1].n_elements();
		}

		PreconditionerSweeping sweep( mpi_comm, locally_owned_dofs.n_elements(), above, dof_handler.max_couplings_between_dofs(), locally_owned_dofs, &fixed_dofs, rank);

		if(rank == 0) {
			sweep.Prepare(solution);
		}

		MPI_Barrier(mpi_comm);

		if(even) {
		  sweep.matrix = & prec_matrix_even.block(rank /2, rank/2);
		} else {
		  sweep.matrix = & prec_matrix_odd.block((rank+1) /2, (rank+1)/2);
		}

		if((int)rank == GlobalParams.NumberProcesses-1) {
		   sweep.matrix = & system_matrix.block(rank, rank);
		}

		MPI_Barrier(mpi_comm);

		deallog << "Initializing the Preconditioner..." <<std::endl;

		if((int)rank < GlobalParams.NumberProcesses-1){
		  sweep.init(solver_control, & system_matrix.block(rank,  rank+1 ));
		} else {
		  sweep.init(solver_control, & system_matrix.block(rank,  rank));
		}

		solver.connect(std_cxx11::bind (&Waveguide::residual_tracker,
                                   this,
                                   std_cxx11::_1,
                                   std_cxx11::_2,
                                   std_cxx11::_3));

  	MPI_Barrier(mpi_comm);

		deallog << "Preconditioner Ready. Solving..." <<std::endl;

		bool changed_console = false;

		if(rank == 0 && GlobalParams.MPI_Rank != 0) {
		  deallog.depth_console(10);
		  changed_console = true;
		}

		struct timeval tp;
		gettimeofday(&tp, NULL);
		solver_start_milis = tp.tv_sec * 1000 + tp.tv_usec / 1000;

		try {
		  solver.solve(system_matrix,solution, system_rhs, sweep);
		} catch(const dealii::SolverControl::NoConvergence & e) {
		  deallog << "NO CONVERGENCE!" <<std::endl;
		}

		while( steps < 40) {
		  struct timeval tp;
      gettimeofday(&tp, NULL);
      long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000 - solver_start_milis;

      // result_file << "" << Iteration << "\t" << residual << "\t" << ms <<std::endl;

      Convergence_Table.add_value(path_prefix + std::to_string(run_number) + "Iteration", steps +1);
      Convergence_Table.add_value(path_prefix + std::to_string(run_number) + "Residual", 0.0);
      Convergence_Table.add_value(path_prefix + std::to_string(run_number) + "Time", std::to_string(ms));
		  steps++;
		}

		if((GlobalParams.O_C_D_ConvergenceFirst && run_number==0) || GlobalParams.O_C_D_ConvergenceAll) {
      Convergence_Table.add_column_to_supercolumn(path_prefix + std::to_string(run_number) + "Iteration", "Run " + std::to_string(run_number));
      Convergence_Table.add_column_to_supercolumn(path_prefix + std::to_string(run_number) + "Residual", "Run " + std::to_string(run_number));
      Convergence_Table.add_column_to_supercolumn(path_prefix + std::to_string(run_number) + "Time", "Run " + std::to_string(run_number));

      // Convergence_Table.omit_column_from_convergence_rate_evaluation(path_prefix + std::to_string(run_number) + "Iteration");
      // Convergence_Table.omit_column_from_convergence_rate_evaluation(path_prefix + std::to_string(run_number) + "Time");
      Convergence_Table.evaluate_convergence_rates(path_prefix + std::to_string(run_number) + "Residual",path_prefix + std::to_string(run_number) + "Iteration",ConvergenceTable::RateMode::reduction_rate);
		}
		if(changed_console) {
		  deallog.depth_console(0);
		}

	  deallog << "Done." << std::endl;

		deallog << "Norm of the solution: " << solution.l2_norm() << std::endl;
	}

	cm.distribute(solution);

	if(GlobalParams.So_Solver == SolverOptions::UMFPACK) {
		SolverControl sc2(2,false,false);
		TrilinosWrappers::SolverDirect temp_s(sc2, TrilinosWrappers::SolverDirect::AdditionalData(false, PrecOptionNames[GlobalParams.So_Preconditioner]));
		// temp_s.solve(system_matrix, solution, system_rhs);
	}

	solution_for_computations.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

	for(unsigned int i= 0; i< locally_owned_dofs.n_elements(); i++){
	  int index = locally_owned_dofs.nth_index_in_set(i);
	  solution_for_computations[index] = solution[index];
	}

	solution_for_computations.update_ghost_values();

}

void Waveguide::store() {
	reinit_storage();
	// storage.reinit(dof_handler.n_dofs());
	storage = solution;
	is_stored = true;
}

void Waveguide::output_results ( bool  )
{

	for(unsigned int i = 0; i < locally_owned_dofs.n_elements(); i++) {
		unsigned int index = locally_owned_dofs.nth_index_in_set(i);
		ErrorOfSolution[index] = solution[index] - EstimatedSolution[index];
	}

	MPI_Barrier(mpi_comm);

	std::vector<IndexSet> i_sys_relevant;
	i_sys_relevant.resize(GlobalParams.NumberProcesses);
	int below = 0;
	for(int i = 0; i < GlobalParams.NumberProcesses; i++){
	  IndexSet local(Block_Sizes[i]);
	  if(i != (int)rank){
      for(unsigned int j = 0; j < locally_relevant_dofs.n_elements(); j++) {
        int idx = locally_relevant_dofs.nth_index_in_set(j);
        if( idx >= below && idx < below + Block_Sizes[i]){
          local.add_index(idx - below);
        }
      }

	  }else {
	    local = i_sys_owned[i];
	  }
	  below += Block_Sizes[i];
	  i_sys_relevant[i] = local;
	}

	TrilinosWrappers::MPI::BlockVector solution_output(i_sys_owned, i_sys_relevant, mpi_comm);
	solution_output = solution;

	TrilinosWrappers::MPI::BlockVector estimate_output(i_sys_owned, i_sys_relevant, mpi_comm);
	estimate_output = EstimatedSolution;

	TrilinosWrappers::MPI::BlockVector error_output(i_sys_owned, i_sys_relevant, mpi_comm);
	error_output = ErrorOfSolution;

  MPI_Barrier(mpi_comm);

	std::cout << rank << ": " <<locally_owned_dofs.n_elements()<< "," <<locally_owned_dofs.nth_index_in_set(0) << "," << locally_owned_dofs.nth_index_in_set(locally_owned_dofs.n_elements()-1) <<std::endl;
	// evaluate_overall();
	if(true) {
		DataOut<3> data_out;

		data_out.attach_dof_handler (dof_handler);
		data_out.add_data_vector (solution_output, "Solution",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_dof_data);
		data_out.add_data_vector (error_output, "Error_Of_Estimated_Solution",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_dof_data);
		data_out.add_data_vector (estimate_output, "Estimated_Solution",dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3, 3>::DataVectorType::type_dof_data);
		// data_out.add_data_vector(differences, "L2error");

		data_out.build_patches ();

		std::ofstream outputvtu (solutionpath + "/" + path_prefix+ "/solution-run" + std::to_string( run_number) + "-P" + std::to_string(rank) +".vtu");
		data_out.write_vtu(outputvtu);

        
        
		if ( false ) {
			std::ofstream pattern (solutionpath  + "/" + path_prefix +"/pattern.gnu");

			std::ofstream patternscript (solutionpath + "/" + path_prefix+ "/displaypattern.gnu");
			patternscript << "set style line 1000 lw 1 lc \"black\"" <<std::endl;
			for(int i = 0; i < GlobalParams.M_W_Sectors; i++) {
				patternscript << "set arrow " << 1000 + 2*i << " from 0,-" << Dofs_Below_Subdomain[i] << " to "<<dof_handler.n_dofs()<<",-"<<Dofs_Below_Subdomain[i]<<" nohead ls 1000 front"<<std::endl;
				patternscript << "set arrow " << 1001 + 2*i  << " from " << Dofs_Below_Subdomain[i] << ",0 to " << Dofs_Below_Subdomain[i] << ", -"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;
			}
			patternscript << "set arrow " << 1000 + 2*GlobalParams.M_W_Sectors << " from 0,-" << dof_handler.n_dofs() << " to "<<dof_handler.n_dofs()<<",-"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;
			patternscript << "set arrow " << 1001 + 2*GlobalParams.M_W_Sectors << " from " << dof_handler.n_dofs() << ",0 to " << dof_handler.n_dofs() << ", -"<<dof_handler.n_dofs()<<" nohead ls 1000 front"<<std::endl;

			patternscript << "plot \"pattern.gnu\" with dots" <<std::endl;
			patternscript.flush();
		}


	}
}

void Waveguide::run ()
{

  deallog.push("Waveguide_" + path_prefix + "_run");

  if(run_number == 0) {

  deallog << "Setting up the mesh..."<<std::endl;
	timer.enter_subsection ("Setup Mesh");
	make_grid ();
	timer.leave_subsection();

	Compute_Dof_Numbers();

	deallog << "Setting up FEM..."<<std::endl;
	timer.enter_subsection ("Setup FEM");
	setup_system ();
	timer.leave_subsection();


	timer.enter_subsection ("Reset");
	timer.leave_subsection();

  } else {
    timer.enter_subsection ("Reset");
    reinit_all();
    timer.leave_subsection();

  }

	deallog.push("Assembly");
	deallog << "Assembling the system..."<<std::endl;
	timer.enter_subsection ("Assemble");
	assemble_system ();
	timer.leave_subsection();
	deallog.pop();

	deallog.push("Solving");
	deallog << "Solving the system..."<<std::endl;
	timer.enter_subsection ("Solve");
	solve ();
	timer.leave_subsection();
	deallog.pop();

	timer.enter_subsection ("Evaluate");
	//evaluate ();
	timer.leave_subsection();

	timer.print_summary();

	deallog << "Writing outputs..."<<std::endl;
	timer.reset();
    
  output_results(false);
    
  deallog.pop();
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

std::vector<std::complex<double>> Waveguide::assemble_adjoint_local_contribution(Waveguide * other, double stepwidth) {
  deallog.push("Waveguide:adj_local");

  deallog << "Computing adjoint based shape derivative contributions..."<<std::endl;

  int other_proc = rank + 1;

  std::vector<std::complex<double>> ret;
  const unsigned int ndofs = st->NDofs();
  ret.resize(ndofs);
  for(unsigned int i = 0; i < ndofs; i++) {
    ret[i] = 0;
  }
  std::vector<bool> local_supported_dof;
  local_supported_dof.resize(ndofs);
  int min = ndofs;
  int max = -1;
  for( int i =0; i < (int)ndofs; i++) {
    std::pair<double, double> support = st->dof_support(i);
    if ((support.first >= minimum_local_z && support.first <= maximum_local_z) || (support.second >= minimum_local_z && support.second <= maximum_local_z) ) {
     if (i > max && st->IsDofFree(i)) {
       max = i;
     }
     if(i < min && st->IsDofFree(i)) {
       min = i;
     }
    }
  }

  QGauss<3>  quadrature_formula(1);
  const FEValuesExtractors::Vector real(0);
  const FEValuesExtractors::Vector imag(3);
  FEValues<3> fe_values (fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
  std::vector<Point<3> > quadrature_points;
  const unsigned int   n_q_points   = quadrature_formula.size();

  Tensor<2,3, std::complex<double>>     transformation;

  int total = triangulation.n_active_cells() * quadrature_formula.size();
  int counter =0;
  double * returned_vector = new double[6];
  for(unsigned int temp_counter = 0; temp_counter < 2; temp_counter++ ){
    if( rank%2 == temp_counter%2  ) {
      if( (int)rank != GlobalParams.NumberProcesses -1){
        deallog.push("local cell phase");
        deallog << "This process is now computing its own contributions to the shape gradient together with "<< other_proc<<"."  << std::endl;

        DoFHandler<3>::active_cell_iterator cell, endc;
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
          if( cell->is_locally_owned()) {
            fe_values.reinit (cell);
            quadrature_points = fe_values.get_quadrature_points();

            for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
            {
              Tensor<1,3,std::complex<double>> own_solution = solution_evaluation(quadrature_points[q_index]);

              MPI_Send(&quadrature_points[q_index][0], 3, MPI_DOUBLE, rank+1, 0, mpi_comm);

              MPI_Recv(&returned_vector[0], 6, MPI_DOUBLE, rank+1, 0, mpi_comm, MPI_STATUS_IGNORE);

              Tensor<1,3,std::complex<double>> other_solution;

              other_solution[0].real(returned_vector[0]);
              other_solution[0].imag(returned_vector[1]);
              other_solution[1].real(returned_vector[2]);
              other_solution[1].imag(returned_vector[3]);
              other_solution[2].real(returned_vector[4]);
              other_solution[2].imag(returned_vector[5]);

              const double JxW = fe_values.JxW(q_index);
              for ( int j = min; j <= max; j++) {
                transformation = st->get_Tensor_for_step(quadrature_points[q_index], j, stepwidth);
                if(st->point_in_dof_support(quadrature_points[q_index], j)) {
                  ret[j] += own_solution * transformation * other_solution * JxW;
                }
              }
              counter ++;
              if((counter-1)/(total/10) != (counter)/(total/10)) {
                deallog << (int) (100 * (counter)/(total)) << "%"<<std::endl ;
             }
            }
          }
        }


        double  * end_signal = new double[3];
        end_signal[0] = -GlobalParams.M_R_ZLength;
        end_signal[1] = -GlobalParams.M_R_ZLength;
        end_signal[2] = -GlobalParams.M_R_ZLength;

        MPI_Send(&end_signal[0], 3, MPI_DOUBLE, rank+1, 0, mpi_comm);
        deallog << "Done." << std::endl;
        deallog.pop();
      }
    } else {
      deallog.push("non-local cell phase");
      if(rank != 0){
        deallog << "This process is now adjoint based contributions for process "<< rank -1<<"." << std::endl;
        bool normal = true;
        while(normal) {
          double * position_array = new double[3];
          MPI_Recv(&position_array[0], 3, MPI_DOUBLE, rank -1 , 0, mpi_comm, MPI_STATUS_IGNORE);
          normal = false;
          for(int i = 0; i < 3; i++) {
            if(position_array[i] != -GlobalParams.M_R_ZLength){
              normal = true;
            }
          }
          // deallog << "Received request for (" << position_array[0] << ", "<< position_array[1] << ", "<<position_array[2]<<")"<<std::endl;
          if(normal) {
            double * result = new double[6];
            Point<3, double> position;
            for(int i = 0; i < 3; i++) {
              position[i] = position_array[i];
            }
            other->adjoint_solution_evaluation(position, result);
            MPI_Send(&result[0], 6, MPI_DOUBLE, rank-1, 0, mpi_comm);
          }
          // deallog << "Sent a solution."<<std::endl;
        }
        deallog << "Done." << std::endl;
      } else {
        deallog << "This process skips phase two of adjoint gradient computation because it is PML blocked." << std::endl;
      }
      deallog.pop();
    }
    MPI_Barrier(mpi_comm);
  }


  deallog << "Done." << std::endl;
  deallog.pop();
  return ret;
}

Tensor<1,3,std::complex<double>> Waveguide::solution_evaluation(Point<3,double> position) const {
  // deallog << "Process " << GlobalParams.MPI_Rank << " as " << rank << " evaluating at (" << position[0] << "," << position[1] << "," << position[2] << "). The local range is ["<< minimum_local_z<<","<<maximum_local_z<<"]"<< std::endl;
  Tensor<1,3,std::complex<double>> ret;
  Vector<double> result(6);

  VectorTools::point_value(dof_handler, solution_for_computations, position, result);

  ret[0] = std::complex<double>(result(0), result(3));
  ret[1] = std::complex<double>(result(1), result(4));
  ret[2] = std::complex<double>(result(2), result(5));
  return ret;
}

void Waveguide::solution_evaluation(Point<3,double> position, double * sol) const {
  // deallog << "Process " << GlobalParams.MPI_Rank << " as " << rank << " evaluating at (" << position[0] << "," << position[1] << "," << position[2] << "). The local range is ["<< minimum_local_z<<","<<maximum_local_z<<"]"<< std::endl;
  Tensor<1,3,std::complex<double>> ret;
  Vector<double> result(6);
  VectorTools::point_value(dof_handler, solution_for_computations, position, result);
  for(int i = 0; i < 6; i++){
    sol[i] = result(i);
  }
}

Tensor<1,3,std::complex<double>> Waveguide::adjoint_solution_evaluation(Point<3,double> position) const {
  // deallog << "Process " << GlobalParams.MPI_Rank << " as " << rank << " evaluating at (" << position[0] << "," << position[1] << "," << position[2] << "). The local range is ["<< minimum_local_z<<","<<maximum_local_z<<"]"<< std::endl;
  Tensor<1,3,std::complex<double>> ret;
  Vector<double> result(6);
  position[2] = - position[2];
  VectorTools::point_value(dof_handler, solution_for_computations, position, result);
  ret[0] = std::complex<double>(result(0), result(3));
  ret[1] = std::complex<double>(result(1), result(4));
  ret[2] = std::complex<double>(- result(2), - result(5));
  return ret;
}

void Waveguide::adjoint_solution_evaluation(Point<3,double> position, double * sol) const {
  // deallog << "Process " << GlobalParams.MPI_Rank << " as " << rank << " evaluating at (" << position[0] << "," << position[1] << "," << position[2] << "). The local range is ["<< minimum_local_z<<","<<maximum_local_z<<"]"<< std::endl;

  Tensor<1,3,std::complex<double>> ret;
  Vector<double> result(6);
  position[2] = - position[2];
  VectorTools::point_value(dof_handler, solution_for_computations, position, result);
  for(int i = 0; i < 6; i++) {
    sol[i] = result(i);
  }
  sol[5] *= -1;
  sol[6] *= -1;
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
	// structure->Print();

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
	//evaluate ();
	timer.leave_subsection();

	timer.print_summary();
	timer.reset();

	run_number++;

}

SolverControl::State Waveguide::residual_tracker(unsigned int Iteration, double residual, dealii::TrilinosWrappers::MPI::BlockVector) {
  if((GlobalParams.O_C_D_ConvergenceFirst && run_number==0) || GlobalParams.O_C_D_ConvergenceAll) {
    
    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000 - solver_start_milis;
        
    // result_file << "" << Iteration << "\t" << residual << "\t" << ms <<std::endl;

    Convergence_Table.add_value(path_prefix + std::to_string(run_number) + "Iteration", Iteration);
    Convergence_Table.add_value(path_prefix + std::to_string(run_number) + "Residual", residual);
    Convergence_Table.add_value(path_prefix + std::to_string(run_number) + "Time", std::to_string(ms));
  }
  steps = Iteration;
  SolverControl::State ret = SolverControl::State::iterate;
  if((int)Iteration > GlobalParams.So_TotalSteps){
    // pout << std::endl;
    return SolverControl::State::failure;
  }
  if(residual < GlobalParams.So_Precision){
    // pout << std::endl;
    return SolverControl::State::success;
  }
  return ret;
}

#endif
