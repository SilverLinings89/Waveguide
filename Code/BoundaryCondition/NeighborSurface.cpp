#include "./NeighborSurface.h"
#include <deal.II/base/geometry_info.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include "../Core/InnerDomain.h"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <mpi.h>
#include <cstdlib>
#include <string>
#include "./BoundaryCondition.h"
#include "PMLMeshTransformation.h"

NeighborSurface::NeighborSurface(unsigned int in_surface, unsigned int in_level)
    : BoundaryCondition(in_surface, in_level, Geometry.surface_extremal_coordinate[in_surface]),
    is_lower_interface(in_surface % 2 == 0) {
    global_partner_mpi_rank = (int)Geometry.get_global_neighbor_for_interface(Geometry.get_direction_for_boundary_id(in_surface)).second;
	local_partner_mpi_rank = (int)Geometry.get_global_neighbor_for_interface(Geometry.get_direction_for_boundary_id(in_surface)).first;
    dof_counter = 0;
}

NeighborSurface::~NeighborSurface() {

}

void NeighborSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed*, Constraints *) {
    matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
}

void NeighborSurface::fill_matrix(dealii::SparseMatrix<ComplexNumber> * matrix, Constraints *) {
	matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
}

void NeighborSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix *, NumericVectorDistributed *, Constraints *) {
	// Nothing to do here, work happens on neighbor process.
}

void NeighborSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed *, Constraints *) {
	matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
	// Nothing to do here, work happens on neighbor process.
}

bool NeighborSurface::is_point_at_boundary(Position2D, BoundaryId) {
	return false;
}

void NeighborSurface::initialize() {
	
}

std::vector<InterfaceDofData> NeighborSurface::get_dof_association() {
	std::vector<InterfaceDofData> dof_indices = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(b_id);
	for(unsigned int i = 0; i < dof_indices.size(); i++) {
		dof_indices[i].index = inner_dofs[i];
	}
	return dof_indices;
}

std::vector<InterfaceDofData> NeighborSurface::get_dof_association_by_boundary_id(BoundaryId in_boundary_id) {
	std::vector<InterfaceDofData> own_dof_indices;
	for(unsigned int i = 0; i < boundary_dofs[in_boundary_id].size(); i++) {
		InterfaceDofData idd;
		idd.order = 0;
		idd.base_point = {0,0,0};
		idd.index = boundary_dofs[in_boundary_id][i];
		own_dof_indices.push_back(idd);
	}
	return own_dof_indices;
}

std::string NeighborSurface::output_results(const dealii::Vector<ComplexNumber> & , std::string) {
	return "";
}

void NeighborSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern * , Constraints * ) {
}

DofCount NeighborSurface::compute_n_locally_owned_dofs() {
	return 0;
}

DofCount NeighborSurface::compute_n_locally_active_dofs() {
	return 0;
}


int NeighborSurface::generate_tag(unsigned int global_rank_sender, unsigned int receiver) {
	int ret = 0;
	srand(global_rank_sender);
	for(unsigned int i = 0; i < receiver; i++) {
		ret = rand()%Geometry.levels[level].n_total_level_dofs;
	}
	return ret;
}

void NeighborSurface::determine_non_owned_dofs() {

}

void NeighborSurface::finish_dof_index_initialization() {
	prepare_dofs();
	if(is_lower_interface) {
		receive();
	} else {
		send();
	}
}

void NeighborSurface::distribute_dof_indices() {
	if(is_lower_interface) {
		std::vector<InterfaceDofData> dof_data_inner = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(b_id);
		std::vector<DofNumber> dof_numbers;
		dof_numbers.resize(dof_data_inner.size());
		for(unsigned int i = 0; i < dof_data_inner.size(); i++) {
			dof_numbers[i] = dof_data_inner[i].index;
		}
		Geometry.levels[level].inner_domain->set_non_local_dof_indices(dof_numbers, inner_dofs);

		for(unsigned int surf = 0; surf < 6; surf++) {
			if(surf != b_id && !are_opposing_sites(surf, b_id)) {
				if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE) {
					std::vector<InterfaceDofData> dof_data = Geometry.levels[level].surfaces[surf]->get_dof_association_by_boundary_id(b_id);
					std::vector<unsigned int> dof_indices;
					for(unsigned int i = 0; i < dof_data.size(); i++) {
						dof_indices.push_back(dof_data[i].index);
					}										
					Geometry.levels[level].surfaces[surf]->set_non_local_dof_indices(dof_indices, boundary_dofs[surf]);
				}
			}
		}
	}
}

void NeighborSurface::prepare_dofs() {
	// For the lower process, i.e. the one where this layer is an upper boundary, I set the correct indices here. For the other process, I use the same code, but basically only calculate n_dofs, because the value arrays will be filled later by receive.
	std::vector<InterfaceDofData> temp = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(b_id);
	n_dofs = 0;
	inner_dofs.resize(temp.size());
	for(unsigned int i = 0; i < temp.size(); i++) {
		inner_dofs[i] = Geometry.levels[level].inner_domain->global_index_mapping[temp[i].index];
	}
	n_dofs += temp.size();
	for(unsigned int surf = 0; surf < 6; surf++) {
		boundary_dofs[surf].resize(0);
		if(surf != b_id && !are_opposing_sites(surf, b_id)) {
			if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE) {
				boundary_dofs[surf] = Geometry.levels[level].surfaces[surf]->get_global_dof_indices_by_boundary_id(b_id);
				n_dofs += boundary_dofs[surf].size();
			}
		}
	}
	dofs_prepared = true;
}

void NeighborSurface::send() {
	DofNumber * global_indices = new DofNumber[n_dofs];
	unsigned int counter = 0; 
	for(unsigned int i = 0; i < inner_dofs.size(); i++) {
		global_indices[counter] = inner_dofs[i];
		counter ++;
	}
	for(unsigned int surf = 0; surf < 6; surf++) {
		if(surf != b_id && !are_opposing_sites(surf, b_id)) {
			if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE) {
				for(unsigned int i = 0; i < boundary_dofs[surf].size(); i++) {
					global_indices[counter] = boundary_dofs[surf][i];
					counter ++;
				}
			}
		}
	}
	int tag = generate_tag(global_partner_mpi_rank, GlobalParams.MPI_Rank);
	MPI_Ssend(global_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, tag, MPI_COMM_WORLD);
	delete[] global_indices;
}

void NeighborSurface::receive() {
	unsigned int * dof_indices = new unsigned int [n_dofs];
	int tag = generate_tag(GlobalParams.MPI_Rank, global_partner_mpi_rank);
	MPI_Recv(dof_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, tag, MPI_COMM_WORLD, 0);
	unsigned int counter2 = 0; 
	for(unsigned int i = 0; i< n_dofs; i++) {
		if(dof_indices[i] > Geometry.levels[level].n_total_level_dofs) {
			counter2++;
		}
	}
	std::cout << "On " << GlobalParams.MPI_Rank << " surface " << b_id << " there were " << counter2 << " wrong dofs." << std::endl;
	unsigned int counter = 0;
	for(unsigned int i = 0; i < inner_dofs.size(); i++) {
		inner_dofs[i] = dof_indices[i];
		counter ++;
	}
	for(unsigned int surf = 0; surf < 6; surf++) {
		if(surf != b_id && !are_opposing_sites(surf, b_id)) {
			if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE) {
				for(unsigned int i = 0; i < boundary_dofs[surf].size(); i++) {
					boundary_dofs[surf][i] = dof_indices[counter];
					counter ++;
				}
			}
		}
	}
	distribute_dof_indices();
}
