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
    is_primary(in_surface % 2 == 0),
    global_partner_mpi_rank(Geometry.get_global_neighbor_for_interface(Geometry.get_direction_for_boundary_id(in_surface)).second),
    partner_mpi_rank_in_level_communicator(Geometry.get_level_neighbor_for_interface(Geometry.get_direction_for_boundary_id(in_surface), in_level).second) {
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

void NeighborSurface::prepare_id_sets_for_boundaries() {

}

bool NeighborSurface::is_point_at_boundary(Position, BoundaryId) {
	// Whs
	return false;
}

void NeighborSurface::identify_corner_cells() {

}

bool NeighborSurface::is_point_at_boundary(Position2D, BoundaryId) {
	return false;
}

bool NeighborSurface::is_position_at_boundary(Position, BoundaryId) {
	// Why
	return false;
}

void NeighborSurface::initialize() {
	for(unsigned int surf = 0; surf < 6; surf++) {

	}
}

unsigned int NeighborSurface::cells_for_boundary_id(unsigned int) {
	return 0;
}

void NeighborSurface::init_fe(){

}

unsigned int NeighborSurface::get_dof_count_by_boundary_id(BoundaryId) {
	return 0;
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

void NeighborSurface::sort_dofs() {

}

void NeighborSurface::compute_coordinate_ranges() {

}

void NeighborSurface::set_boundary_ids() {

}

std::string NeighborSurface::output_results(const dealii::Vector<ComplexNumber> & , std::string) {
	return "";
}

void NeighborSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern * in_dsp, Constraints * in_constraints) {
}

std::array<std::pair<BoundaryId, BoundaryId>, 4> NeighborSurface::get_corner_boundary_id_set() {
  std::array<std::pair<BoundaryId, BoundaryId>, 4> corners;
  if(b_id == 0 || b_id == 1) {
		corners[0] = std::pair<BoundaryId, BoundaryId>(2,4);
		corners[1] = std::pair<BoundaryId, BoundaryId>(3,4);
		corners[2] = std::pair<BoundaryId, BoundaryId>(2,5);
		corners[3] = std::pair<BoundaryId, BoundaryId>(3,5);
  }
  if(b_id == 2 || b_id == 3) {
		corners[0] = std::pair<BoundaryId, BoundaryId>(0,4);
		corners[1] = std::pair<BoundaryId, BoundaryId>(1,4);
		corners[2] = std::pair<BoundaryId, BoundaryId>(0,5);
		corners[3] = std::pair<BoundaryId, BoundaryId>(1,5);
  }
  if(b_id == 4 || b_id == 5) {
		corners[0] = std::pair<BoundaryId, BoundaryId>(0,2);
		corners[1] = std::pair<BoundaryId, BoundaryId>(1,2);
		corners[2] = std::pair<BoundaryId, BoundaryId>(0,3);
		corners[3] = std::pair<BoundaryId, BoundaryId>(1,3);
  }
  return corners;
}


DofCount NeighborSurface::compute_n_locally_owned_dofs() {
	return 0;
}

DofCount NeighborSurface::compute_n_locally_active_dofs() {
	return 0;
}

void NeighborSurface::send_up_inner_dofs() {
	std::vector<InterfaceDofData> dofs = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(b_id);
	const unsigned int n_dofs = dofs.size();
	DofIndexVector local_indices(n_dofs);
	for(unsigned int i = 0; i < n_dofs; i++){
		local_indices[i] = dofs[i].index;
	}
	local_indices = Geometry.levels[level].inner_domain->transform_local_to_global_dofs(local_indices);
	unsigned int * local_indices_buffer = new unsigned int[n_dofs];
	for(unsigned int i = 0; i < n_dofs; i++) {
		local_indices_buffer[i] = local_indices[i];
	}
	MPI_Send(local_indices_buffer, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, 0, MPI_COMM_WORLD);
}

void NeighborSurface::receive_from_below_dofs() {
	std::vector<InterfaceDofData> dofs = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(b_id);
	const unsigned int n_dofs = dofs.size();
	DofIndexVector local_indices(n_dofs);
	for(unsigned int i = 0; i < n_dofs; i++){
			local_indices[i] = dofs[i].index;
	} 
	unsigned int * dof_indices = new unsigned int[n_dofs];
	MPI_Recv(dof_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0);
	DofIndexVector global_indices(n_dofs);
	for(unsigned int i = 0; i < n_dofs; i++){
			global_indices[i] = dof_indices[i];
	}
	Geometry.levels[level].inner_domain->set_non_local_dof_indices(local_indices, global_indices);
}

void NeighborSurface::determine_non_owned_dofs() {

}

void NeighborSurface::send_up_boundary_dofs(unsigned int other_bid) {
	if(other_bid != b_id && !are_opposing_sites(other_bid, b_id)) {
		if(Geometry.levels[level].surface_type[other_bid] == SurfaceType::ABC_SURFACE) {
			std::vector<InterfaceDofData> dof_data = Geometry.levels[level].surfaces[other_bid]->get_dof_association_by_boundary_id(b_id);
			const unsigned int n_dofs = dof_data.size();
			std::vector<DofNumber> indices(n_dofs);
			for(unsigned int i = 0; i < n_dofs; i++) {
				indices[i] = dof_data[i].index;
			}
			indices = Geometry.levels[level].surfaces[other_bid]->transform_local_to_global_dofs(indices);
			unsigned int * global_indices = new unsigned int [n_dofs];
			for(unsigned int i = 0; i < n_dofs; i++) {
				global_indices[i] = indices[i];
			}
			MPI_Send(global_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, 0, MPI_COMM_WORLD);
		}
	}
}

std::vector<DofNumber> NeighborSurface::receive_boundary_dofs(unsigned int other_bid) {
	std::vector<DofNumber> ret;
	const unsigned int n_dofs = Geometry.levels[level].surfaces[other_bid]->get_dof_association_by_boundary_id(b_id).size();
	unsigned int * dof_indices = new unsigned int [n_dofs];
	MPI_Recv(dof_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0);
	ret.resize(n_dofs);
	for(unsigned int i = 0; i < n_dofs; i++) {
		ret[i] = dof_indices[i];
	}
	return ret;
}

void NeighborSurface::finish_dof_index_initialization() {
	if(is_primary) {
		// this interface does not own, so it receives
		receive_from_below_dofs();
		for(unsigned int surf = 0; surf < 6; surf++) {
			if(surf != b_id && !are_opposing_sites(surf, b_id)) {
				if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE) {
					boundary_dofs[surf] = receive_boundary_dofs(surf);
				}
			}
		}
	} else {
		// this interface does own, so it sends
		send_up_inner_dofs();
		for(unsigned int surf = 0; surf < 6; surf++) {
			if(surf != b_id && !are_opposing_sites(surf, b_id)) {
				if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE) {
						send_up_boundary_dofs(surf);
				}
			}
		}
	}
	distribute_dof_indices();
}

void NeighborSurface::distribute_dof_indices() {
	if(is_primary) {
		for(unsigned int surf = 0; surf < 6; surf++) {
			if(surf != b_id && !are_opposing_sites(b_id, surf)) {
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
