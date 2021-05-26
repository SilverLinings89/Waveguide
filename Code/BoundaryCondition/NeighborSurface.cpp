#include "./NeighborSurface.h"
#include <deal.II/base/geometry_info.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include "../Core/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include "../Core/NumericProblem.h"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <mpi.h>
#include <cstdlib>
#include <string>
#include "./BoundaryCondition.h"
#include "PMLMeshTransformation.h"

NeighborSurface::NeighborSurface(unsigned int in_surface, unsigned int in_level, DofNumber in_first_own_index)
    : BoundaryCondition(in_surface, in_level, Geometry.surface_extremal_coordinate[in_level], in_first_own_index),
    is_primary(in_surface > 2),
    global_partner_mpi_rank(Geometry.get_global_neighbor_for_interface(Geometry.get_direction_for_boundary_id(in_surface)).second),
    partner_mpi_rank_in_level_communicator(Geometry.get_level_neighbor_for_interface(Geometry.get_direction_for_boundary_id(in_surface), in_level).second) {
    dof_counter = 0;
}

NeighborSurface::~NeighborSurface() {

}

void NeighborSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed*, dealii::AffineConstraints<ComplexNumber> *) {
    matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
}

void NeighborSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix *, NumericVectorDistributed *, dealii::AffineConstraints<ComplexNumber> *) {
    // Nothing to do here, work happens on neighbor process.
}

void NeighborSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed *, dealii::AffineConstraints<ComplexNumber> *) {
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

}

void NeighborSurface::set_mesh_boundary_ids() {

}

void NeighborSurface::prepare_mesh(){

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
    std::vector<InterfaceDofData> own_dof_indices = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(b_id, level);
    const unsigned int n_dofs = own_dof_indices.size();
    unsigned int * dof_indices = new unsigned int[n_dofs];
    for(unsigned int i = 0; i < n_dofs; i++) {
        dof_indices[i] = own_dof_indices[i].index;
    }
    MPI_Sendrecv_replace(dof_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, 0, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0 );
    std::vector<InterfaceDofData> other_dof_indices;
    for(unsigned int i = 0; i < n_dofs; i++) {
        InterfaceDofData temp = own_dof_indices[i];
        temp.index = dof_indices[i];
        other_dof_indices.push_back(temp);
    }
    return other_dof_indices;
}

std::vector<InterfaceDofData> NeighborSurface::get_dof_association_by_boundary_id(BoundaryId in_boundary_id) {
    std::vector<InterfaceDofData> own_dof_indices;
    if(Geometry.levels[level].is_surface_truncated[in_boundary_id]) {
        own_dof_indices = Geometry.levels[level].surfaces[in_boundary_id]->get_dof_association_by_boundary_id(b_id);
    } else {
        own_dof_indices = Geometry.inner_domain->get_surface_dof_vector_for_edge_and_level(b_id, in_boundary_id, level);
    }
    const unsigned int n_dofs = own_dof_indices.size();
    unsigned int * dof_indices = new unsigned int[n_dofs];
    for(unsigned int i = 0; i < n_dofs; i++) {
        dof_indices[i] = own_dof_indices[i].index;
    }
    MPI_Sendrecv_replace(dof_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, 0, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0 );
    std::vector<InterfaceDofData> other_dof_indices;
    for(unsigned int i = 0; i < n_dofs; i++) {
        InterfaceDofData temp = own_dof_indices[i];
        temp.index = dof_indices[i];
        other_dof_indices.push_back(temp);
    }
    return other_dof_indices;
}

void NeighborSurface::sort_dofs() {

}

void NeighborSurface::compute_coordinate_ranges() {

}

void NeighborSurface::set_boundary_ids() {

}

void NeighborSurface::output_results(const dealii::Vector<ComplexNumber> & , std::string) {

}

void NeighborSurface::make_surface_constraints(dealii::AffineConstraints<ComplexNumber> * constraints) {
    std::vector<InterfaceDofData> own_dof_indices = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(b_id, level);
    const unsigned int n_dofs = own_dof_indices.size();
    unsigned int * dof_indices = new unsigned int[n_dofs];
    for(unsigned int i = 0; i < n_dofs; i++) {
        dof_indices[i] = own_dof_indices[i].index;
    }
    MPI_Sendrecv_replace(dof_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, 0, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0 );
    std::vector<InterfaceDofData> other_dof_indices;
    for(unsigned int i = 0; i < n_dofs; i++) {
        InterfaceDofData temp = own_dof_indices[i];
        temp.index = dof_indices[i];
        other_dof_indices.push_back(temp);
    }
    dealii::AffineConstraints<ComplexNumber> new_constraints = get_affine_constraints_for_InterfaceData(own_dof_indices, other_dof_indices, Geometry.levels[level].n_total_level_dofs);
    constraints->merge(new_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
}

void NeighborSurface::make_edge_constraints(dealii::AffineConstraints<ComplexNumber> * constraints, BoundaryId other_boundary) {
    std::vector<InterfaceDofData> own_dof_indices;
    if(Geometry.levels[level].is_surface_truncated[other_boundary]) {
        own_dof_indices = Geometry.levels[level].surfaces[other_boundary]->get_dof_association_by_boundary_id(b_id);
    } else {
        own_dof_indices = Geometry.inner_domain->get_surface_dof_vector_for_edge_and_level(b_id, other_boundary, level);
    }
    const unsigned int n_dofs = own_dof_indices.size();
    unsigned int * dof_indices = new unsigned int[n_dofs];
    for(unsigned int i = 0; i < n_dofs; i++) {
        dof_indices[i] = own_dof_indices[i].index;
    }
    MPI_Sendrecv_replace(dof_indices, n_dofs, MPI_UNSIGNED, global_partner_mpi_rank, 0, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0 );
    std::vector<InterfaceDofData> other_dof_indices;
    for(unsigned int i = 0; i < n_dofs; i++) {
        InterfaceDofData temp = own_dof_indices[i];
        temp.index = dof_indices[i];
        other_dof_indices.push_back(temp);
    }
    dealii::AffineConstraints<ComplexNumber> new_constraints = get_affine_constraints_for_InterfaceData(own_dof_indices, other_dof_indices, Geometry.levels[level].n_total_level_dofs);
    constraints->merge(new_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
}

std::vector<SurfaceCellData> NeighborSurface::get_surface_cell_data(BoundaryId in_bid) {
    std::vector<SurfaceCellData> data;
    if(Geometry.levels[level].is_surface_truncated[in_bid]) {
        data = Geometry.levels[level].surfaces[in_bid]->get_surface_cell_data(b_id);
    } else {
        return data;
    }
    return mpi_send_recv_surf_cell_data(data);
}

std::vector<SurfaceCellData> NeighborSurface::get_inner_surface_cell_data() {
    std::vector<SurfaceCellData> data = Geometry.inner_domain->get_surface_cell_data_for_boundary_id_and_level(b_id, level);
    return mpi_send_recv_surf_cell_data(data);
}

void NeighborSurface::fill_internal_sparsity_pattern(dealii::DynamicSparsityPattern *, dealii::AffineConstraints<ComplexNumber> *) {
    // Only the other boundary methods do something here, because this one has no "own" dofs.
}

std::vector<SurfaceCellData> NeighborSurface::get_corner_surface_cell_data(BoundaryId b_id_one, BoundaryId b_id_two) {
    std::vector<SurfaceCellData> data = Geometry.levels[level].surfaces[b_id_two]->get_corner_surface_cell_data(b_id_one, b_id);
    return mpi_send_recv_surf_cell_data(data);
}

void NeighborSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern * in_dsp, dealii::AffineConstraints<ComplexNumber> * in_constraints) {
    BoundaryCondition::fill_sparsity_pattern(in_dsp, in_constraints);
    fill_sparsity_pattern_for_edge(in_dsp, in_constraints);
    fill_sparsity_pattern_for_corners(in_dsp, in_constraints);
}

void NeighborSurface::fill_sparsity_pattern_for_edge(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> * in_constraints) {
    for(unsigned int i = 0; i < 6; i++) {
        if(!(b_id == i || are_opposing_sites(i, b_id))) {
            fill_sparsity_pattern_for_edge_and_neighbor(i, in_dsp, in_constraints);
        }
    }
}

void NeighborSurface::fill_sparsity_pattern_for_edge_and_neighbor(BoundaryId edge_bid, dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> * in_constraints) {
    BoundaryId edge_bid_opponent = 6;
    for(unsigned int i = 0; i < 6; i++) {
        if(are_opposing_sites(i, edge_bid)) {
            edge_bid_opponent = i;
        }
    }
    
    std::vector<SurfaceCellData> from_self = Geometry.inner_domain->get_edge_cell_data(b_id, edge_bid, level);
    std::vector<SurfaceCellData> from_other =  mpi_send_recv_surf_cell_data(from_self);
    from_self = Geometry.levels[level].surfaces[edge_bid]->get_corner_surface_cell_data(edge_bid_opponent, b_id);
    fill_sparsity_pattern_with_surface_data_vectors(from_self, from_other, in_dsp, in_constraints);

    from_self = Geometry.levels[level].surfaces[edge_bid]->get_corner_surface_cell_data(edge_bid_opponent, b_id);
    from_other = mpi_send_recv_surf_cell_data(from_self);
    from_self = Geometry.inner_domain->get_edge_cell_data(b_id, edge_bid, level);
    fill_sparsity_pattern_with_surface_data_vectors(from_self, from_other, in_dsp, in_constraints);

}

void NeighborSurface::fill_sparsity_pattern_for_corner(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints, BoundaryId in_b_id_one, BoundaryId in_b_id_two) {
  std::vector<SurfaceCellData> from_self = Geometry.levels[level].surfaces[in_b_id_one]->get_corner_surface_cell_data(in_b_id_two, b_id);
  std::vector<SurfaceCellData> from_other = mpi_send_recv_surf_cell_data(from_self);
  from_self = Geometry.levels[level].surfaces[in_b_id_two]->get_corner_surface_cell_data(in_b_id_one, b_id);
  fill_sparsity_pattern_with_surface_data_vectors(from_self, from_other, in_dsp, constraints);
}

void NeighborSurface::fill_sparsity_pattern_for_corners(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints) {
    std::array<std::pair<BoundaryId, BoundaryId>, 4> corners = get_corner_boundary_id_set();
    for(unsigned int i = 0; i < 4; i++) {
        fill_sparsity_pattern_for_corner(in_dsp, constraints, corners[i].first, corners[i].second);
        fill_sparsity_pattern_for_corner(in_dsp, constraints, corners[i].second, corners[i].first);
    }
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

std::vector<SurfaceCellData> NeighborSurface::mpi_send_recv_surf_cell_data(std::vector<SurfaceCellData> in_data) {
    std::vector<SurfaceCellData> ret = in_data;
    if(in_data.size() > 0) {
        unsigned int * meta_data = new unsigned int[2];
        meta_data[0] = in_data.size();  // N cells
        meta_data[1] = in_data[0].dof_numbers.size(); // N dofs per cell
        MPI_Sendrecv_replace(meta_data, 2, MPI_UNSIGNED, global_partner_mpi_rank, 0, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0 );
        if(meta_data[0] != in_data.size()) {
            // Incompatible numbers of cells. Cannot couple.
            std::cout << "Incompatible cell counts in mpi_send_recv_surf_cell_data" << std::endl;
            exit(0);
        }
        std::vector<unsigned int> indices = dof_indices_from_surface_cell_data(ret);
        if(meta_data[1] != in_data[0].dof_numbers.size()) {
            unsigned int com_buffer_size = (in_data[0].dof_numbers.size() > meta_data[1])?in_data[0].dof_numbers.size(): meta_data[1];
            unsigned int * dof_indices = new unsigned int[com_buffer_size];
            for(unsigned int i = 0; i < indices.size(); i++) {
                dof_indices[i] = indices[i];
            }
            for(unsigned int i = indices.size(); i < com_buffer_size; i++) {
                dof_indices[i] = 0;
            }
            MPI_Sendrecv_replace(dof_indices, com_buffer_size, MPI_UNSIGNED, global_partner_mpi_rank, 0, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0 );
            for(unsigned int i = 0; i < ret.size(); i++) {
                ret[i].dof_numbers.clear();
                for(unsigned int j = 0; j < meta_data[1]; j++) {
                    ret[i].dof_numbers.push_back(dof_indices[i * meta_data[1] + j]);
                }
            }
            delete [] dof_indices;
            return ret;
        } else {
            unsigned int * dof_indices = new unsigned int[indices.size()];
            for(unsigned int i = 0; i < indices.size(); i++) {
                dof_indices[i] = indices[i];
            }
            MPI_Sendrecv_replace(dof_indices, indices.size(), MPI_UNSIGNED, global_partner_mpi_rank, 0, global_partner_mpi_rank, 0, MPI_COMM_WORLD, 0 );
            const unsigned int n_dofs_per_cell = ret[0].dof_numbers.size();
            for(unsigned int i = 0; i < ret.size(); i++) {
                for(unsigned int j = 0; j < n_dofs_per_cell; j++) {
                    ret[i].dof_numbers[j] = dof_indices[i*n_dofs_per_cell + j];
                }
            }
            delete [] dof_indices;
            return ret;
        }
    } else {
        std::cout << "No data provided in mpi_send_recv_surf_cell_data" << std::endl;
    }
    return ret;
}
