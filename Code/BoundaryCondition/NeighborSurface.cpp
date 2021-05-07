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

void NeighborSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed*, dealii::AffineConstraints<ComplexNumber> *) {
    // Nothing to do here, work happens on neighbor process.
}

void NeighborSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix *, NumericVectorDistributed *, dealii::AffineConstraints<ComplexNumber> *) {
    // Nothing to do here, work happens on neighbor process.
}

void NeighborSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed *, dealii::AffineConstraints<ComplexNumber> *) {
    // Nothing to do here, work happens on neighbor process.
}

void NeighborSurface::prepare_id_sets_for_boundaries() {

}

bool NeighborSurface::is_point_at_boundary(Position, BoundaryId) {
    return false;
}

void NeighborSurface::identify_corner_cells() {

}

void NeighborSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints) {

}

bool NeighborSurface::is_point_at_boundary(Position2D in_p, BoundaryId in_bid) {

}

bool NeighborSurface::is_position_at_boundary(Position in_p, BoundaryId in_bid) {

}

void NeighborSurface::initialize() {

}

void NeighborSurface::set_mesh_boundary_ids() {

}

void NeighborSurface::prepare_mesh(){

}

unsigned int NeighborSurface::cells_for_boundary_id(unsigned int boundary_id) {
    return 0;
}

void NeighborSurface::init_fe(){

}

unsigned int NeighborSurface::get_dof_count_by_boundary_id(BoundaryId in_boundary_id) {
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

void NeighborSurface::fill_sparsity_pattern_for_boundary_id(const BoundaryId in_bid, dealii::AffineConstraints<ComplexNumber> * constraints, dealii::DynamicSparsityPattern * dsp) {

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
