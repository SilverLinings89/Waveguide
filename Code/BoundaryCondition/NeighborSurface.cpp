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
    is_primary(in_surface > 2),
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
    std::vector<InterfaceDofData> own_dof_indices = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(b_id);
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
        own_dof_indices = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_edge_and_level(b_id, in_boundary_id, level);
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

std::string NeighborSurface::output_results(const dealii::Vector<ComplexNumber> & , std::string) {
    return "";
}

void NeighborSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern * in_dsp, Constraints * in_constraints) {
    BoundaryCondition::fill_sparsity_pattern(in_dsp, in_constraints);
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

DofCount NeighborSurface::compute_n_locally_owned_dofs(std::array<bool, 6> is_locally_owned_surfac) {
    return 0;
}

DofCount NeighborSurface::compute_n_locally_active_dofs() {
    return 0;
}
