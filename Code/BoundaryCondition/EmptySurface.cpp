#include "./EmptySurface.h"
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

EmptySurface::EmptySurface(unsigned int in_surface, unsigned int in_level)
    : BoundaryCondition(in_surface, in_level, Geometry.surface_extremal_coordinate[in_level])
    {
    dof_counter = 0;
}

EmptySurface::~EmptySurface() {

}

void EmptySurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed*, Constraints *) {
    matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
}

void EmptySurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix *, NumericVectorDistributed *, Constraints *) {
    // Nothing to do here, work happens on neighbor process.
}

void EmptySurface::fill_matrix(dealii::SparseMatrix<ComplexNumber> * matrix, Constraints *) {
    matrix->compress(dealii::VectorOperation::add);
    // Nothing to do here, work happens on neighbor process.
}

void EmptySurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed *, Constraints *) {
     matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
    // Nothing to do here, work happens on neighbor process.
}

void EmptySurface::identify_corner_cells() {

}

bool EmptySurface::is_point_at_boundary(Position2D, BoundaryId) {
    return false;
}

void EmptySurface::initialize() {

}

unsigned int EmptySurface::get_dof_count_by_boundary_id(BoundaryId) {
    return 0;
}

std::vector<InterfaceDofData> EmptySurface::get_dof_association() {
    std::vector<InterfaceDofData> ret;
    return ret;
}

std::vector<InterfaceDofData> EmptySurface::get_dof_association_by_boundary_id(BoundaryId) {
    std::vector<InterfaceDofData> ret;
    return ret;
}

std::string EmptySurface::output_results(const dealii::Vector<ComplexNumber> & , std::string) {
    return "";
}

void EmptySurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern * , Constraints * ) { }

DofCount EmptySurface::compute_n_locally_owned_dofs(std::array<bool, 6> is_locally_owned_surfac) {
    return 0;
}

DofCount EmptySurface::compute_n_locally_active_dofs() {
    return 0;
}