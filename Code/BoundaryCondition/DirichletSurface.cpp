#include "./DirichletSurface.h"
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

DirichletSurface::DirichletSurface(unsigned int in_surface, unsigned int in_level)
    : BoundaryCondition(in_surface, in_level, Geometry.surface_extremal_coordinate[in_level])
    {
    dof_counter = 0;
}

DirichletSurface::~DirichletSurface() {

}

void DirichletSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed*, Constraints *) {
    matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
}

void DirichletSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix *, NumericVectorDistributed *, Constraints *) {
    // Nothing to do here, work happens on neighbor process.
}

void DirichletSurface::fill_matrix(dealii::SparseMatrix<ComplexNumber> * matrix, Constraints *) {
    matrix->compress(dealii::VectorOperation::add);
    // Nothing to do here, work happens on neighbor process.
}

void DirichletSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed *, Constraints *) {
     matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
    // Nothing to do here, work happens on neighbor process.
}

void DirichletSurface::identify_corner_cells() {

}

bool DirichletSurface::is_point_at_boundary(Position2D, BoundaryId) {
    return false;
}

void DirichletSurface::initialize() {

}

unsigned int DirichletSurface::get_dof_count_by_boundary_id(BoundaryId) {
    return 0;
}

std::vector<InterfaceDofData> DirichletSurface::get_dof_association() {
    std::vector<InterfaceDofData> ret;
    return ret;
}

std::vector<InterfaceDofData> DirichletSurface::get_dof_association_by_boundary_id(BoundaryId) {
    std::vector<InterfaceDofData> ret;
    return ret;
}

std::string DirichletSurface::output_results(const dealii::Vector<ComplexNumber> & , std::string) {
    return "";
}

void DirichletSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern * , Constraints * ) { }

DofCount DirichletSurface::compute_n_locally_owned_dofs() {
    return 0;
}

DofCount DirichletSurface::compute_n_locally_active_dofs() {
    return 0;
}

void DirichletSurface::determine_non_owned_dofs() {

}
