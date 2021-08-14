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

DirichletSurface::DirichletSurface(unsigned int in_surface, unsigned int in_level, DofNumber in_first_own_index)
    : BoundaryCondition(in_surface, in_level, Geometry.surface_extremal_coordinate[in_level], in_first_own_index)
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

void DirichletSurface::make_surface_constraints(Constraints * constraints, bool make_inhomogeneities) {
    IndexSet owned_dofs(Geometry.inner_domain->dof_handler.n_dofs());
    owned_dofs.add_range(0, Geometry.inner_domain->dof_handler.n_dofs());
    AffineConstraints<ComplexNumber> constraints_local(owned_dofs);
    VectorTools::project_boundary_values_curl_conforming_l2(Geometry.inner_domain->dof_handler, 0, *GlobalParams.source_field , b_id, constraints_local);
    constraints_local.shift(first_own_dof);
    constraints->merge(constraints_local, Constraints::MergeConflictBehavior::right_object_wins, true);
}

void DirichletSurface::make_edge_constraints(Constraints *, BoundaryId) { }

std::vector<SurfaceCellData> DirichletSurface::get_surface_cell_data(BoundaryId) {
    std::vector<SurfaceCellData> ret;
    return ret;
}

std::vector<SurfaceCellData> DirichletSurface::get_inner_surface_cell_data() {
    std::vector<SurfaceCellData> data;
    return data;
}

void DirichletSurface::fill_internal_sparsity_pattern(dealii::DynamicSparsityPattern *, Constraints *) {
    // Only the other boundary methods do something here, because this one has no "own" dofs.
}

std::vector<SurfaceCellData> DirichletSurface::get_corner_surface_cell_data(BoundaryId, BoundaryId) {
    std::vector<SurfaceCellData> data;
    return data;
}

void DirichletSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern * , Constraints * ) { }
