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

EmptySurface::EmptySurface(unsigned int in_surface, unsigned int in_level, DofNumber in_first_own_index)
    : BoundaryCondition(in_surface, in_level, Geometry.surface_extremal_coordinate[in_level], in_first_own_index)
    {
    dof_counter = 0;
}

EmptySurface::~EmptySurface() {

}

void EmptySurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed*, dealii::AffineConstraints<ComplexNumber> *) {
    matrix->compress(dealii::VectorOperation::add); // <-- this operation is collective and therefore required.
}

void EmptySurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix *, NumericVectorDistributed *, dealii::AffineConstraints<ComplexNumber> *) {
    // Nothing to do here, work happens on neighbor process.
}

void EmptySurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed *, dealii::AffineConstraints<ComplexNumber> *) {
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

void EmptySurface::output_results(const dealii::Vector<ComplexNumber> & , std::string) {

}

void EmptySurface::make_surface_constraints(dealii::AffineConstraints<ComplexNumber> * ) { }

void EmptySurface::make_edge_constraints(dealii::AffineConstraints<ComplexNumber> *, BoundaryId) { }

std::vector<SurfaceCellData> EmptySurface::get_surface_cell_data(BoundaryId) {
    std::vector<SurfaceCellData> ret;
    return ret;
}

std::vector<SurfaceCellData> EmptySurface::get_inner_surface_cell_data() {
    std::vector<SurfaceCellData> data;
    return data;
}

void EmptySurface::fill_internal_sparsity_pattern(dealii::DynamicSparsityPattern *, dealii::AffineConstraints<ComplexNumber> *) {
    // Only the other boundary methods do something here, because this one has no "own" dofs.
}

std::vector<SurfaceCellData> EmptySurface::get_corner_surface_cell_data(BoundaryId, BoundaryId) {
    std::vector<SurfaceCellData> data;
    return data;
}

void EmptySurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern * , dealii::AffineConstraints<ComplexNumber> * ) { }
