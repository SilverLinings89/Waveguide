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
#include "../Solutions/PMLTransformedExactSolution.h"

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

bool DirichletSurface::is_point_at_boundary(Position2D, BoundaryId) {
	return false;
}

void DirichletSurface::initialize() {

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

Constraints DirichletSurface::make_constraints() {
	Constraints ret(Geometry.levels[level].inner_domain->global_dof_indices);
	dealii::IndexSet local_dof_set(Geometry.levels[level].inner_domain->n_locally_active_dofs);
	local_dof_set.add_range(0,Geometry.levels[level].inner_domain->n_locally_active_dofs);
	AffineConstraints<ComplexNumber> constraints_local(local_dof_set);
	ExactSolution es(true, false);
	VectorTools::project_boundary_values_curl_conforming_l2(Geometry.levels[level].inner_domain->dof_handler, 0, es, b_id, constraints_local);
	for(auto line : constraints_local.get_lines()) {
		const unsigned int local_index = line.index;
		const unsigned int global_index = Geometry.levels[level].inner_domain->global_index_mapping[local_index];
		ret.add_line(global_index);
		ret.set_inhomogeneity(global_index, line.inhomogeneity);
	}
	constraints_local.clear();
	for(unsigned int surf = 0; surf < 6; surf++) {
		if(surf != b_id && !are_opposing_sites(b_id, surf)) {
			if(Geometry.levels[level].surface_type[surf] == SurfaceType::ABC_SURFACE) {
				PMLTransformedExactSolution ptes(b_id, additional_coordinate);
				VectorTools::project_boundary_values_curl_conforming_l2(Geometry.levels[level].surfaces[surf]->dof_handler, 0, ptes, b_id, constraints_local);
				for(auto line : constraints_local.get_lines()) {
					const unsigned int local_index = line.index;
					const unsigned int global_index = Geometry.levels[level].surfaces[surf]->global_index_mapping[local_index];
					ret.add_line(global_index);
					ret.set_inhomogeneity(global_index, line.inhomogeneity);
				}
				constraints_local.clear();
			}
		}
	}
	return ret;
}
