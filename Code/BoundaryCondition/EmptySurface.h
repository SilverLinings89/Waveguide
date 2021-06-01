#pragma once

#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/lac/affine_constraints.h>

class EmptySurface : public BoundaryCondition {
  public: 

    EmptySurface(unsigned int in_bid, unsigned int in_level, DofNumber first_own_index);
    ~EmptySurface();
    
    void identify_corner_cells() override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> * in_constriants) override;
    bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;
    void initialize() override;
    auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount override;
    auto get_dof_association() -> std::vector<InterfaceDofData> override;
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;
    void output_results(const dealii::Vector<ComplexNumber> & , std::string) override;
    void make_surface_constraints(dealii::AffineConstraints<ComplexNumber> * constraints) override;
    void make_edge_constraints(dealii::AffineConstraints<ComplexNumber> * constraints, BoundaryId other_boundary) override;
    auto get_surface_cell_data(BoundaryId in_bid) -> std::vector<SurfaceCellData> override;
    auto get_corner_surface_cell_data(BoundaryId main_boundary, BoundaryId secondary_boundary) -> std::vector<SurfaceCellData> override;
    auto get_inner_surface_cell_data() -> std::vector<SurfaceCellData> override;
    void fill_internal_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> * in_constriants) override;
};