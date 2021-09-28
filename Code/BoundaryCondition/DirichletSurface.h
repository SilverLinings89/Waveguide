#pragma once

#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/lac/affine_constraints.h>

class DirichletSurface : public BoundaryCondition {
  public: 

    DirichletSurface(unsigned int in_bid, unsigned int in_level);
    ~DirichletSurface();
    
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) override;
    void fill_matrix(dealii::SparseMatrix<ComplexNumber>*,  Constraints *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) override;
    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) override;
    void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constriants) override;
    bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;
    void initialize() override;
    auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount override;
    auto get_dof_association() -> std::vector<InterfaceDofData> override;
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;
    std::string output_results(const dealii::Vector<ComplexNumber> & , std::string) override;
    DofCount compute_n_locally_owned_dofs() override;
    DofCount compute_n_locally_active_dofs() override;
    void determine_non_owned_dofs() override;
    auto make_constraints() -> Constraints override;
};