#pragma once

#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/lac/affine_constraints.h>

/** 
 * \class EmptySurface
 * 
 * \brief This boundary condition implements a zero surface. It is required for the sweeping scheme where the upper boundary in sweeping-direction has 0-values.
 * 
 * The implementation is the same as the dirichlet surface but for a predefined zero function.
 * 
 */

class EmptySurface : public BoundaryCondition {
  public: 

    EmptySurface(unsigned int in_bid, unsigned int in_level);
    ~EmptySurface();

    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) override;
    void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constriants) override;
    bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;
    void initialize() override;
    auto get_dof_association() -> std::vector<InterfaceDofData> override;
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;
    std::string output_results(const dealii::Vector<ComplexNumber> & , std::string) override;
    DofCount compute_n_locally_owned_dofs() override;
    DofCount compute_n_locally_active_dofs() override;
    void determine_non_owned_dofs() override;
    auto make_constraints() -> Constraints override;
};