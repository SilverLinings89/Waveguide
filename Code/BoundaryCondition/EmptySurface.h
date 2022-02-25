#pragma once

#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/lac/affine_constraints.h>

/** 
 * \class EmptySurface
 * 
 * \brief A surface with tangential component of the solution equals zero, i.e. specialization of the dirichlet surface
 * 
 * \details This is a DirichletSurface with a predefined soltuion to enforce - namely zero. It is used in the sweeping preconditioning scheme where the lower boundary dofs of all domains except the lowest in sweeping direction are set to zero to compute the rhs that acurately describes the signal propagating across the interface.
 * The implementation is extremely simple because most functions perform no tasks at all and the make_constraints() function is a simplified version of the version in DirichletSurface. The members of this class are therefore not documented. See the documentation in the base class for more details.
 * \see DirichletSurface, BoundaeyCondition
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