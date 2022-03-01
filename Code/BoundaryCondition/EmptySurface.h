#pragma once

#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/lac/affine_constraints.h>

/** 
 * \class EmptySurface
 * 
 * \brief A surface with tangential component of the solution equals zero, i.e. specialization of the dirichlet surface. 
 * 
 * \details This is a DirichletSurface with a predefined soltuion to enforce - namely zero, i.e. a PEC boundary condition. It is used in the sweeping preconditioning scheme where the lower boundary dofs of all domains except the lowest in sweeping direction are set to zero to compute the rhs that acurately describes the signal propagating across the interface.
 * The implementation is extremely simple because most functions perform no tasks at all and the make_constraints() function is a simplified version of the version in DirichletSurface. The members of this class are therefore not documented. See the documentation in the base class for more details.
 * \see DirichletSurface, BoundaeyCondition
 * 
 */

class EmptySurface : public BoundaryCondition {
  public: 

    EmptySurface(unsigned int in_bid, unsigned int in_level);
    ~EmptySurface();

    /**
     *  @brief Fill a system matrix
     *
     *  @details See class description.
     * 
     *  @see   EmptySurface::make_constraints()
     * 
     *  @param matrix only for the interface
     *  @param rhs only for the interface
     *  @param constraints only for the interface
     */
    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, Constraints *constraints) override;

    /**
     *  @brief Fill the sparsity pattern
     *
     *  @details See class description.
     * 
     *  @see   EmptySurface::make_constratints()
     * 
     *  @param in_dsp the sparsity pattern to fill
     *  @param in_constraints the constraint object to be considered when writing the sparsity pattern
     */
    void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constraints) override;
    
    /**
     *  @brief Checks if a 2D surface coordinate is on a surface of not.
     *
     *  @details See the description in the base class.
     * 
     *  @param in_p the position to be checked
     *  @param in_bid This function does NOT return the boundary the point is on. Instead, it checks if it is on the boundary provided in this argument and returns true or false
     *  @return boolean indicating if the provided position is on the provided surface
     */
    bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;

    /**
     *  @brief Performs initialization of datastructures. Does nothing for this version of a boundary condition.
     *
     *  @details See the description in the base class.
     */
    void initialize() override;

    /**
     *  @brief returns an empty array.
     *
     *  @details While this boundary condition does influence some degree of freedom values, it does not own any. Surface dofs are always owned by the interior domain and dirichlet surfaces introduce no artificial dofs like HSIE or PML. As a consequence, this object does not store any dof data at all and instead gets a vector of surface dofs from the interior when required.
     * 
     *  @return The returned array is empty.
     */
    auto get_dof_association() -> std::vector<InterfaceDofData> override;

    /**
     *  @brief returns an empty array.
     *
     *  @details See function above.
     * 
     *  @see get_dof_association()
     * 
     *  @param in_boundary_id NOT USED.
     *  @return empty vector of InterfaceDofData type because this boundary condition has no own degrees of freedom.
     */
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;

     /**
     *  @brief Would write output but this function has no own data to store.
     *
     *  @details This function performs no actions. See class and base class description for details.
     * 
     *  @param solution NOT USED.
     *  @param filename NOT USED.
     *  @return     
     */
    std::string output_results(const dealii::Vector<ComplexNumber> & solution, std::string filename) override;

    /**
     *  @brief Computes the number of degrees of freedom that this surface owns which is 0 for empty surfaces.
     *
     *  @details Returns 0. See class description.
  
     *  @return 0.
     */
    DofCount compute_n_locally_owned_dofs() override;

    /**
     *  @brief There are active dofs on this surface. However, empty surfaces never interact with them (Empty surfaces are only active in the phase when constraints are built, bot when matrices are assembled or solutions written to an output). As a consequence, the output of this function is 0.
     *
     *  @details Returns 0. See class description.
  
     *  @return 0.
     */
    DofCount compute_n_locally_active_dofs() override;

     /**
     *  @brief Only exists for the interface. Does nothing.
     * 
     *  @details The surface owns no dofs.
     * 
     */
    void determine_non_owned_dofs() override;

    /**
     *  @brief Writes the constraints of locally active being equal to zero into a contstrint object and returns it.
     *
     *  @details This is the only function on this type that does something. It projects zero values onto the inner domains surface and builds a AffineConstraints<ComplexNumber> object from the resulting values. The object it returns can be merged with other objects of the same type to build the global constraint object.
  
     *  @return A constraint object representing the PEC boundary data.
     */
    auto make_constraints() -> Constraints override;
};