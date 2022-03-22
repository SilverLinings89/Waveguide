#pragma once
/**
 * @file NeighborSurface.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/lac/affine_constraints.h>

/**
 * @class NeighborSurface
 * 
 * \brief For non-local problem, these interfaces are ones, that connect two inner domains and handle the communication between the two as well as the adjacent boundaries.
 * This matrix has no effect for the assembly of system matrices since these boundaries have no own dofs. This object mainly communicates dof indices during the initialization phase.
 */
class NeighborSurface : public BoundaryCondition {
    
  public: 
    const bool is_lower_interface;
    std::array<std::set<unsigned int>, 6> edge_ids_by_boundary_id;
    std::array<std::set<unsigned int>, 6> face_ids_by_boundary_id;
    std::array<std::vector<InterfaceDofData>, 6> dof_indices_by_boundary_id;
    std::array<std::vector<unsigned int>, 6> boundary_dofs;
    std::vector<unsigned int> inner_dofs;
    std::vector<unsigned int> global_indices;
    unsigned int n_dofs;
    bool dofs_prepared;

    NeighborSurface(unsigned int in_bid, unsigned int in_level);
    ~NeighborSurface();
    
    /**
     * @brief Does nothing, only fulfills the interface.
     * 
     * @param matrix Matrix to fill.
     * @param rhs Rhs to fill.
     * @param constraints Constraints to condense.
     */
    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, Constraints *constraints) override;

    /**
     * @brief Does nothing, only fulfills the interface.
     * 
     * @param in_dsp Sparsity pattern to use
     * @param in_constriants Constraints to use
     */
    void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constriants) override;

    /**
     * @brief Does nothing, alwys returns false since this function is only there to fulfill the interface of boundary condition.
     * 
     * @param in_p 
     * @param in_bid 
     * @return true 
     * @return false 
     */
    bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;

    /**
     * @brief Initializes the datastructures.
     * 
     */
    void initialize() override;

    /**
     * @brief sets boundary ids on the surface triangulation.
     * 
     */
    void set_mesh_boundary_ids();

    /**
     * @brief Fulfills the boundary condition interface. 
     * For NeighborSurface this function returns the return value from InnerDomain::get_dof_association.
     * 
     * @return std::vector<InterfaceDofData> a vector of dofs at the interface.
     */
    auto get_dof_association() -> std::vector<InterfaceDofData> override;

    /**
     * @brief Fulfills the boundary condition interface.
     * This function returns either the surface dofs from the inner domain or one of the adjacent interfaces to this one. 
     * 
     * @param in_boundary_id Boundary to search on.
     * @return std::vector<InterfaceDofData> Vector of all the dofs at the surface
     */
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;

    /**
     * @brief Does nothing in this class.
     * 
     * @param solution The solution to be evaluated
     * @param filename The name of the file to write the solution to
     * @return std::string filename
     */
    std::string output_results(const dealii::Vector<ComplexNumber> & solution , std::string filename) override;

    /**
     * @brief Computes the number of locally owned dofs.
     * 
     * @return DofCount number of locally owned dofs.
     */
    DofCount compute_n_locally_owned_dofs() override;

    /**
     * @brief Computes the number of locally active dofs.
     * 
     * @return DofCount number of locally active dofs.
     */
    DofCount compute_n_locally_active_dofs() override;

    /**
     * @brief Prepares internal datastructures for dof numbering
     * On this class, however, this function does nothing since objects of this type own no dofs.
     */
    void determine_non_owned_dofs() override;

    /**
     * @brief Interfaces of this type always have a neighbor. This function exchanges the data.
     * For example, for normal sweeping in z direction, if there are 2 blocks, 0 and 1 then they share one interface. Surface 5 on 0 and 4 at 1.
     * On both these Surfaces, there are boundary conditions of type neighbor and block 1 needs to number the surface dofs with the same numbers as 0 does so the matrix they assemble together.
     * To fulfill this purpose, they retreive the local numbering of the surface dofs from the inner domain and then exchange it, or, more precisely it is sent up.
     * The lower process sends this data to the higher, because the lower process owns the dofs.
     * 
     */
    void finish_dof_index_initialization() override;
    
    /**
     * @brief Distributes the dofs indices to the inner domain and all neighbors.
     * 
     */
    void distribute_dof_indices();

    /**
     * @brief Sends the own dofs to the partner process.
     * 
     */
    void send();

    /**
     * @brief Receives the dof numbers from the partner process.
     * 
     */
    void receive();

    /**
     * @brief Before the dofs can be exchanged, the boundary has to determine which the local dofs actually are.
     * Not all dofs on the surface are necessariy locally owned by the inner domain - they could belong to another prcess via another surface for example. This is an important action during the distribution of dof indices. 
     * 
     */
    void prepare_dofs();

};