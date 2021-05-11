#pragma once

#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/lac/affine_constraints.h>

class NeighborSurface : public BoundaryCondition {
    
  public: 
    const bool is_primary; // The process with lower rank on the surface is the primary. It "goes first" when equal tasks have to be performed one after the other.
    const unsigned int global_partner_mpi_rank;
    const unsigned int partner_mpi_rank_in_level_communicator;
    std::array<std::set<unsigned int>, 6> edge_ids_by_boundary_id;
    std::array<std::set<unsigned int>, 6> face_ids_by_boundary_id;

    NeighborSurface(unsigned int in_bid, unsigned int in_level, DofNumber first_own_index);
    ~NeighborSurface();
    
    void prepare_id_sets_for_boundaries();
    bool is_point_at_boundary(Position, BoundaryId);
    void identify_corner_cells() override;
    void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;
    bool is_position_at_boundary(Position in_p, BoundaryId in_bid);
    void initialize() override;
    void set_mesh_boundary_ids(); 
    void prepare_mesh();
    auto cells_for_boundary_id(unsigned int boundary_id) -> unsigned int;
    void init_fe();
    auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount override;
    auto get_dof_association() -> std::vector<InterfaceDofData> override;
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;
    void sort_dofs();
    void compute_coordinate_ranges();
    void set_boundary_ids();
    void fill_sparsity_pattern_for_boundary_id(const BoundaryId in_bid, dealii::AffineConstraints<ComplexNumber> * constraints, dealii::DynamicSparsityPattern * dsp) override;
    void output_results(const dealii::Vector<ComplexNumber> & , std::string) override;
    void make_surface_constraints(dealii::AffineConstraints<ComplexNumber> * constraints) override;
    void make_edge_constraints(dealii::AffineConstraints<ComplexNumber> * constraints, BoundaryId other_boundary) override;
    auto get_surface_cell_data(BoundaryId in_bid) -> std::vector<SurfaceCellData> override;
    auto get_inner_surface_cell_data() -> std::vector<SurfaceCellData> override;
};