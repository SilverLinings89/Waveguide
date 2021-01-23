#pragma once

#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
 #include <deal.II/lac/affine_constraints.h>

class PMLSurface : public BoundaryCondition {
    const unsigned int cell_layers;
    const unsigned int thickness;
    const unsigned int pml_skaling_order;
    ComplexNumber sigma_0;
    dealii::Triangulation<3> triangulation;
    unsigned int inner_boundary_id;
    unsigned int outer_boundary_id;
    DofHandler3D dof_h_nedelec;
    dealii::FE_NedelecSZ<3> fe_nedelec;
    dealii::AffineConstraints<ComplexNumber> constraints;
    bool constraints_made;

    PMLSurface(unsigned int in_bid, double in_additional_coordinate, dealii::Triangulation<2> & in_surf_tria, unsigned int layers, double in_thickness, unsigned int in_skaling_order);
    void identify_corner_cells() override;
    void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, DofNumber shift, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet,  std::array<bool, 6> surfaces_hsie,  dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, dealii::IndexSet, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints) override;
    bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;
    void initialize() override;
    void set_mesh_boundary_ids(); 
    void prepare_mesh();
    auto cells_for_boundary_id(unsigned int boundary_id) -> unsigned int;
    void init_fe();
    auto fraction_of_pml_direction(Position) -> double;
    auto get_pml_tensor_epsilon(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto get_pml_tensor_mu(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto get_pml_tensor(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto make_inner_constraints() -> void;
    void copy_constraints(dealii::AffineConstraints<ComplexNumber> *, unsigned int shift);
    auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount override;
    auto get_dof_association() -> std::vector<DofIndexAndOrientationAndPosition> override;
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<DofIndexAndOrientationAndPosition> override;
    void sort_dofs();
};