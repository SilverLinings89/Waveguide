#pragma once

#include "../Core/Types.h"
#include "./BoundaryCondition.h"
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/lac/affine_constraints.h>
#include "./PMLMeshTransformation.h"

class PMLSurface : public BoundaryCondition {
    ComplexNumber sigma_0;
    unsigned int inner_boundary_id;
    unsigned int outer_boundary_id;
    DofHandler3D dof_h_nedelec;
    dealii::FE_NedelecSZ<3> fe_nedelec;
    dealii::AffineConstraints<ComplexNumber> constraints;
    bool constraints_made;
    PMLMeshTransformation transformation;
    bool mesh_is_transformed;

  public: 
    std::array<std::set<unsigned int>, 6> edge_ids_by_boundary_id;
    std::array<std::set<unsigned int>, 6> face_ids_by_boundary_id;
    std::pair<double, double> x_range;
    std::pair<double, double> y_range;
    std::pair<double, double> z_range;
    dealii::Triangulation<3> triangulation;
    
    PMLSurface(unsigned int in_bid, unsigned int in_level, DofNumber first_own_index);
    ~PMLSurface();
    
    void prepare_id_sets_for_boundaries();
    bool is_point_at_boundary(Position, BoundaryId);
    void identify_corner_cells() override;
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
    auto fraction_of_pml_direction(Position) -> double;
    auto get_pml_tensor_epsilon(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto get_pml_tensor_mu(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto get_pml_tensor(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto make_inner_constraints() -> void;
    void copy_constraints(dealii::AffineConstraints<ComplexNumber> *, unsigned int shift);
    auto get_dof_count_by_boundary_id(BoundaryId in_boundary_id) -> DofCount override;
    auto get_dof_association() -> std::vector<InterfaceDofData> override;
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;
    void sort_dofs();
    void compute_coordinate_ranges();
    void set_boundary_ids();
    void fix_apply_negative_Jacobian_transformation(dealii::Triangulation<3> * in_tria);
    void output_results(const dealii::Vector<ComplexNumber> & , std::string) override;
    void make_surface_constraints(dealii::AffineConstraints<ComplexNumber> * constraints) override;
    void make_edge_constraints(dealii::AffineConstraints<ComplexNumber> * constraints, BoundaryId other_boundary) override;
    auto get_surface_cell_data(BoundaryId in_bid) -> std::vector<SurfaceCellData> override;
    auto get_inner_surface_cell_data() -> std::vector<SurfaceCellData> override;
    void validate_meshes();
    void fill_internal_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> * in_constriants) override;
    auto get_corner_surface_cell_data(BoundaryId main_boundary, BoundaryId secondary_boundary) -> std::vector<SurfaceCellData> override;
};