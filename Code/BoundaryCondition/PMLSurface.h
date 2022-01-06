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
    dealii::FE_NedelecSZ<3> fe_nedelec;
    PMLMeshTransformation transformation;
    bool mesh_is_transformed;
    const double surface_coordinate;
    std::array<std::vector<InterfaceDofData>, 6> dof_associations;

  public: 
    std::pair<double, double> x_range;
    std::pair<double, double> y_range;
    std::pair<double, double> z_range;
    double non_pml_layer_thickness; // thickness of the cell layer closest to the surface. Here I set the transformation tensor to 0 to ensure surface dofs are not damped.
    dealii::Triangulation<3> triangulation;
    
    PMLSurface(unsigned int in_bid, unsigned int in_level);
    ~PMLSurface();
    
    bool is_point_at_boundary(Position, BoundaryId);
    auto make_constraints() -> Constraints override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) override;
    void fill_matrix(dealii::SparseMatrix<ComplexNumber>*,  Constraints *constraints) override;
    void fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) override;
    void fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix*, NumericVectorDistributed* rhs, Constraints *constraints) override;
    void fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constriants) override;
    bool is_point_at_boundary(Position2D in_p, BoundaryId in_bid) override;
    bool is_position_at_boundary(const Position in_p, const BoundaryId in_bid);
    bool is_position_at_extended_boundary(const Position in_p, const BoundaryId in_bid);
    void initialize() override;
    void set_mesh_boundary_ids(); 
    void prepare_mesh();
    auto cells_for_boundary_id(unsigned int boundary_id) -> unsigned int override;
    void init_fe();
    auto fraction_of_pml_direction(Position) -> std::array<double, 3> ;
    auto get_pml_tensor_epsilon(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto get_pml_tensor_mu(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto get_pml_tensor(Position) -> dealii::Tensor<2,3,ComplexNumber>;
    auto get_dof_association() -> std::vector<InterfaceDofData> override;
    auto get_dof_association_by_boundary_id(BoundaryId in_boundary_id) -> std::vector<InterfaceDofData> override;
    void compute_coordinate_ranges(dealii::Triangulation<3> * in_tria);
    void set_boundary_ids();
    void fix_apply_negative_Jacobian_transformation(dealii::Triangulation<3> * in_tria);
    std::string output_results(const dealii::Vector<ComplexNumber> & , std::string) override;
    void validate_meshes();
    DofCount compute_n_locally_owned_dofs() override;
    DofCount compute_n_locally_active_dofs() override;
    void finish_dof_index_initialization() override;
    void determine_non_owned_dofs() override;
    dealii::IndexSet compute_non_owned_dofs();
    bool finish_initialization(DofNumber first_own_index) override;
    bool mg_process_edge(dealii::Triangulation<3> * current_list, BoundaryId b_id);
    bool mg_process_corner(dealii::Triangulation<3> * current_list, BoundaryId first_bid, BoundaryId second_bid);
    bool extend_mesh_in_direction(BoundaryId in_bid);
    void prepare_dof_associations();
    unsigned int n_cells() override;
};