#pragma once

#include "../Core/Types.h"
#include <mpi.h>
#include <complex>
#include "HierarchicalProblem.h"
#include "./LocalProblem.h"
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include "../Helpers/Enums.h"

class NonLocalProblem: public HierarchicalProblem {
private:
  std::vector<dealii::IndexSet> index_sets_per_process;
  std::array<bool, 6> is_hsie_surface;
  std::array<bool, 6> is_sweeping_hsie_surface;
  DofCount total_number_of_dofs_on_level;
  dealii::PETScWrappers::MPI::SparseMatrix *matrix;
  NumericVectorDistributed *system_rhs;
  dealii::IndexSet local_indices;
  dealii::IndexSet upper_interface_dofs;
  dealii::IndexSet lower_interface_dofs;
  dealii::SolverControl sc;
  dealii::PETScWrappers::SolverGMRES solver;
  dealii::AffineConstraints<ComplexNumber> constraints;
  NumericVectorDistributed current_solution;
  std::vector<std::pair<DofNumber, DofNumber>> coupling_dofs;
  std::vector<ComplexNumber> cached_lower_values;
  std::vector<ComplexNumber> cached_upper_values;
  ComplexNumber * mpi_cache;
  PC pc;
  SampleShellPC shell;
  dealii::DynamicSparsityPattern dsp;
  PetscInt* locally_owned_dofs_index_array;
  unsigned int lower_sweeping_interface_id;
  unsigned int upper_sweeping_interface_id;
  std::vector<bool> dof_orientations_identical;
  
 public:
  NonLocalProblem(unsigned int);

  ~NonLocalProblem() override;

  auto reinit_mpi_cache() -> void;

  auto compute_own_dofs() -> DofCount override;

  auto initialize_own_dofs() -> void override;

  auto compute_lower_interface_dof_count() -> DofCount override;

  auto compute_upper_interface_dof_count() -> DofCount override;

  DofCount compute_interface_dofs(BoundaryId interface_id);

  dealii::IndexSet compute_interface_dof_set(BoundaryId interface_id);

  auto compute_lower_interface_id() -> BoundaryId;

  auto compute_upper_interface_id() -> BoundaryId;

  void assemble() override;

  void solve() override;

  void apply_sweep(Vec x_in, Vec x_out);

  void init_solver_and_preconditioner();

  void initialize() override;

  void make_constraints() override;

  void generate_sparsity_pattern() override;

  void initialize_index_sets() override;

  LocalProblem* get_local_problem() override;

  void reinit() override;

  NumericVectorLocal get_local_vector_from_global() override;

  auto get_center() -> Position const override;

  auto communicate_sweeping_direction(SweepingDirection) -> void override;

  void H_inverse();

  NumericVectorLocal extract_local_upper_dofs();

  NumericVectorLocal extract_local_lower_dofs();

  void send_local_lower_dofs();

  void receive_local_lower_dofs();

  void send_local_upper_dofs();

  void receive_local_upper_dofs();

  bool is_lowest_in_sweeping_direction();

  bool is_highest_in_sweeping_direction();

  auto set_boundary_values(BoundaryId, std::vector<ComplexNumber>) -> void override;
  
  auto release_boundary_values(BoundaryId) -> void override;

  auto make_constraints_for_hsie_surface(unsigned int index) -> void;
  
  auto make_constraints_for_non_hsie_surface(unsigned int index) -> void;

  void propagate_up();

  void compute_solver_factorization() override;
  
  void output_results() override;

  std::vector<bool> get_incoming_dof_orientations();
};
