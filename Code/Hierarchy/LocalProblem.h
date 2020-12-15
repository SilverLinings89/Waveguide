#pragma once

#include <deal.II/lac/petsc_sparse_matrix.h>
#include "HierarchicalProblem.h"
#include "../Core/NumericProblem.h"
#include "../HSIEPreconditioner/HSIESurface.h"
#include "deal.II/lac/petsc_solver.h"
#include <array>
#include <memory>

class LocalProblem: public HierarchicalProblem {
public:
  unsigned int solve_counter = 0;
  NumericProblem base_problem;
  std::array<std::shared_ptr<HSIESurface>,6> surfaces;
  SolverControl sc;
  dealii::PETScWrappers::SparseDirectMUMPS solver;
  dealii::PETScWrappers::SparseMatrix * matrix;
  dealii::AffineConstraints<ComplexNumber> constraints;

  LocalProblem();
  ~LocalProblem() override;

  DofCount compute_lower_interface_dof_count() override;

  DofCount compute_upper_interface_dof_count() override;

  void solve() override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  auto compute_own_dofs() -> DofCount override;

  void initialize_own_dofs() override;

  void make_constraints() override;

  void assemble() override;

  void initialize_index_sets() override;

  auto get_local_problem() -> LocalProblem* override;

  void validate();

  dealii::Vector<ComplexNumber> get_local_vector_from_global() override;

  void output_results() override;

  auto get_center() -> Position const override;

  auto reinit() -> void override;

  auto compare_to_exact_solution() -> void;

  auto communicate_sweeping_direction(SweepingDirection) -> void override;

  auto set_boundary_values(BoundaryId, std::vector<ComplexNumber>) -> void override;
  
  auto release_boundary_values(BoundaryId) -> void override;

  dealii::IndexSet compute_interface_dof_set(BoundaryId interface_id);

  void compute_solver_factorization() override;

  void update_mismatch_vector() override;
};
