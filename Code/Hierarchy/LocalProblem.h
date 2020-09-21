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

  void solve(NumericVectorLocal src, NumericVectorLocal &dst) override;
  
  void solve(NumericVectorDistributed src, NumericVectorDistributed &dst) override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  auto compute_own_dofs() -> DofCount override;

  void initialize_own_dofs() override;

  void make_constraints() override;

  void run() override;

  void solve();

  void assemble() override;

  void initialize_index_sets() override;

  auto get_local_problem() -> LocalProblem* override;

  void validate();

  dealii::Vector<ComplexNumber> get_local_vector_from_global() override;

  void output_results();

  auto get_center() -> Position const override;

  auto reinit() -> void override;

  auto compare_to_exact_solution() -> void;

  auto communicate_sweeping_direction(SweepingDirection) -> void override;

  auto set_boundary_dof_values()  -> void override;

  auto clear_unlocked_dofs()  -> void override;
};
