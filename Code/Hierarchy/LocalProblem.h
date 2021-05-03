#pragma once

#include <deal.II/lac/petsc_sparse_matrix.h>
#include "HierarchicalProblem.h"
#include "../Core/NumericProblem.h"
#include "../BoundaryCondition/BoundaryCondition.h"
#include "../BoundaryCondition/HSIESurface.h"
#include "../BoundaryCondition/PMLSurface.h"
#include "deal.II/lac/petsc_solver.h"
#include <array>
#include <memory>

class LocalProblem: public HierarchicalProblem {
public:
  unsigned int solve_counter = 0;
  SolverControl sc;
  dealii::PETScWrappers::SparseDirectMUMPS solver;

  LocalProblem();
  ~LocalProblem() override;

  DofCount compute_lower_interface_dof_count() override;

  DofCount compute_upper_interface_dof_count() override;

  void solve() override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  void initialize_own_dofs() override;

  void assemble() override;

  void initialize_index_sets() override;

  void validate();

  dealii::Vector<ComplexNumber> get_local_vector_from_global() override;

  void output_results() override;

  auto get_center() -> Position const override;

  auto reinit() -> void override;

  auto reinit_rhs() -> void override;

  auto compare_to_exact_solution() -> void;
  
  dealii::IndexSet compute_interface_dof_set(BoundaryId interface_id);

  void compute_solver_factorization() override;

  void update_mismatch_vector(BoundaryId interface) override;

  double compute_L2_error();

  double compute_error(dealii::VectorTools::NormType, Function<3,ComplexNumber> *, dealii::Vector<ComplexNumber> & , dealii::DataOut<3> *);

  auto write_phase_plot() -> void;

  DofOwner get_dof_owner(unsigned int id);
};
