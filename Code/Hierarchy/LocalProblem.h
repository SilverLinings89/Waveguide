#pragma once

#include <deal.II/lac/petsc_sparse_matrix.h>
#include "HierarchicalProblem.h"
#include "../Core/InnerDomain.h"
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

  void solve() override;

  void initialize() override;

  void initialize_own_dofs() override;

  void assemble() override;

  void initialize_index_sets() override;

  void validate();

  dealii::Vector<ComplexNumber> get_local_vector_from_global() override;
  
  auto get_center() -> Position const override;

  auto reinit() -> void override;

  auto reinit_rhs() -> void override;

  auto compare_to_exact_solution() -> void;
  
  dealii::IndexSet compute_interface_dof_set(BoundaryId interface_id);

  void compute_solver_factorization() override;

  double compute_L2_error();

  double compute_error(dealii::VectorTools::NormType, Function<3,ComplexNumber> *, dealii::Vector<ComplexNumber> & , dealii::DataOut<3> *);

  DofOwner get_dof_owner(unsigned int id);

  DofCount compute_n_locally_owned_dofs(std::array<bool, 6> is_locally_owned_surface) override;

  DofCount compute_n_locally_active_dofs() override;
};
