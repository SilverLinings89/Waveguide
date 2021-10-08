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
  SolverControl sc;
  dealii::PETScWrappers::SparseDirectMUMPS solver;

  LocalProblem();
  ~LocalProblem() override;

  void solve() override;

  void initialize() override;

  void assemble() override;

  void initialize_index_sets() override;

  void validate();

  auto reinit() -> void override;

  auto reinit_rhs() -> void override;

  auto compare_to_exact_solution() -> void;
  
  dealii::IndexSet compute_interface_dof_set(BoundaryId interface_id);

  void compute_solver_factorization() override;

  double compute_L2_error();

  double compute_error(dealii::VectorTools::NormType, Function<3,ComplexNumber> *, dealii::Vector<ComplexNumber> & , dealii::DataOut<3> *);

};
