//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H

#include <mpi.h>
#include "HierarchicalProblem.h"
#include "./LocalProblem.h"

class NonLocalProblem: public HierarchicalProblem {
private:
  SweepingDirection sweeping_direction;
  bool *is_hsie_surface;
  dealii::TrilinosWrappers::SparseMatrix system_matrix;
  dealii::Vector<double> system_rhs;
  dealii::IndexSet local_indices;
  dealii::SolverGMRES<dealii::TrilinosWrappers::SparseMatrix> solver;

 public:
  NonLocalProblem(unsigned int);
  ~NonLocalProblem() override;

  unsigned int compute_own_dofs();

  void initialize_own_dofs() override;

  unsigned int compute_lower_interface_dof_count() override;

  unsigned int compute_upper_interface_dof_count() override;

  void assemble() override;

  void solve(dealii::Vector<double> src, dealii::Vector<double> &dst) override;

  void run() override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  void initialize_index_sets() override;

  void apply_sweep(dealii::LinearAlgebra::distributed::Vector<double>) override;

  LocalProblem* get_local_problem() override;

  void reinit();
};

#endif  // WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
