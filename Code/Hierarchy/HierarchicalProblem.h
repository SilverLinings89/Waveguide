//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_HIERARCHICALPROBLEM_H
#define WAVEGUIDEPROBLEM_HIERARCHICALPROBLEM_H
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include "../Core/DOFManager.h"
#include "../Helpers/Parameters.h"
#include "DofIndexData.h"
class LocalProblem;

class HierarchicalProblem {
 public:
  bool has_child;
  HierarchicalProblem* child;
  const unsigned int global_level;
  const unsigned int local_level;
  DOFManager* dof_manager;
  DofIndexData indices;
  dealii::TrilinosWrappers::SparseMatrix matrix;

  HierarchicalProblem(unsigned int in_own_level, unsigned int in_global_level,
      DOFManager *in_dof_manager);

  virtual void send_vector_to_upper();
  virtual void receive_vector_from_upper();
  virtual void send_vector_to_lower();
  virtual void receive_vector_from_lower();

  virtual unsigned int compute_lower_interface_dof_count();
  virtual unsigned int compute_upper_interface_dof_count();

  virtual void solve();
  virtual void initialize();
  virtual void generate_sparsity_pattern();
  virtual unsigned int compute_own_dofs();

  virtual void solve_inner();
  virtual void assemble();
  virtual void make_sparsity_pattern();
  virtual void initialize_index_sets();
  virtual void apply_sweep(dealii::LinearAlgebra::distributed::Vector<double>);
  virtual dealii::IndexSet get_owned_dofs_for_level(unsigned int level);
  virtual LocalProblem* get_local_problem();
};

#endif  // WAVEGUIDEPROBLEM_HIERARCHICALPROBLEM_H
