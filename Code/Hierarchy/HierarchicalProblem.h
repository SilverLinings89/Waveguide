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
  std::vector<unsigned int> surface_first_dofs;
  bool is_dof_manager_set;
  bool has_child;
  HierarchicalProblem* child;
  const unsigned int local_level;
  DOFManager* dof_manager;
  DofIndexData indices;
  dealii::TrilinosWrappers::SparseMatrix *matrix;
  unsigned int n_own_dofs;
  unsigned int first_own_index;
  unsigned int dofs_process_above;
  unsigned int dofs_process_below;
  unsigned int n_procs_in_sweep;
  unsigned int rank;

  HierarchicalProblem(unsigned int in_own_level);
  virtual ~HierarchicalProblem() =0;

  virtual unsigned int compute_lower_interface_dof_count()=0;
  virtual unsigned int compute_upper_interface_dof_count()=0;

  virtual void solve(dealii::Vector<double> src, dealii::Vector<double> &dst)=0;
  virtual void initialize()=0;
  virtual void generate_sparsity_pattern()=0;
  virtual unsigned int compute_own_dofs()=0;
  virtual void run() =0;
  virtual void initialize_own_dofs();
  virtual void assemble()=0;
  virtual void initialize_index_sets()=0;
  virtual void apply_sweep(
      dealii::LinearAlgebra::distributed::Vector<double>)=0;
  virtual LocalProblem* get_local_problem()=0;
  void setup_dof_manager(DOFManager *in_dof_manager);
  void constrain_identical_dof_sets(std::vector<unsigned int> *set_one,
      std::vector<unsigned int> *set_two,
      dealii::AffineConstraints<double> *affine_constraints);
};

#endif  // WAVEGUIDEPROBLEM_HIERARCHICALPROBLEM_H
