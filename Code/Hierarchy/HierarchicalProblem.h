#pragma once

#include "../Core/Types.h"
#include "../Helpers/Parameters.h"
#include "DofIndexData.h"
#include <deal.II/base/index_set.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h> 

class LocalProblem;
class HierarchicalProblem {

 public:
  SweepingDirection sweeping_direction;
  std::vector<DofNumber> surface_first_dofs;
  std::vector<bool> is_surface_locked;
  bool is_dof_manager_set;
  bool has_child;
  HierarchicalProblem* child;
  dealii::SparsityPattern sp;
  const SweepingLevel local_level;
  DofIndexData indices;
  NumericVectorDistributed rhs;
  NumericVectorDistributed solution;
  DofCount n_own_dofs;
  DofNumber first_own_index;
  DofCount dofs_process_above;
  DofCount dofs_process_below;
  unsigned int n_procs_in_sweep;
  unsigned int rank;
  dealii::IndexSet own_dofs;

  HierarchicalProblem(unsigned int in_own_level);
  virtual ~HierarchicalProblem() =0;

  virtual DofCount compute_lower_interface_dof_count()=0;
  virtual DofCount compute_upper_interface_dof_count()=0;

  virtual void solve(NumericVectorDistributed src,
      NumericVectorDistributed &dst)=0;
  virtual void solve(NumericVectorLocal src,
      NumericVectorLocal &dst)=0;

  virtual void initialize()=0;
  virtual void generate_sparsity_pattern()=0;
  virtual DofCount compute_own_dofs()=0;
  virtual void run() =0;
  virtual void initialize_own_dofs() =0;
  virtual void make_constraints() = 0;
  virtual void assemble()=0;
  virtual void initialize_index_sets()=0;
  virtual LocalProblem* get_local_problem()=0;
  void constrain_identical_dof_sets(std::vector<unsigned int> *set_one,
      std::vector<unsigned int> *set_two,
      dealii::AffineConstraints<ComplexNumber> *affine_constraints);
  virtual dealii::Vector<ComplexNumber> get_local_vector_from_global() = 0;
  virtual auto get_center() -> Position const = 0;
  virtual auto reinit() -> void = 0;
  virtual auto communicate_sweeping_direction(SweepingDirection) -> void = 0;

  virtual void set_boundary_dof_values() = 0;
  void lock_boundary_dofs(BoundaryId);
  void unlock_boundary_dofs(BoundaryId);
  virtual void clear_unlocked_dofs() = 0;
};
