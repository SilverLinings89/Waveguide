#pragma once

#include "../Core/Types.h"
#include "../Helpers/Parameters.h"
#include "DofIndexData.h"
#include <deal.II/base/index_set.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h> 

class LocalProblem;
class NonLocalProblem; 

class HierarchicalProblem {
 public:
  SweepingDirection sweeping_direction;
  std:: vector<DofNumber> surface_first_dofs;
  std::array<dealii::IndexSet, 6> surface_index_sets;
  std:: vector<bool> is_surface_locked;
  bool is_dof_manager_set;
  bool has_child;
  HierarchicalProblem* child;
  dealii:: SparsityPattern sp;
  const SweepingLevel local_level;
  DofIndexData indices;
  NumericVectorDistributed solution;
  NumericVectorDistributed rhs;
  DofCount n_own_dofs;
  DofNumber first_own_index;
  DofCount dofs_process_above;
  DofCount dofs_process_below;
  unsigned int n_procs_in_sweep;
  unsigned int rank;
  dealii::IndexSet own_dofs;
  dealii::IndexSet current_upper_sweeping_dofs;
  dealii::IndexSet current_lower_sweeping_dofs;
  std::array<std::vector<DofIndexAndOrientationAndPosition>, 6> surface_dof_associations;
  std::array<std::vector<DofNumber>, 6> surface_dof_index_vectors;

  HierarchicalProblem(unsigned int in_own_level);
  virtual ~HierarchicalProblem() =0;

  virtual DofCount compute_lower_interface_dof_count()=0;
  virtual DofCount compute_upper_interface_dof_count()=0;
  virtual void solve()=0;
  virtual void initialize()=0;
  virtual void generate_sparsity_pattern()=0;
  virtual DofCount compute_own_dofs()=0;
  virtual void initialize_own_dofs() =0;
  virtual void make_constraints() = 0;
  virtual void assemble()=0;
  virtual void initialize_index_sets()=0;
  virtual LocalProblem* get_local_problem()=0;
  void constrain_identical_dof_sets(std::vector<unsigned int> *set_one, std::vector<unsigned int> *set_two,
      dealii::AffineConstraints<ComplexNumber> *affine_constraints);
  virtual dealii::Vector<ComplexNumber> get_local_vector_from_global() = 0;
  virtual auto get_center() -> Position const = 0;
  virtual auto reinit() -> void = 0;
  virtual auto communicate_sweeping_direction(SweepingDirection) -> void = 0;
  void lock_boundary_dofs(BoundaryId);
  void unlock_boundary_dofs(BoundaryId);
  auto opposing_site_bid(BoundaryId) -> BoundaryId;
  auto set_dofs_in_inner_problem(BoundaryId, NumericVectorLocal);
  auto set_dofs_in_inner_problem_from_other_process(BoundaryId, DofDataVector);
  virtual auto set_boundary_values(BoundaryId, std::vector<ComplexNumber>) -> void = 0;
  virtual auto release_boundary_values(BoundaryId) -> void = 0;
  void copy_temp_to_current_solution();
  std::vector<ComplexNumber> get_surface_values_with_orientation_fix(BoundaryId bid,NumericVectorDistributed vector);
  void set_surface_values_with_orientation_fix(BoundaryId bid, NumericVectorDistributed * vector, std::vector<ComplexNumber> values);
};

typedef struct {
  // PetscInt * parent_elements_vec_petsc;
  // PetscInt * child_elements_vec_petsc;
  // std::vector<DofNumber> parent_elements_vec;
  // std::vector<DofNumber> child_elements_vec;
  // DofCount n_local_problem_indices;
  // HierarchicalProblem * child;
  NonLocalProblem * parent;
} SampleShellPC;