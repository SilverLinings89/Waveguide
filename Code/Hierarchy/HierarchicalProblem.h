#pragma once

#include "../Core/Types.h"
#include "../Helpers/Parameters.h"
#include "DofIndexData.h"
#include <deal.II/base/index_set.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include "../Core/FEDomain.h"

class LocalProblem;
class NonLocalProblem; 

class HierarchicalProblem {
 public:
  const SweepingDirection sweeping_direction;
  const SweepingLevel level;
  Constraints constraints;
  std::array<dealii::IndexSet, 6> surface_index_sets;
  std::array<bool, 6> is_hsie_surface;
  std:: vector<bool> is_surface_locked;
  bool is_dof_manager_set;
  bool has_child;
  HierarchicalProblem* child;
  dealii:: SparsityPattern sp;
  DofIndexData indices;
  NumericVectorDistributed solution;
  NumericVectorDistributed direct_solution;
  NumericVectorDistributed solution_error;
  NumericVectorDistributed rhs;
  unsigned int n_procs_in_sweep;
  unsigned int rank;
  dealii::IndexSet own_dofs;
  dealii::IndexSet current_upper_sweeping_dofs;
  dealii::IndexSet current_lower_sweeping_dofs;
  std::array<std::vector<InterfaceDofData>, 6> surface_dof_associations;
  dealii::PETScWrappers::MPI::SparseMatrix * matrix;
  std::vector<std::string> filenames;

  HierarchicalProblem(unsigned int level, SweepingDirection direction);
  virtual ~HierarchicalProblem() =0;

  virtual void solve()=0;
  virtual void initialize()=0;
  void make_constraints();
  virtual void assemble()=0;
  virtual void initialize_index_sets()=0;
  void constrain_identical_dof_sets(std::vector<unsigned int> *set_one, std::vector<unsigned int> *set_two, Constraints *affine_constraints);
  virtual auto get_center() -> Position const = 0;
  virtual auto reinit() -> void = 0;
  auto opposing_site_bid(BoundaryId) -> BoundaryId;
  void compute_final_rhs_mismatch();

  virtual void compute_solver_factorization() = 0;
  std::string output_results(std::string in_fname_part = "solution_inner_domain_level");
  virtual void reinit_rhs() = 0;
  virtual void make_sparsity_pattern();
  void execute_vmult();
  void compute_rhs_representation_of_incoming_wave();
  void initialize_dof_counts();
  void distribute_global_indices();
};

typedef struct {
  NonLocalProblem * parent;
} SampleShellPC;