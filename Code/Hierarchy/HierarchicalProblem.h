#pragma once

#include "../Core/Types.h"
#include "../Helpers/Parameters.h"
#include "DofIndexData.h"
#include <deal.II/base/index_set.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include "../Core/FEDomain.h"
#include "../OutputGenerators/Images/ResidualOutputGenerator.h"

class LocalProblem;
class NonLocalProblem; 

class HierarchicalProblem {
 public:
  SweepingDirection sweeping_direction;
  const SweepingLevel level;
  Constraints constraints;
  std::array<dealii::IndexSet, 6> surface_index_sets;
  std::array<bool, 6> is_hsie_surface;
  std:: vector<bool> is_surface_locked;
  bool is_dof_manager_set;
  bool has_child;
  HierarchicalProblem* child;
  dealii:: SparsityPattern sp;
  NumericVectorDistributed solution;
  NumericVectorDistributed direct_solution;
  NumericVectorDistributed solution_error;
  NumericVectorDistributed rhs;
  dealii::IndexSet own_dofs;
  std::array<std::vector<InterfaceDofData>, 6> surface_dof_associations;
  dealii::PETScWrappers::MPI::SparseMatrix * matrix;
  std::vector<std::string> filenames;
  ResidualOutputGenerator * residual_output;
  unsigned int solve_counter;
  int parent_sweeping_rank = -1;

  HierarchicalProblem(unsigned int level, SweepingDirection direction);
  virtual ~HierarchicalProblem() =0;

  virtual void solve()=0;
  void solve_with_timers_and_count();
  virtual void initialize()=0;
  void make_constraints();
  virtual void assemble()=0;
  virtual void initialize_index_sets()=0;
  void constrain_identical_dof_sets(std::vector<unsigned int> *set_one, std::vector<unsigned int> *set_two, Constraints *affine_constraints);
  virtual auto reinit() -> void = 0;
  auto opposing_site_bid(BoundaryId) -> BoundaryId;
  void compute_final_rhs_mismatch();

  virtual void compute_solver_factorization() = 0;
  std::string output_results(std::string in_fname_part = "solution_inner_domain_level");
  virtual void reinit_rhs() = 0;
  virtual void make_sparsity_pattern();
  void initialize_dof_counts();
  virtual void update_convergence_criterion(double) {}
  virtual unsigned int compute_global_solve_counter() {
    return 0;
  }
  void print_solve_counter_list();
  virtual void empty_memory();
  virtual void write_multifile_output(const std::string & filename, bool apply_coordinate_transform) =0;
};

typedef struct {
  NonLocalProblem * parent;
} SampleShellPC;