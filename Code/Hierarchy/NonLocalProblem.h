#pragma once

#include "../Core/Types.h"
#include <mpi.h>
#include <complex>
#include "HierarchicalProblem.h"
#include "./LocalProblem.h"
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include "../Helpers/Enums.h"

class NonLocalProblem: public HierarchicalProblem {
private:
  SweepingDirection sweeping_direction;
  bool *is_hsie_surface;
  DofCount total_number_of_dofs_on_level;
  dealii::PETScWrappers::MPI::SparseMatrix *matrix;
  NumericVectorDistributed *system_rhs;
  dealii::IndexSet local_indices;
  dealii::SolverControl sc;
  dealii::SolverGMRES<NumericVectorDistributed> solver;
 public:
  NonLocalProblem(unsigned int);
  ~NonLocalProblem() override;

  auto compute_own_dofs() -> DofCount override;

  auto initialize_own_dofs() -> void override;

  auto compute_lower_interface_dof_count() -> DofCount override;

  auto compute_upper_interface_dof_count() -> DofCount override;

  auto compute_lower_interface_id() -> BoundaryId;

  auto compute_upper_interface_id() -> BoundaryId;

  void assemble() override;

  void solve(NumericVectorDistributed src,
      NumericVectorDistributed &dst) override;

  auto solve(NumericVectorLocal,
      NumericVectorLocal &) -> void override {
        std::cout << "Calling wrong solve on NonLocal Problem." << std::endl;
      } ;

  void run() override;

  void initialize() override;

  void make_constraints() override;

  void generate_sparsity_pattern() override;

  void initialize_index_sets() override;

  void apply_sweep(NumericVectorDistributed) override;

  LocalProblem* get_local_problem() override;

  void reinit() override;

  NumericVectorLocal get_local_vector_from_global() override;

  auto get_center() -> Position const override;
};
