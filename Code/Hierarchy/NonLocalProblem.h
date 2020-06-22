//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H

#include "../Core/Types.h"
#include <mpi.h>
#include <complex>
#include "HierarchicalProblem.h"
#include "./LocalProblem.h"
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/parallel_vector.h>
#include "../Helpers/Enums.h"

using namespace dealii;

class NonLocalProblem: public HierarchicalProblem {
private:
  SweepingDirection sweeping_direction;
  bool *is_hsie_surface;
  dealii::SparseMatrix<EFieldComponent> *system_matrix;
  NumericVectorDistributed *system_rhs;
  dealii::IndexSet local_indices;
  dealii::SolverGMRES<NumericVectorDistributed> solver;
  dealii::SolverControl sc;
 public:
  NonLocalProblem(unsigned int);
  ~NonLocalProblem() override;

  auto compute_own_dofs() -> DofCount override;

  auto initialize_own_dofs() -> void override;

  auto compute_lower_interface_dof_count() -> DofCount override;

  auto compute_upper_interface_dof_count() -> DofCount override;

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

  void reinit();

  NumericVectorLocal get_local_vector_from_global() override;

};

#endif  // WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
