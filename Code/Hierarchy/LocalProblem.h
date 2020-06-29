#pragma once

#include "HierarchicalProblem.h"
#include "../Core/NumericProblem.h"
#include "../HSIEPreconditioner/HSIESurface.h"
#include <array>
#include <memory>

class LocalProblem: public HierarchicalProblem {
public:

  NumericProblem base_problem;
  std::array<std::shared_ptr<HSIESurface>,6> surfaces;
  dealii::SparseDirectUMFPACK solver;
  dealii::AffineConstraints<std::complex<double>> constraints;
  dealii::SparsityPattern *sp;

  LocalProblem();
  ~LocalProblem() override;

  DofCount compute_lower_interface_dof_count() override;

  DofCount compute_upper_interface_dof_count() override;

  auto solve(NumericVectorDistributed, NumericVectorDistributed &) -> void override {
    std::cout << "Wrong solve function called in LocalProblem." << std::endl;
  };

  void solve(NumericVectorLocal src, NumericVectorLocal &dst) override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  auto compute_own_dofs() -> DofCount override;

  void initialize_own_dofs() override;

  void make_constraints() override;

  void run() override;

  void solve();

  void assemble() override;

  void initialize_index_sets() override;

  void apply_sweep(dealii::LinearAlgebra::distributed::Vector<std::complex<double>>);

  auto get_local_problem() -> LocalProblem* override;

  void validate();

  dealii::Vector<std::complex<double>> get_local_vector_from_global() override;

  void output_results();
};
