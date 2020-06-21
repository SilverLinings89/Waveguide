//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_LOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_LOCALPROBLEM_H


#include "HierarchicalProblem.h"
#include "../Core/DOFManager.h"
#include "../Core/NumericProblem.h"
#include "../HSIEPreconditioner/HSIESurface.h"

class LocalProblem: public HierarchicalProblem {
public:

  NumericProblem base_problem;
  HSIESurface **surfaces;
  HSIESurface *surface_0;
  HSIESurface *surface_1;
  HSIESurface *surface_2;
  HSIESurface *surface_3;
  HSIESurface *surface_4;
  HSIESurface *surface_5;

  dealii::SparseDirectUMFPACK solver;
  dealii::AffineConstraints<std::complex<double>> constraints;
  dealii::SparsityPattern *sp;

  LocalProblem();
  ~LocalProblem() override;

  unsigned int compute_lower_interface_dof_count() override;

  unsigned int compute_upper_interface_dof_count() override;

  void solve(dealii::Vector<std::complex<double>> src,
      dealii::Vector<std::complex<double>> &dst) override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  unsigned int compute_own_dofs() override;

  void initialize_own_dofs() override;

  void make_constraints() override;

  void run() override;

  void solve();

  void assemble() override;

  void initialize_index_sets() override;

  void apply_sweep(
      dealii::LinearAlgebra::distributed::Vector<std::complex<double>>);

  LocalProblem* get_local_problem() override;

  void validate();

  dealii::Vector<std::complex<double>> get_local_vector_from_global() override;

  void output_results();
};


#endif //WAVEGUIDEPROBLEM_LOCALPROBLEM_H
