//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_LOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_LOCALPROBLEM_H


#include "HierarchicalProblem.h"
#include "../Core/DOFManager.h"
#include "../Core/NumericProblem.h"

class LocalProblem: public HierarchicalProblem,
    dealii::TrilinosWrappers::PreconditionBase {
  using dealii::TrilinosWrappers::PreconditionBase::vmult;
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
  dealii::AffineConstraints<double> constraints;
  dealii::SparsityPattern *sp;

  LocalProblem();
  ~LocalProblem() override;

  virtual void vmult(dealii::TrilinosWrappers::MPI::Vector &dst,
      const dealii::TrilinosWrappers::MPI::Vector &src) const;

  unsigned int compute_lower_interface_dof_count() override;

  unsigned int compute_upper_interface_dof_count() override;

  void solve(dealii::Vector<double> src, dealii::Vector<double> &dst) override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  unsigned int compute_own_dofs() override;

  void initialize_own_dofs() override;

  void make_constraints() override;

  void run() override;

  void solve();

  void assemble() override;

  void initialize_index_sets() override;

  void apply_sweep(dealii::LinearAlgebra::distributed::Vector<double>);

  LocalProblem* get_local_problem() override;
};


#endif //WAVEGUIDEPROBLEM_LOCALPROBLEM_H
