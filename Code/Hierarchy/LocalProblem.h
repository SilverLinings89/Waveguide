//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_LOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_LOCALPROBLEM_H


#include "HierarchicalProblem.h"
#include "../Core/DOFManager.h"
#include "../Core/NumericProblem.h"

class LocalProblem: public HierarchicalProblem {
public:

  NumericProblem base_problem;
  HSIESurface *surface_0;
  HSIESurface *surface_1;
  HSIESurface *surface_2;
  HSIESurface *surface_3;
  HSIESurface *surface_4;
  HSIESurface *surface_5;

  LocalProblem(unsigned int, unsigned int);
  ~LocalProblem() override;

  unsigned int compute_lower_interface_dof_count() override;

  unsigned int compute_upper_interface_dof_count() override;

  void solve() override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  unsigned int compute_own_dofs() override;

  void assemble() override;

  void initialize_index_sets() override;

  void apply_sweep(dealii::LinearAlgebra::distributed::Vector<double>);

  dealii::IndexSet get_owned_dofs_for_level(unsigned int level) override;

  LocalProblem* get_local_problem() override;
};


#endif //WAVEGUIDEPROBLEM_LOCALPROBLEM_H
