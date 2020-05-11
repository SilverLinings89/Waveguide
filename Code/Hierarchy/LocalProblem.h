//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_LOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_LOCALPROBLEM_H


#include "HierarchicalProblem.h"
#include "../Core/DOFManager.h"

class LocalProblem: public HierarchicalProblem {
public:
  HSIESurface *hsie_surfaces;

  LocalProblem(unsigned int, unsigned int, DOFManager *dm);

  void solve() override;

  void initialize() override;

  void generate_sparsity_pattern() override;

  dealii::IndexSet get_owned_dofs_for_level(unsigned int level) override;

  LocalProblem* get_local_problem() override;
};


#endif //WAVEGUIDEPROBLEM_LOCALPROBLEM_H
