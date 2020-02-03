//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_LOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_LOCALPROBLEM_H


#include "HierarchicalProblem.h"
#include "../Core/DOFManager.h"

class LocalProblem : public HierarchicalProblem {
public:
    LocalProblem(unsigned int, unsigned int, DOFManager * dm, Parameters * parameters);

    unsigned int compute_own_dofs();

    void compute_level_dofs_total() override;

    void solve() override;

    void initialize() override;

    void generateSparsityPattern() override;
};


#endif //WAVEGUIDEPROBLEM_LOCALPROBLEM_H
