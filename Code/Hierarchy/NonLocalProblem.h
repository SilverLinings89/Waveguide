//
// Created by pascal on 03.02.20.
//

#ifndef WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
#define WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H

#include <mpi.h>
#include "HierarchicalProblem.h"
#include "../Helpers/Parameters.h"

class NonLocalProblem : public HierarchicalProblem{
    unsigned int compute_own_dofs();
    MPI_Comm level_communicator;
    Parameters * params;
    void initialize_MPI_communicator_for_level();

public:
    NonLocalProblem(unsigned int, unsigned int, DOFManager *, MPI_Comm, Parameters *);
    void compute_level_dofs_total() override;

    void solve() override;

    void initialize() override;

    void generateSparsityPattern() override;

    void initialize_index_sets() override;
};


#endif //WAVEGUIDEPROBLEM_NONLOCALPROBLEM_H
