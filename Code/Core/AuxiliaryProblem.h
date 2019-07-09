/*
 * AuxiliaryProblem.h
 * Objects of this kind only solve a local auxiliary problems. As a consequence
 * they don't communicate via MPI and rely on direct solvers. There will be
 * options for how this should be done since auxiliary problems occur in both
 * the Hardy Sweeping and in the Moving PML Sweeping Ansatz.
 * An important difference is the following: The Sweeping preconditioner
 * extracts the local contributions from the global vector and can then proceed
 * its work in local numbering. For that reason, the auxiliary problem does NOT
 * have to ever number its degrees of freedom globally. This makes this object
 * very efficient in the run. \date Jun 24, 2019 \author Pascal Kraft
 */

#ifndef CODE_CORE_AUXILIARYPROBLEM_H_
#define CODE_CORE_AUXILIARYPROBLEM_H_

#include "NumericProblem.h"
#include "../MeshGenerators/SquareMeshGenerator.h"

class AuxiliaryProblem : public NumericProblem {
public:
    AuxiliaryProblem(MPI_Comm inMpiComm, SquareMeshGenerator *inMg, SpaceTransformation *inSt);

    virtual ~AuxiliaryProblem();

    void PrepareMesh();

    void PrepareBoundaryConstraints();

    void AssembleMatricesAndRHS();

    void Solve();

    void reset();
};

#endif /* CODE_CORE_AUXILIARYPROBLEM_H_ */
