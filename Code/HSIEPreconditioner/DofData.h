//
// Created by pascal on 14.11.19.
//

#ifndef WAVEGUIDEPROBLEM_DOFDATA_H
#define WAVEGUIDEPROBLEM_DOFDATA_H

enum DofType {
    EDGE, SURFACE, RAY, IFFa, IFFb, SEGMENTa, SEGMENTb
};

union DofBaseStructureID {
    std::string face_id;
    unsigned int non_face_id;

    DofBaseStructureID() {};
    ~DofBaseStructureID() {};
};

struct DofData {
    DofType type;
    int hsie_order;
    int inner_order;
    bool is_real;
    unsigned int global_index;
    DofBaseStructureID base_structure_id;

    DofData() {
        base_structure_id.face_id = "";
    }

    DofData(std::string in_id) {
        base_structure_id.face_id = in_id;
    }

    DofData(unsigned int in_id) {
        base_structure_id.non_face_id = in_id;
    }
};

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include "HSIEPolynomial.h"

#endif //WAVEGUIDEPROBLEM_DOFDATA_H
