//
// Created by pascal on 14.11.19.
//

#ifndef WAVEGUIDEPROBLEM_DOFDATA_H
#define WAVEGUIDEPROBLEM_DOFDATA_H

enum DofType {
    EDGE, SURFACE, RAY, IFFa, IFFb, SEGMENTa, SEGMENTb
};

struct DofData {
    DofType type;
    int hsie_order;
    int inner_order;
    bool nodal_basis;
    bool is_real;
    unsigned int global_index;
    bool got_base_dof_index;
    unsigned int base_dof_index; // The basis functions require either an edge or hat function for computation of some components. This value names the exact number of that dof.
    std::string base_structure_id_face;
    unsigned int base_structure_id_non_face;

    DofData() {
        base_structure_id_face = "";
        base_structure_id_non_face = 0;
        this->got_base_dof_index = false;
    }

    void set_base_dof(unsigned int in_base_dof_index) {
        this->got_base_dof_index = true;
        this->base_dof_index = in_base_dof_index;
    }

    DofData(std::string in_id) {
        this->got_base_dof_index = false;
        base_structure_id_face = in_id;
    }

    DofData(unsigned int in_id) {
        this->got_base_dof_index = false;
        base_structure_id_non_face = in_id;
    }

    void update_nodal_basis_flag() {
        this->nodal_basis = (this->type == DofType::RAY || this->type == DofType::IFFb || this->type == DofType::SEGMENTb);
    }
};

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <utility>
#include "HSIEPolynomial.h"

#endif //WAVEGUIDEPROBLEM_DOFDATA_H
