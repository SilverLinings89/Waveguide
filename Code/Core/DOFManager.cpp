//
// Created by kraft on 16.07.19.
//

#include "DOFManager.h"
#include "NumericProblem.h"

void DOFManager::compute_and_communicate_edge_dofs() {
    if (GlobalParams.Blocks_in_x_direction > 1 && GlobalParams.Blocks_in_y_direction > 1) {

    } else {
        if (GlobalParams.Blocks_in_y_direction > 1) {

        } else {

        }
    }
}

void DOFManager::MPI_build_global_index_set_vector() {
    this->own_dofs_per_process = dealii::Utilities::MPI::create_ascending_partitioning(MPI_COMM_WORLD,
                                                                                       this->n_locally_owned_dofs);
    this->local_dofs = this->own_dofs_per_process[GlobalParams.MPI_Rank];
    this->n_global_dofs = this->own_dofs_per_process[0].size();
    // TODO: add the not locally owned dofs to this set.
}

void DOFManager::init() {
    this->dof_handler->distribute_dofs(*fe);
}

struct DofSortData {
    types::global_dof_index index;
    unsigned short order;
    Point<3,double> base_point;
};

bool compareIndexCenterPairs(DofSortData c1,
                             DofSortData c2) {
    //return c1.second < c2.second;
    if(c1.base_point[2] < c2.base_point[2]) {
        return true;
    }
    if(c1.base_point[2] > c2.base_point[2]) {
        return false;
    }
    if(c1.base_point[1] < c2.base_point[1]) {
        return true;
    }
    if(c1.base_point[1] > c2.base_point[1]) {
        return false;
    }
    if(c1.base_point[0] < c2.base_point[0]) {
        return true;
    }
    if(c1.base_point[0] > c2.base_point[0]) {
        return false;
    }
    if(c1.order < c2.order) {
        return true;
    }
    if(c1.order > c2.order) {
        return false;
    }
    std::cout << "There was an error in dof sorting - two object were undistinguishable." <<std::endl;
}

void DOFManager::SortDofs() {
    std::vector<DofSortData> dofs_sort_objects;
    DoFHandler<3>::active_cell_iterator cell = dof_handler->begin_active(),
            endc = dof_handler->end();
    const unsigned int n_local_dofs = dof_handler->n_dofs();
    bool * dof_touched = new bool[n_local_dofs];
    for(unsigned int i = 0; i < n_local_dofs; i++) {
        dof_touched[i] = false;
    }
    short line_order = 0;
    short face_order = 0;
    short cell_order = 0;
    std::vector<types::global_dof_index> local_line_dofs(fe->dofs_per_line);
    std::vector<types::global_dof_index> local_face_dofs(fe->dofs_per_face);
    std::vector<types::global_dof_index> local_cell_dofs(fe->dofs_per_cell);
    for (; cell != endc; ++cell) {
        cell_order = 0;
        cell->get_dof_indices(local_cell_dofs);
        for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
            face_order = 0;
            cell->face(i)->get_dof_indices(local_face_dofs);
            for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
                cell->face(i)->line(j)->get_dof_indices(local_line_dofs);
                line_order = 0;
                for (unsigned k = 0; k < local_line_dofs.size(); k++) {
                    if (!dof_touched[local_line_dofs[k]]) {
                        DofSortData temp;
                        temp.base_point = (cell->face(i))->line(j)->center();
                        temp.index = local_line_dofs[k];
                        temp.order = line_order;
                        dofs_sort_objects.push_back(temp);
                        line_order++;
                        dof_touched[local_line_dofs[k]];
                    }
                }
            }
            for(unsigned int k = 0; k < local_face_dofs.size(); k++) {
                if(!dof_touched[local_face_dofs[k]]) {
                    DofSortData temp;
                    temp.base_point = cell->face(i)->center();
                    temp.index = local_face_dofs[k];
                    temp.order = face_order;
                    dof_touched[local_face_dofs[k]] = true;
                    face_order ++;
                }
            }
        }
        for(unsigned int k = 0; k < local_cell_dofs.size(); k++) {
            if(!dof_touched[local_cell_dofs[k]]) {
                DofSortData temp;
                temp.base_point = cell->center();
                temp.index = local_cell_dofs[k];
                temp.order = cell_order;
                dof_touched[local_cell_dofs[k]] = true;
                cell_order ++;
            }
        }
    }
    std::sort(dofs_sort_objects.begin(), dofs_sort_objects.end(), compareIndexCenterPairs);
    std::vector<unsigned int> new_numbering;
    new_numbering.resize(dofs_sort_objects.size());
    for (unsigned int i = 0; i < dofs_sort_objects.size(); i++) {
        new_numbering[dofs_sort_objects[i].index] = i;
    }
    dof_handler->renumber_dofs(new_numbering);
}



unsigned int DOFManager::compute_n_own_dofs() {
    return 0;
}

DOFManager::DOFManager(unsigned int i_dofs_per_cell, unsigned int i_dofs_per_face, unsigned int i_dofs_per_edge,
                       dealii::DoFHandler<3, 3> *in_dof_handler, dealii::Triangulation<3, 3> *in_triangulation,
                       const dealii::FiniteElement<3, 3> *in_fe) : dofs_per_cell(i_dofs_per_cell),
                                                              dofs_per_edge(i_dofs_per_edge),
                                                              dofs_per_face(i_dofs_per_face),
                                                              fe(in_fe){
    this->triangulation = in_triangulation;
    this->dof_handler = in_dof_handler;
}
