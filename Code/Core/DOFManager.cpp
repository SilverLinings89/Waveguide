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

void NumericProblem::SortDofsDownstream() {
    std::vector<DofSortData> dofs_sort_objects;
    DoFHandler<3>::active_cell_iterator cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    std::vector<unsigned int> lines_touched;
    std::vector<unsigned int> cells_touched;
    std::vector<unsigned int> faces_touched;
    std::vector<types::global_dof_index> local_line_dofs(fe.dofs_per_line);
    std::vector<types::global_dof_index> local_face_dofs(fe.dofs_per_face);
    std::vector<types::global_dof_index> local_cell_dofs(fe.dofs_per_cell);
    for (; cell != endc; ++cell) {
        if (!(std::find(cells_touched.begin(), cells_touched.end(),
                        cell->index()) !=
                cells_touched.end())) {
            cell->get_dof_indices()

        }
        for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
            for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
                if (!(std::find(lines_touched.begin(), lines_touched.end(),
                                cell->face(i)->line(j)->index()) !=
                      lines_touched.end())) {
                    ((cell->face(i))->line(j))->get_dof_indices(local_line_dofs);
                    for (unsigned k = 0; k < local_line_dofs.size(); k++) {
                        dofs_sort_objects.push_back(std::pair<int, Point<3,double>>(
                                local_line_dofs[k], (cell->face(i))->line(j)->center()));
                    }
                    lines_touched.push_back(cell->face(i)->line(j)->index());
                }
            }
        }
    }
    std::sort(dofs_sort_objects.begin(), dofs_sort_objects.end(), compareIndexCenterPairs);
    std::vector<unsigned int> new_numbering;
    new_numbering.resize(dofs_sort_objects.size());
    for (unsigned int i = 0; i < dofs_sort_objects.size(); i++) {
        new_numbering[dofs_sort_objects[i].first] = i;
    }
    dof_handler.renumber_dofs(new_numbering);
}

bool compareIndexCenterPairs(std::pair<int, double> c1,
                             std::pair<int, double> c2) {
    return c1.second < c2.second;
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
