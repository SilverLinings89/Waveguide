//
// Created by kraft on 16.07.19.
//

#include "DOFManager.h"
#include "NumericProblem.h"
#include "mpi.h"

using namespace dealii;

void DOFManager::compute_and_send_x_dofs() {
    if(GlobalParams.Blocks_in_x_direction > 1) {
        if(GlobalParams.Index_in_x_direction < GlobalParams.Blocks_in_x_direction-1){
            IndexSet boundary_dofs = get_dofs_for_boundary_id(1);
            int * bdof_vals = new int[boundary_dofs.n_elements()];
            for(unsigned int j = 0; j < boundary_dofs.n_elements(); j++) {
                bdof_vals[j] = boundary_dofs.nth_index_in_set(j);
            }
            MPI_Send(bdof_vals, boundary_dofs.n_elements(), MPI_UNSIGNED, GlobalParams.geometry.get_neighbor_for_interface(Direction::PlusX).second, 0, MPI_COMM_WORLD );
        }
    }
}

void DOFManager::compute_and_send_y_dofs() {
    if(GlobalParams.Blocks_in_y_direction > 1 ) {
        if (GlobalParams.Index_in_y_direction < GlobalParams.Blocks_in_y_direction -1) {
                IndexSet boundary_dofs = get_dofs_for_boundary_id(3);
                int *bdof_vals = new int[boundary_dofs.n_elements()];
                for (unsigned int j = 0; j < boundary_dofs.n_elements(); j++) {
                    bdof_vals[j] = boundary_dofs.nth_index_in_set(j);
                }
                MPI_Send(bdof_vals, boundary_dofs.n_elements(), MPI_UNSIGNED,
                         GlobalParams.geometry.get_neighbor_for_interface(Direction::PlusY).second, 0, MPI_COMM_WORLD);
        }
    }
}

void DOFManager::compute_and_send_z_dofs() {
    if(GlobalParams.Blocks_in_z_direction > 1) {
        if(GlobalParams.Index_in_z_direction < GlobalParams.Blocks_in_z_direction -1){
            if( GlobalParams.Index_in_z_direction != GlobalParams.Coupling_Interface_Z_Block_Index-1) {

                IndexSet boundary_dofs = get_dofs_for_boundary_id(5);
                int *bdof_vals = new int[boundary_dofs.n_elements()];
                for (unsigned int j = 0; j < boundary_dofs.n_elements(); j++) {
                    bdof_vals[j] = boundary_dofs.nth_index_in_set(j);
                }
                MPI_Send(bdof_vals, boundary_dofs.n_elements(), MPI_UNSIGNED,
                         GlobalParams.geometry.get_neighbor_for_interface(Direction::PlusZ).second, 0, MPI_COMM_WORLD);
            }
        }
    }
}

void DOFManager::receive_x_dofs() {
    if(GlobalParams.Blocks_in_x_direction > 1) {
        if(GlobalParams.Index_in_x_direction > 0) {
            IndexSet boundary_dofs = get_dofs_for_boundary_id(0);
            int * bdof_vals = new int[boundary_dofs.n_elements()];
            MPI_Recv(bdof_vals, boundary_dofs.n_elements(), MPI_UNSIGNED, GlobalParams.geometry.get_neighbor_for_interface(Direction::MinusX).second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            IndexSet new_numbers;
            new_numbers.set_size(n_global_dofs);
            for (unsigned int i = 0; i < boundary_dofs.n_elements(); i++) {
                new_numbers.add_index(bdof_vals[i]);
            }
            update_interface_dofs_with_IndexSet(new_numbers,0);
        }
    }
}

void DOFManager::receive_y_dofs() {
    if(GlobalParams.Blocks_in_y_direction > 1 ) {
        if (GlobalParams.Index_in_y_direction > 0) {
            IndexSet boundary_dofs = get_dofs_for_boundary_id(2);
            int *bdof_vals = new int[boundary_dofs.n_elements()];
            MPI_Recv(bdof_vals, boundary_dofs.n_elements(), MPI_UNSIGNED, GlobalParams.geometry.get_neighbor_for_interface(Direction::MinusY).second, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            IndexSet new_numbers;
            new_numbers.set_size(n_global_dofs);
            for (unsigned int i = 0; i < boundary_dofs.n_elements(); i++) {
                new_numbers.add_index(bdof_vals[i]);
            }
            update_interface_dofs_with_IndexSet(new_numbers,2);
        }
    }
}

void DOFManager::receive_z_dofs() {
    if(GlobalParams.Blocks_in_z_direction > 1) {
        if(GlobalParams.Index_in_z_direction > 0){
            if(GlobalParams.Index_in_z_direction != GlobalParams.Coupling_Interface_Z_Block_Index) {
                IndexSet boundary_dofs = get_dofs_for_boundary_id(4);
                int *bdof_vals = new int[boundary_dofs.n_elements()];
                MPI_Recv(bdof_vals, boundary_dofs.n_elements(), MPI_UNSIGNED,
                         GlobalParams.geometry.get_neighbor_for_interface(Direction::MinusZ).second, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                IndexSet new_numbers;
                new_numbers.set_size(n_global_dofs);
                for (unsigned int i = 0; i < boundary_dofs.n_elements(); i++) {
                    new_numbers.add_index(bdof_vals[i]);
                }
                update_interface_dofs_with_IndexSet(new_numbers, 4);
            }
        }
    }
}

void DOFManager::update_interface_dofs_with_IndexSet(IndexSet in_new_indices, types::boundary_id in_bid) {
    IndexSet local_numbering = get_dofs_for_boundary_id(in_bid);
    if(in_new_indices.n_elements() != local_numbering.n_elements()) {
        std::cout << "There was a communication problem in update_interface_dofs_with_IndexSet."<<std::endl;
    }
    IndexSet all_local_dofs = dof_handler->locally_owned_dofs();
    std::vector<unsigned int> new_numbering;
    for(unsigned int i = 0; i < all_local_dofs.size(); i++) {
        unsigned int index = all_local_dofs.nth_index_in_set(i);
        if(local_numbering.is_element(index)) {
            new_numbering.push_back(in_new_indices.nth_index_in_set(local_numbering.is_element(index)));
        } else {
            new_numbering.push_back(index);
        }
    }
    dof_handler->renumber_dofs(new_numbering);
}

void DOFManager::shift_own_to_final_dof_numbers() {
    dealii::IndexSet non_owned_dofs = get_non_owned_dofs();
    std::vector<unsigned int> shifted_numbering;
    unsigned int counter = 0;
    for (unsigned int i = 0; i < dof_handler->n_dofs(); i++) {
        if(! non_owned_dofs.is_element(i)) {
            shifted_numbering.push_back(local_dofs.nth_index_in_set(i));
            counter ++;
        } else {
            shifted_numbering.push_back(i);
        }
    }
    dof_handler->renumber_dofs(shifted_numbering);
}

IndexSet DOFManager::get_dofs_for_boundary_id(types::boundary_id in_bid) {
    IndexSet ret;
    ret.set_size(this->n_global_dofs);
    DoFHandler<3>::active_cell_iterator cell = dof_handler->begin_active(),
            endc = dof_handler->end();

    std::vector<types::global_dof_index> local_face_dofs(fe->dofs_per_face);
    for (; cell != endc; ++cell) {
        for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
            if (cell->face(i)->boundary_id() == in_bid) {
                cell->face(i)->get_dof_indices(local_face_dofs);
                ret.add_indices(local_face_dofs.begin(), local_face_dofs.end());
            }
        }
    }
    return ret;
}

void DOFManager::MPI_build_global_index_set_vector() {
    this->own_dofs_per_process = dealii::Utilities::MPI::create_ascending_partitioning(MPI_COMM_WORLD,
                                                                                       this->n_locally_owned_dofs);
    this->local_dofs = this->own_dofs_per_process[GlobalParams.MPI_Rank];
    this->n_global_dofs = this->own_dofs_per_process[0].size();
    computed_n_global = true;
}

void DOFManager::init() {
    this->dof_handler->distribute_dofs(*fe);
    SortDofs();
    compute_n_own_dofs();
    MPI_build_global_index_set_vector();
    shift_own_to_final_dof_numbers();
    receive_z_dofs();
    compute_and_send_z_dofs();
    receive_y_dofs();
    compute_and_send_y_dofs();
    receive_x_dofs();
    compute_and_send_x_dofs();
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
    return false;
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
                        dof_touched[local_line_dofs[k]] = true;
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

IndexSet DOFManager::get_non_owned_dofs() {
    IndexSet non_owned_dofs;
    if(computed_n_global) {
        non_owned_dofs.set_size(n_global_dofs);
    } else {
        non_owned_dofs.set_size(GlobalParams.NumberProcesses *
                                dof_handler->n_dofs()); // rough estimate that guarantees that all entries in this index set will be stored.
    }
    std::vector<types::global_dof_index> local_face_dofs(fe->dofs_per_face);
    DoFHandler<3>::active_cell_iterator cell = dof_handler->begin_active(),
            endc = dof_handler->end();
    for (; cell != endc; ++cell) {
        if(cell->at_boundary()) {
            for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
                types::boundary_id f_bid = cell->face(i)->boundary_id();
                if (f_bid == 0 && GlobalParams.Index_in_x_direction > 0) {
                    cell->face(i)->get_dof_indices(local_face_dofs);
                    for (unsigned int j = 0; j < fe->dofs_per_face; j++) {
                        non_owned_dofs.add_index(local_face_dofs[j]);
                    }
                }
                if (f_bid == 2 && GlobalParams.Index_in_y_direction > 0) {
                    cell->face(i)->get_dof_indices(local_face_dofs);
                    for (unsigned int j = 0; j < fe->dofs_per_face; j++) {
                        non_owned_dofs.add_index(local_face_dofs[j]);
                    }
                }
                if (f_bid == 4 && GlobalParams.Index_in_z_direction > 0) {
                    cell->face(i)->get_dof_indices(local_face_dofs);
                    for (unsigned int j = 0; j < fe->dofs_per_face; j++) {
                        non_owned_dofs.add_index(local_face_dofs[j]);
                    }
                }
            }
        }
    }

    IndexSet InterfaceDofs(n_global_dofs);
    if(GlobalParams.Index_in_z_direction == GlobalParams.Coupling_Interface_Z_Block_Index) {
        cell = dof_handler->begin_active();
        for (; cell != endc; ++cell) {
            if(cell->at_boundary()) {
                for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
                    types::boundary_id f_bid = cell->face(i)->boundary_id();
                    if (f_bid == 4 && GlobalParams.Index_in_z_direction > 0) {
                        cell->face(i)->get_dof_indices(local_face_dofs);
                        for (unsigned int j = 0; j < fe->dofs_per_face; j++) {
                            InterfaceDofs.add_index(local_face_dofs[j]);
                        }
                    }
                }
            }
        }
    }
    non_owned_dofs.subtract_set(InterfaceDofs);
    return non_owned_dofs;
}

unsigned int DOFManager::compute_n_own_dofs() {
    n_locally_owned_dofs = dof_handler->n_dofs();
    dealii::IndexSet non_owned = get_non_owned_dofs();
    n_locally_owned_dofs = dof_handler->n_dofs() - non_owned.n_elements();
    return 0;
}

DOFManager::DOFManager(unsigned int i_dofs_per_cell, unsigned int i_dofs_per_face, unsigned int i_dofs_per_edge,
                       dealii::DoFHandler<3, 3> *in_dof_handler, dealii::Triangulation<3, 3> *in_triangulation,
                       const dealii::FiniteElement<3, 3> *in_fe) :
                            dofs_per_edge(i_dofs_per_edge),
                            dofs_per_face(i_dofs_per_face),
                            dofs_per_cell(i_dofs_per_cell),
                            fe(in_fe){
    this->triangulation = in_triangulation;
    this->dof_handler = in_dof_handler;
    computed_n_global = false;
}

int DOFManager::testValue() {
    return 4;
}
