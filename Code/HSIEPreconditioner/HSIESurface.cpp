//
// Created by kraft on 16.08.19.
//

#include "HSIESurface.h"
#include "../Core/NumericProblem.h"
#include "HSIEPolynomial.h"
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_accessor.h>
#include "DofData.h"

const unsigned int MAX_DOF_NUMBER = INT_MAX;


template<unsigned int ORDER>
HSIESurface<ORDER>::HSIESurface( dealii::Triangulation<2,2> & in_surface_triangulation, unsigned int in_boundary_id,
                                unsigned int in_level, unsigned int in_inner_order, std::complex<double> in_k0, std::map<dealii::Triangulation<2,3>::cell_iterator, dealii::Triangulation<3,3>::face_iterator > in_assoc):
        b_id(in_boundary_id),
        dof_h_nedelec(in_surface_triangulation),
        dof_h_q(in_surface_triangulation),
        Inner_Element_Order(in_inner_order),
        fe_nedelec(Inner_Element_Order),
        fe_q(Inner_Element_Order+1),
        level(in_level)
{
    association = in_assoc;
    surface_triangulation.copy_triangulation(in_surface_triangulation);
    dof_counter = 0;
    k0 = in_k0;
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::compute_dof_numbers() {
    this->n_edge_dofs = compute_n_edge_dofs();
    this->n_face_dofs = compute_n_face_dofs();
    this->n_vertex_dofs = compute_n_vertex_dofs();
}

template<unsigned int ORDER>
std::vector<DofData> HSIESurface<ORDER>::get_dof_data_for_cell(dealii::DoFHandler<2>::active_cell_iterator cell_nedelec, dealii::DoFHandler<2>::active_cell_iterator cell_q) {
    std::vector<DofData> ret;

    // get cell dofs:
    std::string cell_id =  cell_nedelec->id().to_string();
    unsigned int * nedelec_edge_ids = new unsigned int[4];
    unsigned int * q_edge_ids = new unsigned int[4];
    unsigned int * vertex_ids = new unsigned int[4];
    // get edge dofs:
    for(unsigned int i = 0; i < 4; i++) {
        nedelec_edge_ids[i] = cell_nedelec->face_index(i);
    }

    for(unsigned int i = 0; i < 4; i++) {
        q_edge_ids[i] = cell_q->face_index(i);
    }

    // get vertex dofs:
    for(unsigned int i = 0; i < 4; i++) {
        vertex_ids[i]= cell_q->vertex_index(i);
    }

    // add cell dofs
    for(unsigned int i = 0; i < this->face_dof_data.size(); i++) {
        if(this->face_dof_data[i].base_structure_id_face == cell_id) {
            ret.push_back(this->face_dof_data[i]);
        }
    }

    // add edge-based dofs
    for(unsigned int i = 0; i < this->edge_dof_data.size(); i++) {
        if(this->edge_dof_data[i].type == DofType::EDGE || this->edge_dof_data[i].type == DofType::IFFa) {
            unsigned int idx = this->edge_dof_data[i].base_structure_id_non_face;
            if (idx == nedelec_edge_ids[0] || idx == nedelec_edge_ids[1] || idx == nedelec_edge_ids[2] || idx == nedelec_edge_ids[3]) {
                ret.push_back(this->edge_dof_data[i]);
            }
        } else {
            unsigned int idx = this->edge_dof_data[i].base_structure_id_non_face;
            if (idx == q_edge_ids[0] || idx == q_edge_ids[1] || idx == q_edge_ids[2] || idx == q_edge_ids[3]) {
                ret.push_back(this->edge_dof_data[i]);
            }
        }
    }

    // add vertex-based dofs
    for(unsigned int i = 0; i < this->vertex_dof_data.size(); i++) {
        unsigned int idx = this->vertex_dof_data[i].base_structure_id_non_face;
        if(idx == vertex_ids[0] || idx == vertex_ids[1] || idx == vertex_ids[2] || idx == vertex_ids[3]) {
            ret.push_back(this->vertex_dof_data[i]);
        }
    }

    return ret;
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::fill_matrix(dealii::SparseMatrix<double> * matrix, dealii::IndexSet global_indices) {
    HSIEPolynomial::computeDandI(ORDER + 2, k0);
    auto it = dof_h_nedelec.begin_active();
    auto end = dof_h_nedelec.end();

    QGauss<2> quadrature_formula(2);
    FEValues<2,2> fe_q_values(fe_q, quadrature_formula,
                            update_values | update_gradients | update_JxW_values |
                            update_quadrature_points);
    FEValues<2,2> fe_n_values(fe_nedelec, quadrature_formula,
                            update_values | update_gradients | update_JxW_values |
                            update_quadrature_points);
    std::vector<Point<2>> quadrature_points;
    const unsigned int dofs_per_cell = GeometryInfo<2>::vertices_per_cell * compute_dofs_per_vertex() + GeometryInfo<2>::lines_per_cell * compute_dofs_per_edge(false) + compute_dofs_per_face(false);
    FullMatrix<double> cell_matrix_real(dofs_per_cell, dofs_per_cell);
    unsigned int cell_counter = 0;
    auto it2 = dof_h_q.begin_active();
    for(; it != end; ++it) {
        cell_matrix_real = 0;
        std::vector<DofData> cell_dofs = this->get_dof_data_for_cell(it, it2);
        std::vector<HSIEPolynomial> polynomials;
        std::vector<unsigned int> q_dofs(fe_q.dofs_per_cell);
        std::vector<unsigned int> n_dofs(fe_nedelec.dofs_per_cell);;
        it2->get_dof_indices(q_dofs);
        it->get_dof_indices(n_dofs);
        for(unsigned int i = 0; i < cell_dofs.size(); i++) {
            polynomials.push_back(HSIEPolynomial(cell_dofs[i], k0));
        }
        std::vector<unsigned int> local_related_fe_index;
        for(unsigned int i = 0; i < cell_dofs.size(); i++) {
            if(cell_dofs[i].type == DofType::RAY) {
                for(unsigned int j= 0; j < q_dofs.size(); j++) {
                    if(q_dofs[j] == cell_dofs[i].base_dof_index) {
                        local_related_fe_index.push_back(j);
                        break;
                    }
                }
            } else {
                for(unsigned int j= 0; j < n_dofs.size(); j++) {
                    if(n_dofs[j] == cell_dofs[i].base_dof_index) {
                        local_related_fe_index.push_back(j);
                        break;
                    }
                }
            }
        }

        fe_n_values.reinit(it);
        fe_q_values.reinit(it2);
        quadrature_points = fe_q_values.get_quadrature_points();
        std::vector<double> jxw_values = fe_n_values.get_JxW_values();

        for(unsigned int q_point = 0; q_point < quadrature_points.size(); q_point++) {
            double JxW = jxw_values[q_point];
            for (unsigned int i = 0; i < cell_dofs.size(); i++) {
                DofData &u = cell_dofs[i];
                std::vector<HSIEPolynomial> u_contrib_curl, u_contrib;
                if(cell_dofs[i].type == DofType::RAY || cell_dofs[i].type == DofType::IFFb) {
                    u_contrib_curl = build_curl_term_q(u.hsie_order, fe_q_values.shape_grad(local_related_fe_index[i], q_point));
                    u_contrib = build_non_curl_term_q(u.hsie_order, fe_q_values.shape_value(local_related_fe_index[i], q_point));
                } else {
                    u_contrib_curl = build_curl_term_nedelec(u.hsie_order,
                            fe_n_values.shape_grad_component(local_related_fe_index[i],q_point,0),
                            fe_n_values.shape_grad_component(local_related_fe_index[i],q_point,1),
                            fe_n_values.shape_value_component(local_related_fe_index[i],q_point,0),
                            fe_n_values.shape_value_component(local_related_fe_index[i],q_point,1));
                    u_contrib = build_non_curl_term_nedelec(u.hsie_order,
                            fe_n_values.shape_value_component(local_related_fe_index[i],q_point,0),
                            fe_n_values.shape_value_component(local_related_fe_index[i],q_point,1));
                }

                for (unsigned int j = 0; j < cell_dofs.size(); j++) {
                    DofData &v = cell_dofs[j];
                    std::vector<HSIEPolynomial> v_contrib_curl, v_contrib;

                    if(cell_dofs[j].type == DofType::RAY || cell_dofs[j].type == DofType::IFFb) {
                        v_contrib_curl = build_curl_term_q(v.hsie_order, fe_q_values.shape_grad(local_related_fe_index[j], q_point));
                        v_contrib = build_non_curl_term_q(v.hsie_order, fe_q_values.shape_value(local_related_fe_index[j], q_point));
                    } else {
                        v_contrib_curl = build_curl_term_nedelec(v.hsie_order,
                                                                 fe_n_values.shape_grad_component(local_related_fe_index[j],q_point,0),
                                                                 fe_n_values.shape_grad_component(local_related_fe_index[j],q_point,1),
                                                                 fe_n_values.shape_value_component(local_related_fe_index[j],q_point,0),
                                                                 fe_n_values.shape_value_component(local_related_fe_index[j],q_point,1));
                        v_contrib = build_non_curl_term_nedelec(v.hsie_order,
                                                                fe_n_values.shape_value_component(local_related_fe_index[j],q_point,0),
                                                                fe_n_values.shape_value_component(local_related_fe_index[j],q_point,1));
                    }
                    cell_matrix_real.set(i, j,((evaluate_a(u_contrib_curl, v_contrib_curl, 100) + evaluate_a(u_contrib, v_contrib, 100)) * JxW).real());
                }
            }
            std::vector<unsigned int> local_indices;
            for(unsigned int i = 0; i < cell_dofs.size(); i++) {
                local_indices.push_back(global_indices.nth_index_in_set(cell_dofs[i].global_index));
            }
            matrix->set(local_indices, cell_matrix_real);
        }
        it2++;
        cell_counter ++;
    }

}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_edge_dofs() {
    DoFHandler<2>::active_cell_iterator cell, cell2, endc;
    endc = dof_h_nedelec.end();
    DofCount ret;
    cell2 = dof_h_q.begin_active();
    for(cell = dof_h_nedelec.begin_active(); cell != endc; cell++) {
        for(unsigned int edge = 0; edge < GeometryInfo<2>::lines_per_cell; edge++) {
            if(!cell->line(edge)->user_flag_set()){
                update_dof_counts_for_edge(cell, edge, ret);
                register_new_edge_dofs(cell, cell2, edge);
                cell->line(edge)->set_user_flag();
            }
        }
        cell2++;
    }
    return ret;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_vertex_dofs() {
    std::set<unsigned int> touched_vertices;
    DoFHandler<2>::active_cell_iterator cell, endc;
    endc = dof_h_q.end();
    DofCount ret;
    for(cell = dof_h_q.begin_active(); cell != endc; cell++) {
        // for each edge
        for(unsigned int vertex = 0; vertex < GeometryInfo<2>::vertices_per_cell; vertex++) {
            unsigned int idx = cell->vertex_dof_index(vertex, 0);
            if(touched_vertices.end() == touched_vertices.find(idx)) {
                // handle it
                update_dof_counts_for_vertex(cell, idx, vertex, ret);
                register_new_vertex_dofs(cell, idx, vertex);
                // remember that it has been handled
                touched_vertices.insert(idx);
            }
        }
    }
    return ret;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_face_dofs() {
    std::set<std::string> touched_faces;
    DoFHandler<2>::active_cell_iterator cell,cell2, endc;
    endc = dof_h_nedelec.end();
    DofCount ret;
    // for each cell
    cell2 = dof_h_q.begin_active();
    for(cell = dof_h_nedelec.begin_active(); cell != endc; cell++) {
        // if it wasn't handled before
        if(touched_faces.end() == touched_faces.find(cell->id().to_string())) {
            // handle it
            update_dof_counts_for_face(cell, ret);
            register_new_surface_dofs(cell, cell2);
            // remember that it has been handled
            touched_faces.insert(cell->id().to_string());
        }
        cell2++;
    }
    return ret;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_edge(bool only_hsie_dofs) {
    unsigned int ret = 0;
    const unsigned int INNER_REAL_DOFS_PER_LINE = fe_nedelec.dofs_per_line;

    // Number of inner dofs 1.
    if(! only_hsie_dofs) {
      ret += INNER_REAL_DOFS_PER_LINE;
    }

    //Number of infinite face functions 4a and 4b.
    ret += INNER_REAL_DOFS_PER_LINE * (ORDER+1) + (INNER_REAL_DOFS_PER_LINE-1)*(ORDER+2);

    // everything double for real and imaginary part.
    ret *= 2;
    return ret;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_face(bool only_hsie_dofs) {
    unsigned int ret = 0;
    const unsigned int INNER_REAL_NEDELEC_DOFS_PER_FACE = fe_nedelec.dofs_per_cell - dealii::GeometryInfo<2>::faces_per_cell * fe_nedelec.dofs_per_face;

    ret = INNER_REAL_NEDELEC_DOFS_PER_FACE * (ORDER + 2) * 3;
    if(only_hsie_dofs) {
        ret -= INNER_REAL_NEDELEC_DOFS_PER_FACE;
    }
    return ret*2;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_vertex() {
    // Number of elements of type 3
    unsigned int ret = ORDER + 2;

    // Real and imaginary part.
    ret *= 2;
    return ret;
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::initialize() {
    // prepare_surface_triangulation();
    initialize_dof_handlers_and_fe();
    compute_dof_numbers();
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::initialize_dof_handlers_and_fe() {
    dof_h_q.distribute_dofs(fe_q);
    dof_h_nedelec.distribute_dofs(fe_nedelec);
    // std::cout << "Base Dof Counts Nedelec: " << fe_nedelec.dofs_per_cell << " per cell, " << fe_nedelec.dofs_per_line << " per line and " << fe_nedelec.dofs_per_vertex << "." << std::endl;
    // std::cout << "Base Dof Counts Q: " << fe_q.dofs_per_cell << " per cell, " << fe_q.dofs_per_line << " per line and " << fe_q.dofs_per_vertex << "." << std::endl;
}



template<unsigned int ORDER>
void HSIESurface<ORDER>::update_dof_counts_for_edge(const dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge, DofCount & in_dof_count) {
    bool edge_is_owned = is_edge_owned(cell, edge);
    const unsigned int dofs_per_edge_all = compute_dofs_per_edge(false);
    const unsigned int dofs_per_edge_hsie = compute_dofs_per_edge(true);
    in_dof_count.total +=  dofs_per_edge_all;
    in_dof_count.hsie += dofs_per_edge_hsie;
    in_dof_count.non_hsie += dofs_per_edge_all - dofs_per_edge_hsie;
    if(edge_is_owned) {
        in_dof_count.owned += dofs_per_edge_all;
        in_dof_count.owned_hsie += dofs_per_edge_hsie;
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::update_dof_counts_for_face(const dealii::DoFHandler<2>::active_cell_iterator cell, DofCount & in_dof_count) {
    bool edge_is_owned = is_face_owned(cell);
    const unsigned int dofs_per_face_all = compute_dofs_per_face(false);
    const unsigned int dofs_per_face_hsie = compute_dofs_per_face(true);
    in_dof_count.total +=  dofs_per_face_all;
    in_dof_count.hsie += dofs_per_face_hsie;
    in_dof_count.non_hsie += dofs_per_face_all - dofs_per_face_hsie;
    if(edge_is_owned) {
        in_dof_count.owned += dofs_per_face_all;
        in_dof_count.owned_hsie += dofs_per_face_hsie;
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::update_dof_counts_for_vertex(const dealii::DoFHandler<2>::active_cell_iterator cell, unsigned  int edge, unsigned int vertex, DofCount & in_dof_count) {
    bool edge_is_owned = is_vertex_owned(cell, edge, vertex);
    const unsigned int dofs_per_vertex_all = compute_dofs_per_vertex();

    in_dof_count.total +=  dofs_per_vertex_all;
    in_dof_count.hsie += dofs_per_vertex_all;
    if(edge_is_owned) {
        in_dof_count.owned += dofs_per_vertex_all;
        in_dof_count.owned_hsie += dofs_per_vertex_all;
    }
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::is_edge_owned(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge) {

    /**
     if(level == GlobalParams.HSIE_SWEEPING_LEVEL) {
        return true;
    } else {
        Triangulation<3>::face_iterator face3d = association.find(cell)->second;
        Point<3> location = face3d->line(edge)->center();
        if(location[0] == Geometry.x_range.first){
            return false;
        }
        if(location[1] == Geometry.y_range.first){
            return false;
        }
    }
     **/
    return true;
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::is_face_owned(dealii::DoFHandler<2>::active_cell_iterator) {
    return true;
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::is_vertex_owned(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int ,
                                         unsigned int vertex) {
    return true;

    // TODO: Fix this.
    /**
    if(level == GlobalParams.HSIE_SWEEPING_LEVEL) {
        return true;
    } else {
        Triangulation<3>::face_iterator face3d = association.find(cell)->second;
        Point<3> location = face3d->vertex(vertex);
        if(location[0] == Geometry.x_range.first){
            return false;
        }
        if(location[1] == Geometry.y_range.first){
            return false;
        }

    }
    return true;
     **/
}


template<unsigned int ORDER>
void HSIESurface<ORDER>::register_new_vertex_dofs(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int dof_index,
                                                  unsigned int vertex) {
    const int max_hsie_order = ORDER;
    for(int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order++ ) {
        register_single_dof(cell->vertex_index(vertex), hsie_order, -1, true, DofType::RAY, vertex_dof_data, dof_index);
        register_single_dof(cell->vertex_index(vertex), hsie_order, -1, false, DofType::RAY, vertex_dof_data, dof_index);
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_new_edge_dofs(dealii::DoFHandler<2>::active_cell_iterator cell_nedelec,dealii::DoFHandler<2>::active_cell_iterator cell_q, unsigned int edge) {
    const int max_hsie_order = ORDER;
    // EDGE Dofs
    std::vector<unsigned int> local_dofs(fe_nedelec.dofs_per_line);
    cell_nedelec->line(edge)->get_dof_indices(local_dofs);
    for(int inner_order = 0; inner_order < (int)fe_nedelec.dofs_per_line; inner_order++ ) {
        register_single_dof(cell_nedelec->face_index(edge), -1, inner_order+1, true, DofType::EDGE, edge_dof_data, local_dofs[inner_order]);
        register_single_dof(cell_nedelec->face_index(edge), -1, inner_order+1, false, DofType::EDGE, edge_dof_data, local_dofs[inner_order]);
    }

    // INFINITE FACE Dofs Type a
    for(int inner_order = 0; inner_order < (int)fe_nedelec.dofs_per_line; inner_order++ ) {
        for(int hsie_order = 0; hsie_order <= max_hsie_order; hsie_order ++) {
            register_single_dof(cell_nedelec->face_index(edge), hsie_order, inner_order+1, true, DofType::IFFa, edge_dof_data, local_dofs[inner_order]);
            register_single_dof(cell_nedelec->face_index(edge), hsie_order, inner_order+1, false, DofType::IFFa, edge_dof_data, local_dofs[inner_order]);
        }
    }

    // INFINITE FACE Dofs Type b
    local_dofs.clear();
    local_dofs.resize(fe_q.dofs_per_line + 2* fe_q.dofs_per_vertex);
    cell_q->line(edge)->get_dof_indices(local_dofs);
    IndexSet line_dofs(MAX_DOF_NUMBER), non_line_dofs(MAX_DOF_NUMBER);
    for(unsigned int i = 0; i < local_dofs.size(); i++) {
        line_dofs.add_index(local_dofs[i]);
    }
    for(unsigned int i = 0; i < fe_q.dofs_per_vertex; i++) {
        non_line_dofs.add_index(cell_q->line(edge)->vertex_dof_index(0,i));
        non_line_dofs.add_index(cell_q->line(edge)->vertex_dof_index(1,i));
    }
    line_dofs.subtract_set(non_line_dofs);
    for(int inner_order = 0; inner_order < (int)line_dofs.n_elements(); inner_order++ ) {
        for(int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order ++) {
            register_single_dof(cell_q->face_index(edge), hsie_order, inner_order, true, DofType::IFFb, edge_dof_data, line_dofs.nth_index_in_set(inner_order));
            register_single_dof(cell_q->face_index(edge), hsie_order, inner_order, false, DofType::IFFb, edge_dof_data, line_dofs.nth_index_in_set(inner_order));
        }
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_new_surface_dofs(dealii::DoFHandler<2>::active_cell_iterator cell_nedelec, dealii::DoFHandler<2>::active_cell_iterator cell_q) {
    unsigned int ret = 0;
    const unsigned int INNER_REAL_NEDELEC_DOFS_PER_FACE = fe_nedelec.dofs_per_cell - dealii::GeometryInfo<2>::faces_per_cell * fe_nedelec.dofs_per_face;

    ret = INNER_REAL_NEDELEC_DOFS_PER_FACE * (ORDER + 2) * 3;

    const int max_hsie_order = ORDER;
    std::vector<unsigned int> surface_dofs(fe_nedelec.dofs_per_cell);
    cell_nedelec->get_dof_indices(surface_dofs);
    IndexSet surf_dofs(MAX_DOF_NUMBER), edge_dofs(MAX_DOF_NUMBER);
    for(unsigned int i = 0; i < surface_dofs.size(); i++) {
        surf_dofs.add_index(surface_dofs[i]);
    }
    // std::cout << surf_dofs.n_elements() << " -> ";
    for(unsigned int i = 0; i < dealii::GeometryInfo<2>::lines_per_cell; i++) {
        std::vector<unsigned int> line_dofs(fe_nedelec.dofs_per_line);
        cell_nedelec->line(i)->get_dof_indices(line_dofs);
        for(unsigned int j = 0; j < line_dofs.size(); j++) {
            edge_dofs.add_index(line_dofs[j]);
        }
    }
    surf_dofs.subtract_set(edge_dofs);
    // std::cout << surf_dofs.n_elements() << std::endl;
    std::string id = cell_q->id().to_string();
    // SURFACE functions
    for(unsigned int inner_order = 0; inner_order < surf_dofs.n_elements(); inner_order++ ) {
        register_single_dof(cell_nedelec->id().to_string(), -1, inner_order, true, DofType::SURFACE, face_dof_data, surf_dofs.nth_index_in_set(inner_order));
        register_single_dof(cell_nedelec->id().to_string(), -1, inner_order, false, DofType::SURFACE, face_dof_data, surf_dofs.nth_index_in_set(inner_order));
    }

    // SEGMENT functions a
    for(unsigned int inner_order = 0; inner_order < surf_dofs.n_elements() ; inner_order++ ) {
        for(int hsie_order = 0; hsie_order <= max_hsie_order; hsie_order ++) {
            register_single_dof(id, hsie_order, inner_order, true, DofType::SEGMENTa, face_dof_data, surf_dofs.nth_index_in_set(inner_order));
            register_single_dof(id, hsie_order, inner_order, false, DofType::SEGMENTa, face_dof_data, surf_dofs.nth_index_in_set(inner_order));
        }
    }

    for(unsigned int inner_order = 0; inner_order < surf_dofs.n_elements()/2 ; inner_order++ ) {
        for(int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order ++) {
            register_single_dof(id, hsie_order, inner_order, true, DofType::SEGMENTb, face_dof_data, surf_dofs.nth_index_in_set(inner_order*2));
            register_single_dof(id, hsie_order, inner_order, false, DofType::SEGMENTb, face_dof_data, surf_dofs.nth_index_in_set(inner_order*2));
        }
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_single_dof(std::string in_id, const int in_hsie_order, const int in_inner_order, bool in_is_real, DofType in_dof_type, std::vector<DofData> & in_vector, unsigned int in_base_dof_index) {
    DofData dd(in_id);
    dd.global_index = register_dof();
    dd.hsie_order = in_hsie_order;
    dd.inner_order = in_inner_order;
    dd.is_real = in_is_real;
    dd.type = in_dof_type;
    dd.set_base_dof(in_base_dof_index);
    dd.update_nodal_basis_flag();
    in_vector.push_back(dd);
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_single_dof(unsigned int in_id, const int in_hsie_order, const int in_inner_order, const bool in_is_real, DofType in_dof_type, std::vector<DofData> & in_vector, unsigned int in_base_dof_index) {
    DofData dd(in_id);
    dd.global_index = register_dof();
    dd.hsie_order = in_hsie_order;
    dd.inner_order = in_inner_order;
    dd.is_real = in_is_real;
    dd.type = in_dof_type;
    dd.set_base_dof(in_base_dof_index);
    dd.update_nodal_basis_flag();
    in_vector.push_back(dd);
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::register_dof() {
    this->dof_counter++;
    return this->dof_counter - 1;
}

template<unsigned int ORDER>
std::complex<double> HSIESurface<ORDER>::evaluate_a(std::vector<HSIEPolynomial> &u, std::vector<HSIEPolynomial> &v, unsigned int steps) {
    std::complex<double> expected_result(0,0);
    for(unsigned j = 0; j < 3; j++) {
        for (unsigned int i = 0; i < 10; i++) {
            expected_result += u[j].a[i] * v[j].a[i];
        }
    }
    return expected_result;

    /**
    std::complex<double> ret(0,0);
    const double PI = 3.14159265358;
    double stepwidth = 2.0 * PI / (steps-1);
    double x,y;
    const double weight = 1.0 / steps;
    for(unsigned int i = 0; i < steps; i++) {
        x = sin(i * stepwidth);
        y = cos(i*stepwidth);
        ret += (  u[0].evaluate(std::complex<double>(x,y)) * v[0].evaluate(std::complex<double>(x,-y)) +
                  u[1].evaluate(std::complex<double>(x,y)) * v[1].evaluate(std::complex<double>(x,-y)) +
                  u[2].evaluate(std::complex<double>(x,y)) * v[2].evaluate(std::complex<double>(x,-y)) );
    }
    return ret*weight;
     **/
}

// NEW GENERATORS:

template<unsigned int ORDER>
std::vector<HSIEPolynomial>
HSIESurface<ORDER>::build_curl_term_nedelec(unsigned int dof_hsie_order, const Tensor<1, 2> fe_shape_gradient_component_0, const Tensor<1, 2> fe_shape_gradient_component_1, const double fe_shape_value_component_0, const double fe_shape_value_component_1) {
    std::vector<HSIEPolynomial> ret;
    HSIEPolynomial temp = HSIEPolynomial::PsiJ(dof_hsie_order,k0);
    temp.multiplyBy(fe_shape_gradient_component_0[1]);
    temp.applyI();
    HSIEPolynomial temp2 = HSIEPolynomial::PsiJ(dof_hsie_order,k0);
    temp2.multiplyBy(-1.0 * fe_shape_gradient_component_1[0]);
    temp2.applyI();
    temp.add(temp2);
    ret.push_back(temp);

    temp = HSIEPolynomial::PsiJ(dof_hsie_order,k0);
    temp.multiplyBy(-1.0 * fe_shape_value_component_1);
    temp.applyDerivative();
    ret.push_back(temp);

    temp = HSIEPolynomial::PsiJ(dof_hsie_order,k0);
    temp.multiplyBy(fe_shape_value_component_0);
    temp.applyDerivative();
    ret.push_back(temp);

    this->transform_coordinates_in_place(&ret);
    return ret;
}

template<unsigned int ORDER>
std::vector<HSIEPolynomial>
HSIESurface<ORDER>::build_non_curl_term_nedelec(const unsigned int dof_hsie_order, const double fe_shape_value_component_0, const double fe_shape_value_component_1) {
    std::vector<HSIEPolynomial> ret;
    ret.push_back(HSIEPolynomial::ZeroPolynomial());
    HSIEPolynomial temp = HSIEPolynomial::PsiJ(dof_hsie_order,k0);
    temp.multiplyBy(fe_shape_value_component_0);
    ret.push_back(temp);
    temp = HSIEPolynomial::PsiJ(dof_hsie_order,k0);
    temp.multiplyBy(fe_shape_value_component_1);
    ret.push_back(temp);
    this->transform_coordinates_in_place(&ret);
    return ret;
}



template<unsigned int ORDER>
std::vector<HSIEPolynomial>
HSIESurface<ORDER>::build_curl_term_q(const unsigned int dof_hsie_order, const Tensor<1,2> fe_gradient) {
    std::vector<HSIEPolynomial> ret;
    ret.push_back(HSIEPolynomial::ZeroPolynomial());
    HSIEPolynomial temp = HSIEPolynomial::PhiJ(dof_hsie_order, k0);
    temp.multiplyBy(fe_gradient[1]);
    ret.push_back(temp);
    temp = HSIEPolynomial::PhiJ(dof_hsie_order, k0);
    temp.multiplyBy(-1.0 * fe_gradient[0]);
    ret.push_back(temp);
    this->transform_coordinates_in_place(&ret);
    return ret;
}

template<unsigned int ORDER>
std::vector<HSIEPolynomial>
HSIESurface<ORDER>::build_non_curl_term_q(const unsigned int dof_hsie_order, const double fe_shape_value) {
    std::vector<HSIEPolynomial> ret;
    HSIEPolynomial temp = HSIEPolynomial::PhiJ(dof_hsie_order, k0);
    temp.multiplyBy(fe_shape_value);
    temp = temp.applyD();
    ret.push_back(temp);
    ret.push_back(HSIEPolynomial::ZeroPolynomial());
    ret.push_back(HSIEPolynomial::ZeroPolynomial());
    this->transform_coordinates_in_place(&ret);
    return ret;
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::transform_coordinates_in_place(std::vector<HSIEPolynomial> * vector) {
    // The ray direction before transformation is x. This has to be adapted.
    HSIEPolynomial temp = (*vector)[0];
    switch (this->b_id){
        case 2:
            (*vector)[0] = (*vector)[1];
            (*vector)[1] = temp;
            break;
        case 3:
            (*vector)[0] = (*vector)[1];
            (*vector)[1] = temp;
            break;
        case 4:
            (*vector)[0] = (*vector)[2];
            (*vector)[2] = temp;
            break;
        case 5:
            (*vector)[0] = (*vector)[2];
            (*vector)[2] = temp;
            break;
    }
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::check_dof_assignment_integrity() {
    HSIEPolynomial::computeDandI(ORDER + 2, k0);
    auto it = dof_h_nedelec.begin_active();
    auto end = dof_h_nedelec.end();
    // for each cell

    const unsigned int dofs_per_cell = GeometryInfo<2>::vertices_per_cell * compute_dofs_per_vertex() + GeometryInfo<2>::lines_per_cell * compute_dofs_per_edge(false) + compute_dofs_per_face(false);

    auto it2 = dof_h_q.begin_active();
    unsigned int counter = 1;
    for(; it != end; ++it) {
        if(it->id() != it2->id()) std::cout << "Identity failure!" <<std::endl;
        std::vector<DofData> cell_dofs = this->get_dof_data_for_cell(it, it2);
        std::vector<unsigned int> q_dofs(fe_q.dofs_per_cell);
        std::vector<unsigned int> n_dofs(fe_nedelec.dofs_per_cell);
        it2->get_dof_indices(q_dofs);
        it->get_dof_indices(n_dofs);
        std::vector<unsigned int> local_related_fe_index;
        bool found = false;
        for(unsigned int i = 0; i < cell_dofs.size(); i++) {
            found = false;
            if(cell_dofs[i].type == DofType::RAY || cell_dofs[i].type == DofType::IFFb) {
                for(unsigned int j= 0; j < q_dofs.size(); j++) {
                    if(q_dofs[j] == cell_dofs[i].base_dof_index) {
                        local_related_fe_index.push_back(j);
                        found = true;
                    }
                }
            } else {
                for(unsigned int j= 0; j < n_dofs.size(); j++) {
                    if(n_dofs[j] == cell_dofs[i].base_dof_index) {
                        local_related_fe_index.push_back(j);
                        found = true;
                    }
                }
            }
            if(!found) {
                std::cout << cell_dofs[i].base_dof_index << " not found. Type: << " << cell_dofs[i].type << " Available: ";
                for(unsigned int j= 0; j < q_dofs.size(); j++) {
                    std::cout << q_dofs[j] << ", ";
                }
                std::cout << " AND ";
                for(unsigned int j= 0; j < n_dofs.size(); j++) {
                    std::cout << n_dofs[j] << ", ";
                }
                std::cout << std::endl;
            }
        }

        if(!(local_related_fe_index.size() == cell_dofs.size())) {
            std::cout << "Mismatch in cell " << counter << ": Found indices: " << local_related_fe_index.size() << " of a total " << cell_dofs.size() <<std::endl;
            return false;
        }
        counter ++;
        it2++;
    }

    return true;
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::check_number_of_dofs_for_cell_integrity() {
    auto it = dof_h_nedelec.begin_active();
    auto it2 = dof_h_q.begin_active();
    auto end = dof_h_nedelec.end();
    const unsigned int dofs_per_cell = 4 * compute_dofs_per_vertex() + 4 * compute_dofs_per_edge(false) + compute_dofs_per_face(false);
    unsigned int counter = 0;
    for(; it != end; ++it) {
        std::vector<DofData> cell_dofs = this->get_dof_data_for_cell(it, it2);
        if(cell_dofs.size() != dofs_per_cell) {
            std::cout << cell_dofs.size() << " is not " << dofs_per_cell << " in cell " << counter << std::endl;
            for(unsigned int i = 0; i < 7; i++) {
                unsigned int count = 0;
                for (int j = 0; j < cell_dofs.size(); ++j) {
                    if(cell_dofs[j].type == i) count++;
                }
                std::cout << "For type " << i << " I found " << count << " dofs"<< std::endl;
            }
            return false;
        }
        counter ++;
        it2++;
    }
    return true;
}

template <unsigned int ORDER>
void HSIESurface<ORDER>::fill_sparsity_pattern(dealii::DynamicSparsityPattern * pattern) {
    auto it = dof_h_nedelec.begin_active();
    auto it2 = dof_h_q.begin_active();
    auto end = dof_h_nedelec.end();
    for(; it != end; ++it) {
        std::vector<DofData> cell_dofs = this->get_dof_data_for_cell(it, it2);
        for(unsigned int i=0; i < cell_dofs.size(); i++) {
            for(unsigned int j=0; j < cell_dofs.size(); j++) {
                pattern->add(cell_dofs[i].global_index, cell_dofs[j].global_index);
            }
        }
        it2++;
    }
}

