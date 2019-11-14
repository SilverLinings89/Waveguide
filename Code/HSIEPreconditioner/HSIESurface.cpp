//
// Created by kraft on 16.08.19.
//

#include <deal.II/grid/grid_generator.h>
#include "HSIESurface.h"
#include "../Core/NumericProblem.h"
#include "HSIEPolynomial.h"
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_accessor.h>

#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "DofData.h"
#include <utility>


template<unsigned int ORDER>
void HSIESurface<ORDER>::prepare_surface_triangulation() {
    std::set<unsigned int> b_ids;
    b_ids.insert(this->b_id);
    association = dealii::GridGenerator::extract_boundary_mesh(*main_triangulation, surface_triangulation, b_ids);
}

template<unsigned int ORDER>
HSIESurface<ORDER>::HSIESurface(dealii::Triangulation<3, 3> *in_main_triangulation, unsigned int in_boundary_id,
                                unsigned int in_level, unsigned int in_inner_order, std::complex<double> in_k0):
                                main_triangulation(in_main_triangulation),
                                b_id(in_boundary_id),
                                level(in_level),
                                Inner_Element_Order(in_inner_order),
                                fe_nedelec(dealii::FE_Nedelec<2>(Inner_Element_Order), 2),
                                fe_q(dealii::FE_Q<2>(GlobalParams.So_ElementOrder), 2)
                                {
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
std::vector<DofData> HSIESurface<ORDER>::get_dof_data_for_cell(dealii::Triangulation<2,3>::cell_iterator * cell) {
    std::vector<DofData> ret;

    // get cell dofs:
    std::string cell_id = (*cell)->id().to_string();
    unsigned int * edge_ids = new unsigned int[4];
    unsigned int * vertex_ids = new unsigned int[4];
    // get edge dofs:
    for(unsigned int i = 0; i < 4; i++) {
        edge_ids[i] = (*cell)->line_index(i);
    }

    // get vertex dofs:
    for(unsigned int i = 0; i < 4; i++) {
        vertex_ids[i]= (*cell)->vertex_index(i);
    }

    // add cell dofs
    for(unsigned int i = 0; i < this->face_dof_data.size(); i++) {
        if(this->face_dof_data[i].base_structure_id.face_id == cell_id) {
            ret.push_back(this->face_dof_data[i]);
        }
    }

    // add edge-based dofs
    for(unsigned int i = 0; i < this->edge_dof_data.size(); i++) {
        unsigned int idx = this->edge_dof_data[i].base_structure_id.non_face_id;
        if(idx == edge_ids[0] || idx == edge_ids[1] || idx == edge_ids[2] || idx == edge_ids[3]) {
            ret.push_back(this->edge_dof_data[i]);
        }
    }

    // add vertex-based dofs
    for(unsigned int i = 0; i < this->vertex_dof_data.size(); i++) {
        unsigned int idx = this->vertex_dof_data[i].base_structure_id.non_face_id;
        if(idx == vertex_ids[0] || idx == vertex_ids[1] || idx == vertex_ids[2] || idx == vertex_ids[3]) {
            ret.push_back(this->vertex_dof_data[i]);
        }
    }

    return ret;
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::fill_matrix(dealii::SparseMatrix<double> * matrix, dealii::IndexSet global_indices) {
    auto it = surface_triangulation.begin_active();
    auto end = surface_triangulation.end();
    // for each cell
    for(; it != end; ++it) {
        std::vector<DofData> cell_dofs = this->get_dofs_for_cell(it);
        // for each dof i
        for(unsigned int i = 0; i < cell_dofs.size(); i++) {
            // get dof i type data (type and degree, base point etc.)
            DofData& u = cell_dofs[i] ;
            // for each dof j
            for (unsigned int j = 0; j < cell_dofs.size(); j++) {
                // get dof j type data (type and degree, base point etc.)
                DofData& v = cell_dofs[j];
                // compute their coupling and write it to matrix
                matrix[global_indices.nth_index_in_set(i), global_indices.nth_index_in_set(j)] = this->compute_coupling(u,v,it);
            }
        }
    }
}

template <unsigned int ORDER>
double HSIESurface<ORDER>::compute_coupling(DofData & u, DofData & v, dealii::Triangulation<2,3>::cell_iterator * cell) {
    std::complex<double> ret(0,0);
    HSIEPolynomial up(u, k0);
    HSIEPolynomial vp(v, k0);

    return ret.real();
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_edge_dofs() {
    std::set<unsigned int> touched_edges;
    DoFHandler<2>::active_cell_iterator cell, endc;
    endc = dof_h_nedelec.end();
    DofCount ret;
    // for each cell
    for(cell = dof_h_nedelec.begin(); cell != endc; cell++) {
        // for each edge
        for(unsigned int edge = 0; edge < GeometryInfo<2>::lines_per_cell; edge++) {
            // if it wasn't handled before
            if(!(touched_edges.end() == touched_edges.find(cell->line_index(edge)))) {
                // handle it
                update_dof_counts_for_edge(cell, edge, &ret);
                register_new_edge_dofs(cell, edge);
                // remember that it has been handled
                touched_edges.insert(cell->line_index(edge));
            }
        }
    }
    return ret;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_vertex_dofs() {
    std::set<unsigned int> touched_vertices;
    DoFHandler<2>::active_cell_iterator cell, endc;
    endc = dof_h_nedelec.end();
    DofCount ret;
    // for each cell
    for(cell = dof_h_nedelec.begin(); cell != endc; cell++) {
        // for each edge
        for(unsigned int edge = 0; edge < GeometryInfo<2>::lines_per_cell; edge++) {
            // First Vertex
            if(!(touched_vertices.end() == touched_vertices.find(cell->line(edge)->vertex_index(0)))) {
                // handle it
                update_dof_counts_for_vertex(cell, edge, 0, &ret);
                register_new_vertex_dofs(cell, edge, 0);
                // remember that it has been handled
                touched_vertices.insert(cell->line_index(edge));
            }

            // if it wasn't handled before
            if(!(touched_vertices.end() == touched_vertices.find(cell->line(edge)->vertex_index(1)))) {
                // handle it
                update_dof_counts_for_vertex(cell, edge, 1, &ret);
                register_new_vertex_dofs(cell, edge, 1);
                // remember that it has been handled
                touched_vertices.insert(cell->line_index(edge));
            }
        }
    }
    return ret;
}

template<unsigned int ORDER>
DofCount HSIESurface<ORDER>::compute_n_face_dofs() {
    std::set<std::string> touched_faces;
    DoFHandler<2>::active_cell_iterator cell, endc;
    endc = dof_h_nedelec.end();
    DofCount ret;
    // for each cell
    for(cell = dof_h_nedelec.begin(); cell != endc; cell++) {
        // if it wasn't handled before
        if(!(touched_faces.end() == touched_faces.find(cell->id().to_string()))) {
            // handle it
            update_dof_counts_for_face(cell, &ret);
            register_new_surface_dofs(cell);
            // remember that it has been handled
            touched_faces.insert(cell->id().to_string());
        }
    }
    return ret;
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::compute_dofs_per_edge(bool only_hsie_dofs) {
    unsigned int ret = 0;
    const unsigned int INNER_REAL_DOFS_PER_LINE = fe_nedelec.dofs_per_line / 2;

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
    const unsigned int INNER_REAL_NEDELEC_DOFS_PER_FACE = fe_nedelec.dofs_per_face / 2;
    const unsigned int INNER_REAL_Q_DOFS_PER_FACE = fe_q.dofs_per_face / 2;
    if(! only_hsie_dofs) {
        ret += INNER_REAL_NEDELEC_DOFS_PER_FACE;
    }

    // Number of elements of type 5a.
    ret += INNER_REAL_NEDELEC_DOFS_PER_FACE * (ORDER + 1);

    // Number of elements of type 5b.
    ret += INNER_REAL_Q_DOFS_PER_FACE* (ORDER + 2);

    return ret;
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
    prepare_surface_triangulation();
    initialize_dof_handlers_and_fe();
    compute_dof_numbers();
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::initialize_dof_handlers_and_fe() {
    dof_h_nedelec = {surface_triangulation};
    dof_h_q = {surface_triangulation};
    dof_h_q.distribute_dofs(fe_q);
    dof_h_nedelec.distribute_dofs(fe_nedelec);
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

    // These are only hsie dofs.
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
    return true;
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::is_face_owned(dealii::DoFHandler<2>::active_cell_iterator cell) {
    return true;
}

template<unsigned int ORDER>
bool HSIESurface<ORDER>::is_vertex_owned(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge,
                                         unsigned int vertex) {
    if(level == GlobalParams.HSIE_SWEEPING_LEVEL) {
        return true;
    } else {
        Triangulation<3>::face_iterator face3d = association.find(cell)->second;
        Point<3> location = face3d->line(edge)->vertex(vertex);
        if(location[0] == Geometry.x_range.first){
            return false;
        }
        if(location[1] == Geometry.y_range.first){
            return false;
        }

    }
    return true;
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_new_vertex_dofs(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge,
                                                  unsigned int vertex) {
    const int max_hsie_order = ORDER;
    for(int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order++ ) {
        register_single_dof(cell->line(edge)->vertex_index(vertex), hsie_order, -1, true, DofType::RAY, &vertex_dof_data);
        register_single_dof(cell->line(edge)->vertex_index(vertex), hsie_order, -1, false, DofType::RAY, &vertex_dof_data);
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_new_edge_dofs(dealii::DoFHandler<2>::active_cell_iterator cell, unsigned int edge) {
    const int max_hsie_order = ORDER;
    // EDGE Dofs
    for(int inner_order = 1; inner_order <= fe_nedelec.dofs_per_line / 2; inner_order++ ) {
        register_single_dof(cell->line_index(edge), -2, inner_order, true, DofType::EDGE, &edge_dof_data);
        register_single_dof(cell->line_index(edge), -2, inner_order, false, DofType::EDGE, &edge_dof_data);
    }

    // INFINITE FACE Dofs Type a
    for(int inner_order = 1; inner_order <= fe_nedelec.dofs_per_line / 2; inner_order++ ) {
        for(int hsie_order = 0; hsie_order <= max_hsie_order; hsie_order ++) {
            register_single_dof(cell->line_index(edge), hsie_order, inner_order, true, DofType::IFFa, &edge_dof_data);
            register_single_dof(cell->line_index(edge), hsie_order, inner_order, false, DofType::IFFa, &edge_dof_data);
        }
    }

    // INFINITE FACE Dofs Type b
    for(int inner_order = 1; inner_order <= ((int)fe_nedelec.dofs_per_line / 2) - 1; inner_order++ ) {
        for(int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order ++) {
            register_single_dof(cell->line_index(edge), hsie_order, inner_order, true, DofType::IFFb, &edge_dof_data);
            register_single_dof(cell->line_index(edge), hsie_order, inner_order, false, DofType::IFFb, &edge_dof_data);
        }
    }
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_new_surface_dofs(dealii::DoFHandler<2>::active_cell_iterator cell) {
    const int max_hsie_order = ORDER;
    // SURFACE functions
    for(int inner_order = 1; inner_order <= ((int)fe_nedelec.dofs_per_line / 2) * ((int)fe_nedelec.dofs_per_line / 2 - 2) ; inner_order++ ) {
        register_single_dof(cell->id().to_string(), -2, inner_order, true, DofType::SURFACE, &face_dof_data);
        register_single_dof(cell->id().to_string(), -2, inner_order, false, DofType::SURFACE, &face_dof_data);
    }

    // SEGMENT functions a
    for(int inner_order = 1; inner_order <= ((int)fe_nedelec.dofs_per_line / 2) * ((int)fe_nedelec.dofs_per_line / 2 - 2) ; inner_order++ ) {
        for(int hsie_order = 0; hsie_order <= max_hsie_order; hsie_order ++) {
            register_single_dof(cell->id().to_string(), hsie_order, inner_order, true, DofType::SEGMENTa, &face_dof_data);
            register_single_dof(cell->id().to_string(), hsie_order, inner_order, false, DofType::SEGMENTa, &face_dof_data);
        }
    }

    // SEGMENT functions b
    const int INNER_ELEMENT_ORDER = fe_nedelec.dofs_per_line / 2;
    for(int inner_order = 1; inner_order <= (INNER_ELEMENT_ORDER-1)*(INNER_ELEMENT_ORDER-2)/2 ; inner_order++ ) {
        for(int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order ++) {
            register_single_dof(cell->id().to_string(), hsie_order, inner_order, true, DofType::SEGMENTb, &face_dof_data);
            register_single_dof(cell->id().to_string(), hsie_order, inner_order, false, DofType::SEGMENTb, &face_dof_data);
        }
    }

}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_single_dof(std::string & in_id, const int in_hsie_order, const int in_inner_order, bool in_is_real, DofType in_dof_type, std::vector<DofData> & in_vector) {
    DofData dd(in_id);
    dd.global_index = register_dof();
    dd.hsie_order = in_hsie_order;
    dd.inner_order = in_inner_order;
    dd.is_real = in_is_real;
    dd.type = in_dof_type;
    in_vector.push_back(dd);
}

template<unsigned int ORDER>
void HSIESurface<ORDER>::register_single_dof(unsigned int in_id, const int in_hsie_order, const int in_inner_order, const bool in_is_real, DofType in_dof_type, std::vector<DofData> & in_vector) {
    DofData dd(in_id);
    dd.global_index = register_dof();
    dd.hsie_order = in_hsie_order;
    dd.inner_order = in_inner_order;
    dd.is_real = in_is_real;
    dd.type = in_dof_type;
    in_vector.push_back(dd);
}

template<unsigned int ORDER>
unsigned int HSIESurface<ORDER>::register_dof() {
    this->dof_counter++;
    return this->dof_counter - 1;
}

template<unsigned int ORDER>
std::complex<double> HSIESurface<ORDER>::evaluate_a(std::vector<HSIEPolynomial> &u, std::vector<HSIEPolynomial> &v, unsigned int gauss_order) {
    double *r = NULL;
    double *t = NULL;
    double *q = NULL;
    double *A = NULL;
    double B;
    double x, y;
    std::complex<double> s(0.0, 0.0);

    int i, j;

    /* Load appropriate predefined table */
    for (i = 0; i < GSPHERESIZE; i++) {
        if (gauss_order == gsphere[i].n) {
            r = gsphere[i].r;
            t = gsphere[i].t;
            q = gsphere[i].q;
            A = gsphere[i].A;
            B = gsphere[i].B;
            break;
        }
    }

    if (NULL == r) return -1.0;

    for (i = 0; i < gauss_order; i++) {
        for (j = 0; j < gauss_order; j++) {
            x = r[j] * q[i];
            y = r[j] * t[i];
            s += (  u[0].evaluate(std::complex<double>(x,y)) * v[0].evaluate(std::complex<double>(x,-y)) +
                    u[1].evaluate(std::complex<double>(x,y)) * v[1].evaluate(std::complex<double>(x,-y)) +
                    u[2].evaluate(std::complex<double>(x,y)) * v[2].evaluate(std::complex<double>(x,-y)) ) * A[j];
        }
    }

    s *= B;

    return s;
}
