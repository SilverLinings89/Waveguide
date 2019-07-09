#ifndef HSIEPRECONDITIONERBASE_CPP
#define HSIEPRECONDITIONERBASE_CPP

#include "HSIE_Preconditioner_Base.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <complex>
#include <vector>

#include "../Core/NumericProblem.h"
#include "FaceSurfaceComparator.h"
#include "FaceSurfaceComparatorZ.h"

/*
 * HSIE_Preconditioner_Base.cpp
 *
 *  Created on: Jul 11, 2018
 *      Author: kraft
 */

const unsigned int S1IntegrationPoints = 100;
const double k0 = 0.5;

std::map<
        dealii::Triangulation<2, 3>::cell_iterator,
        typename dealii::Triangulation<3, 3>::face_iterator>
extract_surface_mesh_at_z(
        const dealii::Triangulation<3, 3> *volume_mesh,
        dealii::Triangulation<2, 3> *surface_mesh, FaceSurfaceComparator &comp) {
    const int dim = 3;
    const int spacedim = 3;

    // This function works using the following assumption:
    //    Triangulation::create_triangulation(...) will create cells that preserve
    //    the order of cells passed in using the CellData argument; also,
    //    that it will not reorder the vertices.

    std::map<typename dealii::Triangulation<2, 3>::cell_iterator,
            typename dealii::Triangulation<3, 3>::face_iterator>
            surface_to_volume_mapping;

    const unsigned int boundary_dim = 2;  // dimension of the boundary mesh

    // First create surface mesh and mapping
    // from only level(0) cells of volume_mesh
    std::vector<typename dealii::Triangulation<3, 3>::face_iterator>
            mapping;  // temporary map for level==0

    std::vector<bool> touched(volume_mesh->get_triangulation().n_vertices(),
                              false);
    std::vector<dealii::CellData<boundary_dim>> cells;
    dealii::SubCellData subcell_data;
    std::vector<dealii::Point<3>> vertices;

    std::map<unsigned int, unsigned int>
            map_vert_index;  // volume vertex indices to surf ones

    for (typename dealii::Triangulation<
            3, 3>::cell_iterator cell = volume_mesh->begin(0);
         cell != volume_mesh->end(0); ++cell)
        for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::faces_per_cell;
             ++i) {
            const typename dealii::Triangulation<3, 3>::face_iterator face =
                    cell->face(i);

            if (comp.check_face(face)) {
                dealii::CellData<boundary_dim> c_data;

                for (unsigned int j = 0;
                     j < dealii::GeometryInfo<boundary_dim>::vertices_per_cell; ++j) {
                    const unsigned int v_index = face->vertex_index(j);

                    if (!touched[v_index]) {
                        vertices.push_back(face->vertex(j));
                        map_vert_index[v_index] = vertices.size() - 1;
                        touched[v_index] = true;
                    }

                    c_data.vertices[j] = map_vert_index[v_index];
                    c_data.material_id =
                            static_cast<dealii::types::material_id>(face->boundary_id());
                }

                // if we start from a 3d mesh, then we have copied the
                // vertex information in the same order in which they
                // appear in the face; however, this means that we
                // impart a coordinate system that is right-handed when
                // looked at *from the outside* of the cell if the
                // current face has index 0, 2, 4 within a 3d cell, but
                // right-handed when looked at *from the inside* for the
                // other faces. we fix this by flipping opposite
                // vertices if we are on a face 1, 3, 5
                if (dim == 3)
                    if (i % 2 == 1) std::swap(c_data.vertices[1], c_data.vertices[2]);

                // in 3d, we also need to make sure we copy the manifold
                // indicators from the edges of the volume mesh to the
                // edges of the surface mesh
                //
                // one might think that we we can also prescribe
                // boundary indicators for edges, but this is only
                // possible for edges that aren't just on the boundary
                // of the domain (all of the edges we consider are!) but
                // that would actually end up at the boundary of the
                // surface mesh. there is no easy way to check this, so
                // we simply don't do it and instead set it to an
                // invalid value that makes sure
                // Triangulation::create_triangulation doesn't copy it
                if (dim == 3)
                    for (unsigned int e = 0; e < 4; ++e) {
                        // see if we already saw this edge from a
                        // neighboring face, either in this or the reverse
                        // orientation. if so, skip it.
                        {
                            bool edge_found = false;
                            for (unsigned int i = 0; i < subcell_data.boundary_lines.size();
                                 ++i)
                                if (((subcell_data.boundary_lines[i].vertices[0] ==
                                      map_vert_index[face->line(e)->vertex_index(0)]) &&
                                     (subcell_data.boundary_lines[i].vertices[1] ==
                                      map_vert_index[face->line(e)->vertex_index(1)])) ||
                                    ((subcell_data.boundary_lines[i].vertices[0] ==
                                      map_vert_index[face->line(e)->vertex_index(1)]) &&
                                     (subcell_data.boundary_lines[i].vertices[1] ==
                                      map_vert_index[face->line(e)->vertex_index(0)]))) {
                                    edge_found = true;
                                    break;
                                }
                            if (edge_found == true)
                                continue;  // try next edge of current face
                        }

                        dealii::CellData<1> edge;
                        edge.vertices[0] = map_vert_index[face->line(e)->vertex_index(0)];
                        edge.vertices[1] = map_vert_index[face->line(e)->vertex_index(1)];
                        edge.boundary_id = dealii::numbers::internal_face_boundary_id;
                        edge.manifold_id = face->line(e)->manifold_id();

                        subcell_data.boundary_lines.push_back(edge);
                    }

                cells.push_back(c_data);
                mapping.push_back(face);
            }
        }

    // create level 0 surface triangulation
    Assert(cells.size() > 0, ExcMessage("No boundary faces selected"));
    const_cast<dealii::Triangulation<dim - 1, spacedim> &>(
            surface_mesh->get_triangulation())
            .create_triangulation(vertices, cells, subcell_data);

    // Make the actual mapping
    for (typename dealii::Triangulation<2, 3>::active_cell_iterator cell =
            surface_mesh->begin(0);
         cell != surface_mesh->end(0); ++cell)
        surface_to_volume_mapping[cell] = mapping.at(cell->index());

    do {
        bool changed = false;

        for (typename dealii::Triangulation<2, 3>::active_cell_iterator cell =
                surface_mesh->begin_active();
             cell != surface_mesh->end(); ++cell)
            if (surface_to_volume_mapping[cell]->has_children() == true) {
                cell->set_refine_flag();
                changed = true;
            }

        if (changed) {
            const_cast<dealii::Triangulation<dim - 1, spacedim> &>(
                    surface_mesh->get_triangulation())
                    .execute_coarsening_and_refinement();

            for (typename dealii::Triangulation<2, 3>::cell_iterator surface_cell =
                    surface_mesh->begin();
                 surface_cell != surface_mesh->end(); ++surface_cell)
                for (unsigned int c = 0; c < surface_cell->n_children(); c++)
                    if (surface_to_volume_mapping.find(surface_cell->child(c)) ==
                        surface_to_volume_mapping.end())
                        surface_to_volume_mapping[surface_cell->child(c)] =
                                surface_to_volume_mapping[surface_cell]->child(c);
        } else
            break;
    } while (true);

    return surface_to_volume_mapping;
}

template<int hsie_order>
HSIEPreconditionerBase<hsie_order>::~HSIEPreconditionerBase() {
    delete system_matrix;
}

template<int hsie_order>
HSIEPreconditionerBase<hsie_order>::HSIEPreconditionerBase(
        const dealii::Triangulation<3, 3> *in_tria,
        double in_z) {
    FaceSurfaceComparatorZ fscz = new FaceSurfaceComparatorZ(in_z, 0.00001);
    association = extract_surface_mesh_at_z(in_tria, &surf_tria, fscz);
    surface_edges = surf_tria.n_active_lines();
    surface_faces = surf_tria.n_active_faces();
    surface_vertices = surf_tria.n_vertices();
    HSIE_dofs_type_1_factor = dealii::GeometryInfo<2>::lines_per_face * 2;
    HSIE_dofs_type_3_factor =
            dealii::GeometryInfo<2>::lines_per_face * 2 * (hsie_order + 1);
    HSIE_dofs_type_2_factor =
            dealii::GeometryInfo<2>::vertices_per_face * 2 * (hsie_order + 1);
    HSIE_degree = hsie_order;
    hsie_dof_handler = dealii::DoFHandler<3>(surf_tria);
}

template<int hsie_order>
void HSIEPreconditionerBase<hsie_order>::setup_system() {
    const unsigned int n_total_dofs = n_dofs();
    // IndexSet dofs(n_dofs);
    // dofs.add_range(0, n_dofs - 1);
    const int max_couplings = 4 * n_dofs_per_face();
    system_matrix = new dealii::TrilinosWrappers::SparseMatrix(n_total_dofs, n_total_dofs, max_couplings);
}

dealii::Point<3, std::complex<double>> base_fun(int i, int j,
                                                dealii::Point<2, double> x) {
    dealii::Point<3, std::complex<double>> ret;
    ret[0] = std::complex<double>(0.0, 0.0);
    ret[1] = std::complex<double>(0.0, 0.0);
    ret[2] = std::complex<double>(0.0, 0.0);
    return ret;
}

std::complex<double> a(int in_i, int in_j) {
    std::complex<double> ret(0, 0);
    for (unsigned int i = 0; i < S1IntegrationPoints; i++) {
        double x = std::sin(2 * 3.14159265359 * ((double) i) /
                            ((double) S1IntegrationPoints));
        double y = std::cos(2 * 3.14159265359 * ((double) i) /
                            ((double) S1IntegrationPoints));
        dealii::Point<2, double> p1(x, y);
        dealii::Point<2, double> p2(x, -y);
        ret += base_fun(in_i, 0, p1) * base_fun(in_j, 0, p2);
    }
    ret /= (double) S1IntegrationPoints;
    ret *= k0 * std::complex<double>(0.0, 1.0) / 3.14159265359;
    return ret;
}

int count_HSIE_dofs(dealii::Triangulation<3> *tria,
                    unsigned int degree, double in_z) {
    dealii::IndexSet interfacevertices(tria->n_vertices());
    dealii::IndexSet interfaceedges(tria->n_active_lines());
    dealii::Triangulation<3>::active_cell_iterator
            cell = tria->begin_active(),
            endc = tria->end();
    for (; cell != endc; ++cell) {
        if (cell->is_locally_owned()) {
            for (unsigned int i = 0; i < dealii::GeometryInfo<3>::faces_per_cell;
                 i++) {
                dealii::Point<3, double> pos = cell->face(i)->center(true, true);
                if (abs(pos[2] - in_z) < 0.0001) {
                    for (unsigned int j = 0; j < dealii::GeometryInfo<3>::lines_per_face;
                         j++) {
                        interfacevertices.add_index(
                                cell->face(i)->line(j)->vertex_index(0));
                        interfacevertices.add_index(
                                cell->face(i)->line(j)->vertex_index(1));
                        interfaceedges.add_index(cell->face(i)->line(j)->index());
                    }
                }
            }
        }
    }

    int n_if_edges = interfaceedges.n_elements();
    int n_if_vertices = interfacevertices.n_elements();

    return n_if_vertices * (degree + 2) + n_if_edges * 1 * (degree + 1);
}

template<int hsie_order>
unsigned int HSIEPreconditionerBase<hsie_order>::n_dofs() {
    unsigned int ret = 0;
    ret += surface_edges * HSIE_dofs_type_1_factor;
    ret += surface_vertices * HSIE_dofs_type_2_factor;
    ret += surface_edges * HSIE_dofs_type_3_factor;
    return ret;
}

template<int hsie_order>
unsigned int HSIEPreconditionerBase<hsie_order>::n_dofs_per_face() {
    unsigned int ret = 0;
    ret += 4 * HSIE_dofs_type_1_factor;
    ret += 4 * HSIE_dofs_type_2_factor;
    ret += 4 * HSIE_dofs_type_3_factor;
    return ret;
}

template<int hsie_order>
std::complex<double> HSIEPreconditionerBase<hsie_order>::a(
        HSIE_Dof_Type<hsie_order> u, HSIE_Dof_Type<hsie_order> v,
        bool use_curl_formulation, dealii::Point<2, double> x) {}

template<int hsie_order>
std::complex<double> HSIEPreconditionerBase<hsie_order>::A(
        HSIE_Dof_Type<hsie_order> u, HSIE_Dof_Type<hsie_order> v,
        dealii::Tensor<2, 3, double> G, bool use_curl_fomulation) {
    dealii::QGauss<2> quadrature_formula(2);
    std::complex<double> ret(0, 0);
    std::vector<dealii::Point<2>> quad_points =
            quadrature_formula.quadrature_points;
    for (unsigned int i = 0; i < quad_points.size(); i++) {
        for (unsigned int j = 0; j < 3; j++) {
            for (unsigned int k = 0; k < 3; k++) {
                ret += quadrature_formula.weight(i) * G[j, k] *
                       a(u, v, true, quad_points[i]);
            }
        }
    }
    return ret;
}

template<int hsie_order>
void HSIEPreconditionerBase<hsie_order>::assemble_block() {
    QGauss<2> quadrature_formula(2);
    dealii::FE_Nedelec<2> fe_nedelec;
    dealii::FE_Q<2> fe_q;
    FEValues<2, 3> ned_fe_values(fe_nedelec, quadrature_formula,
                                 update_values | update_gradients |
                                 update_JxW_values |
                                 update_quadrature_points);
    FEValues<2, 3> q_fe_values(fe_q, quadrature_formula,
                               update_values | update_gradients |
                               update_JxW_values | update_quadrature_points);
    std::vector<Point<3>> quadrature_points;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int dofs_per_cell = this->n_dofs_per_face();
    deallog << "Starting assembly process" << std::endl;

    FullMatrix<double> cell_matrix_real(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    DoFHandler<2, 3>::active_cell_iterator cell, endc;
    cell = hsie_dof_handler.begin_active(), endc = hsie_dof_handler.end();
    const unsigned int n_nodal_functions = 4;
    const unsigned int n_nedelec_functions = 4;
    FullMatrix<double> shape_funs(n_nodal_functions + n_nedelec_functions, 3);
    FullMatrix<double> gradients(n_nodal_functions + n_nedelec_functions, 3 + 3);
    HSIE_Dof_Type<hsie_order> *hsie_dofs =
            new HSIE_Dof_Type<hsie_order>[dofs_per_cell];
    for (unsigned int i = 0; i < dofs_per_cell; i++) {
        hsie_dofs[i] = new HSIE_Dof_Type<hsie_order>(
                0, 0, 0, HSIE_Infinite_Direction::z, &shape_funs, &gradients);
    }
    for (; cell != endc; ++cell) {
        ned_fe_values.reinit(cell);
        quadrature_points = ned_fe_values.get_quadrature_points();
        q_fe_values.reinit(cell);
        cell_matrix_real = 0;
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            Point<2, double> current_point_2D;
            current_point_2D[0] = quadrature_points[q_index][0];
            current_point_2D[1] = quadrature_points[q_index][1];

            for (unsigned int i = 0; i < n_nodal_functions; i++) {
                for (unsigned int j = 0; j < 3; j++) {
                    shape_funs[i, j] = fe_q.shape_value_component(i, current_point_2D, j);
                }
            }
            for (unsigned int i = 0; i < n_nedelec_functions; i++) {
                for (unsigned int j = 0; j < 3; j++) {
                    shape_funs[i + n_nodal_functions, j] =
                            fe_nedelec.shape_value_component(i, current_point_2D, j);
                }
            }
            for (unsigned int i = 0; i < n_nodal_functions; i++) {
                for (unsigned int j = 0; j < 3; j++) {
                    const Tensor<1, 2, double> temp =
                            fe_q.shape_grad_component(i, current_point_2D, j);
                    gradients[i, j] = temp[0];
                    gradients[i + 3, j] = temp[1];
                }
            }
            for (unsigned int i = 0; i < n_nedelec_functions; i++) {
                for (unsigned int j = 0; j < 3; j++) {
                    const Tensor<1, 2, double> temp =
                            fe_nedelec.shape_grad_component(i, current_point_2D, j);
                    gradients[i + n_nodal_functions, j] = temp[0];
                    gradients[i + n_nodal_functions, j + 3] = temp[1];
                }
            }
        }
    }
}

#endif
