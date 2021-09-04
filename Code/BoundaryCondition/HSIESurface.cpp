#include "HSIESurface.h"
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include "../Core/InnerDomain.h"
#include "DofData.h"
#include "HSIEPolynomial.h"
#include "../Helpers/staticfunctions.h"
#include "./JacobianForCell.h"
#include "../Helpers/staticfunctions.h"
#include <vector>
#include "./BoundaryCondition.h"

const unsigned int MAX_DOF_NUMBER = INT_MAX;

HSIESurface::HSIESurface(unsigned int surface, unsigned int in_level)
    : BoundaryCondition(surface, in_level, Geometry.surface_extremal_coordinate[surface]),
      order(GlobalParams.HSIE_polynomial_degree),
      dof_h_q(Geometry.surface_meshes[surface]),
      Inner_Element_Order(GlobalParams.Nedelec_element_order),
      fe_nedelec(Inner_Element_Order),
      fe_q(Inner_Element_Order + 1),
      kappa(2.0 * GlobalParams.Pi / GlobalParams.Lambda) {
    dof_h_nedelec.initialize(Geometry.surface_meshes[surface], fe_nedelec);
    dof_h_q.initialize(Geometry.surface_meshes[surface], fe_q);
    set_mesh_boundary_ids();
    dof_counter = 0;
    k0 = GlobalParams.kappa_0;
}

HSIESurface::~HSIESurface() {}

DofDataVector HSIESurface::get_dof_data_for_cell(CellIterator2D cell_nedelec, CellIterator2D cell_q) {
  DofDataVector ret;

  // get cell dofs:
  std::string cell_id = cell_nedelec->id().to_string();
  unsigned int *nedelec_edge_ids = new unsigned int[4];
  unsigned int *q_edge_ids = new unsigned int[4];
  unsigned int *vertex_ids = new unsigned int[4];
  
  // get edge dofs:
  for (unsigned int i = 0; i < 4; i++) {
    nedelec_edge_ids[i] = cell_nedelec->face_index(i);
    q_edge_ids[i] = cell_q->face_index(i);
    vertex_ids[i] = cell_q->vertex_index(i);
  }

  // add cell dofs
  for (unsigned int i = 0; i < this->face_dof_data.size(); i++) {
    if (this->face_dof_data[i].base_structure_id_face == cell_id) {
      ret.push_back(this->face_dof_data[i]);
    }
  }

  // add edge-based dofs
  for (unsigned int i = 0; i < this->edge_dof_data.size(); i++) {
    if (this->edge_dof_data[i].type == DofType::EDGE || this->edge_dof_data[i].type == DofType::IFFa) {
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
  for (unsigned int i = 0; i < this->vertex_dof_data.size(); i++) {
    unsigned int idx = this->vertex_dof_data[i].base_structure_id_non_face;
    if (idx == vertex_ids[0] || idx == vertex_ids[1] || idx == vertex_ids[2] || idx == vertex_ids[3]) {
      ret.push_back(this->vertex_dof_data[i]);
    }
  }

  return ret;
}

DofDataVector HSIESurface::get_dof_data_for_base_dof_nedelec(unsigned int in_index) {
  DofDataVector ret;
  for (unsigned int index = 0; index < this->edge_dof_data.size(); index++) {
    if ((this->edge_dof_data[index].base_dof_index == in_index)
        && (this->edge_dof_data[index].type != DofType::RAY
            && this->edge_dof_data[index].type != DofType::IFFb)) {
      ret.push_back(this->edge_dof_data[index]);
    }
  }
  for (unsigned int index = 0; index < this->vertex_dof_data.size(); index++) {
    if ((this->vertex_dof_data[index].base_dof_index == in_index)
        && (this->vertex_dof_data[index].type != DofType::RAY
            && this->vertex_dof_data[index].type != DofType::IFFb)) {
      ret.push_back(this->vertex_dof_data[index]);
    }
  }
  for (unsigned int index = 0; index < this->face_dof_data.size(); index++) {
    if ((this->face_dof_data[index].base_dof_index == in_index)
        && (this->face_dof_data[index].type != DofType::RAY
            && this->face_dof_data[index].type != DofType::IFFb)) {
      ret.push_back(this->face_dof_data[index]);
    }
  }
  return ret;
}

DofDataVector HSIESurface::get_dof_data_for_base_dof_q(unsigned int in_index) {
  DofDataVector ret;
  for (unsigned int index = 0; index < this->edge_dof_data.size(); index++) {
    if ((this->edge_dof_data[index].base_dof_index == in_index)
        && (this->edge_dof_data[index].type == DofType::RAY
            || this->edge_dof_data[index].type == DofType::IFFb)) {
      ret.push_back(this->edge_dof_data[index]);
    }
  }
  for (unsigned int index = 0; index < this->vertex_dof_data.size(); index++) {
    if ((this->vertex_dof_data[index].base_dof_index == in_index)
        && (this->vertex_dof_data[index].type == DofType::RAY
            || this->vertex_dof_data[index].type == DofType::IFFb)) {
      ret.push_back(this->vertex_dof_data[index]);
    }
  }
  for (unsigned int index = 0; index < this->face_dof_data.size(); index++) {
    if ((this->face_dof_data[index].base_dof_index == in_index)
        && (this->face_dof_data[index].type == DofType::RAY
            || this->face_dof_data[index].type == DofType::IFFb)) {
      ret.push_back(this->face_dof_data[index]);
    }
  }
  return ret;
}

void HSIESurface::fill_matrix(
    dealii::PETScWrappers::SparseMatrix *matrix, NumericVectorDistributed* rhs, Constraints *constraints) {
    HSIEPolynomial::computeDandI(order + 2, k0);
    auto it = dof_h_nedelec.begin();
    auto end = dof_h_nedelec.end();
    QGauss<2> quadrature_formula(2);
    FEValues<2, 2> fe_q_values(fe_q, quadrature_formula,
                              update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    FEValues<2, 2> fe_n_values(fe_nedelec, quadrature_formula,
                              update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    std::vector<Point<2>> quadrature_points;
    auto temp_it = dof_h_nedelec.begin();
    auto temp_it2 = dof_h_q.begin();
    unsigned int dofs_per_cell = this->get_dof_data_for_cell(temp_it, temp_it2).size();
    FullMatrix<ComplexNumber> cell_matrix(dofs_per_cell, dofs_per_cell);
    unsigned int cell_counter = 0;
    auto it2 = dof_h_q.begin();
    for (; it != end; ++it) {
      FaceAngelingData fad = build_fad_for_cell(it);
      JacobianForCell jacobian_for_cell = {fad, b_id, additional_coordinate};
      cell_matrix = 0;
      DofDataVector cell_dofs = this->get_dof_data_for_cell(it, it2);
      std::vector<HSIEPolynomial> polynomials;
      std::vector<unsigned int> q_dofs(fe_q.dofs_per_cell);
      std::vector<unsigned int> n_dofs(fe_nedelec.dofs_per_cell);
      it2->get_dof_indices(q_dofs);
      it->get_dof_indices(n_dofs);
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        polynomials.push_back(HSIEPolynomial(cell_dofs[i], k0));
      }
      std::vector<unsigned int> local_related_fe_index;
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        if (cell_dofs[i].type == DofType::RAY || cell_dofs[i].type == DofType::IFFb) {
          for (unsigned int j = 0; j < q_dofs.size(); j++) {
            if (q_dofs[j] == cell_dofs[i].base_dof_index) {
              local_related_fe_index.push_back(j);
              break;
            }
          }
        } else {
          for (unsigned int j = 0; j < n_dofs.size(); j++) {
            if (n_dofs[j] == cell_dofs[i].base_dof_index) {
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
      std::vector<std::vector<HSIEPolynomial>> contribution_value;
      std::vector<std::vector<HSIEPolynomial>> contribution_curl;
      JacobianAndTensorData C_G_J;
      for (unsigned int q_point = 0; q_point < quadrature_points.size();
          q_point++) {
        C_G_J = jacobian_for_cell.get_C_G_and_J(quadrature_points[q_point]);
        for (unsigned int i = 0; i < cell_dofs.size(); i++) {
          DofData &u = cell_dofs[i];
          if (cell_dofs[i].type == DofType::RAY
              || cell_dofs[i].type == DofType::IFFb) {
            contribution_curl.push_back(
                build_curl_term_q(u.hsie_order,
                    fe_q_values.shape_grad(local_related_fe_index[i], q_point)));
            contribution_value.push_back(
                build_non_curl_term_q(u.hsie_order,
                    fe_q_values.shape_value(local_related_fe_index[i], q_point)));
          } else {
            contribution_curl.push_back(
                build_curl_term_nedelec(u.hsie_order,
                    fe_n_values.shape_grad_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_grad_component(local_related_fe_index[i],
                        q_point, 1),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 1)));
            contribution_value.push_back(
                build_non_curl_term_nedelec(u.hsie_order,
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 1)));
          }
        }

        double JxW = jxw_values[q_point];
        const double eps_kappa_2 = Geometry.eps_kappa_2(undo_transform(quadrature_points[q_point]));
        for (unsigned int i = 0; i < cell_dofs.size(); i++) {
          for (unsigned int j = 0; j < cell_dofs.size(); j++) {
            ComplexNumber part =
                (evaluate_a(contribution_curl[i], contribution_curl[j], C_G_J.C)
                + eps_kappa_2 * evaluate_a(contribution_value[i], contribution_value[j], C_G_J.G)) *
                JxW;
              cell_matrix[i][j] += part;
          }
        }
      }
      std::vector<unsigned int> local_indices;
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        local_indices.push_back(cell_dofs[i].global_index);
      }
      Vector<ComplexNumber> cell_rhs(cell_dofs.size());
      cell_rhs = 0;
      constraints->distribute_local_to_global(cell_matrix, cell_rhs, local_indices, *matrix, *rhs);
      it2++;
      cell_counter++;
    }
}

void HSIESurface::fill_matrix(
    dealii::SparseMatrix<ComplexNumber> *matrix, Constraints *constraints) {
    HSIEPolynomial::computeDandI(order + 2, k0);
    auto it = dof_h_nedelec.begin();
    auto end = dof_h_nedelec.end();
    QGauss<2> quadrature_formula(2);
    FEValues<2, 2> fe_q_values(fe_q, quadrature_formula,
                              update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    FEValues<2, 2> fe_n_values(fe_nedelec, quadrature_formula,
                              update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    std::vector<Point<2>> quadrature_points;
    auto temp_it = dof_h_nedelec.begin();
    auto temp_it2 = dof_h_q.begin();
    unsigned int dofs_per_cell = this->get_dof_data_for_cell(temp_it, temp_it2).size();
    FullMatrix<ComplexNumber> cell_matrix(dofs_per_cell, dofs_per_cell);
    unsigned int cell_counter = 0;
    auto it2 = dof_h_q.begin();
    for (; it != end; ++it) {
      FaceAngelingData fad = build_fad_for_cell(it);
      JacobianForCell jacobian_for_cell = {fad, b_id, additional_coordinate};
      cell_matrix = 0;
      DofDataVector cell_dofs = this->get_dof_data_for_cell(it, it2);
      std::vector<HSIEPolynomial> polynomials;
      std::vector<unsigned int> q_dofs(fe_q.dofs_per_cell);
      std::vector<unsigned int> n_dofs(fe_nedelec.dofs_per_cell);
      it2->get_dof_indices(q_dofs);
      it->get_dof_indices(n_dofs);
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        polynomials.push_back(HSIEPolynomial(cell_dofs[i], k0));
      }
      std::vector<unsigned int> local_related_fe_index;
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        if (cell_dofs[i].type == DofType::RAY || cell_dofs[i].type == DofType::IFFb) {
          for (unsigned int j = 0; j < q_dofs.size(); j++) {
            if (q_dofs[j] == cell_dofs[i].base_dof_index) {
              local_related_fe_index.push_back(j);
              break;
            }
          }
        } else {
          for (unsigned int j = 0; j < n_dofs.size(); j++) {
            if (n_dofs[j] == cell_dofs[i].base_dof_index) {
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
      std::vector<std::vector<HSIEPolynomial>> contribution_value;
      std::vector<std::vector<HSIEPolynomial>> contribution_curl;
      JacobianAndTensorData C_G_J;
      for (unsigned int q_point = 0; q_point < quadrature_points.size();
          q_point++) {
        C_G_J = jacobian_for_cell.get_C_G_and_J(quadrature_points[q_point]);
        for (unsigned int i = 0; i < cell_dofs.size(); i++) {
          DofData &u = cell_dofs[i];
          if (cell_dofs[i].type == DofType::RAY
              || cell_dofs[i].type == DofType::IFFb) {
            contribution_curl.push_back(
                build_curl_term_q(u.hsie_order,
                    fe_q_values.shape_grad(local_related_fe_index[i], q_point)));
            contribution_value.push_back(
                build_non_curl_term_q(u.hsie_order,
                    fe_q_values.shape_value(local_related_fe_index[i], q_point)));
          } else {
            contribution_curl.push_back(
                build_curl_term_nedelec(u.hsie_order,
                    fe_n_values.shape_grad_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_grad_component(local_related_fe_index[i],
                        q_point, 1),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 1)));
            contribution_value.push_back(
                build_non_curl_term_nedelec(u.hsie_order,
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 1)));
          }
        }

        double JxW = jxw_values[q_point];
        const double eps_kappa_2 = Geometry.eps_kappa_2(undo_transform(quadrature_points[q_point]));
        for (unsigned int i = 0; i < cell_dofs.size(); i++) {
          for (unsigned int j = 0; j < cell_dofs.size(); j++) {
            ComplexNumber part =
                (evaluate_a(contribution_curl[i], contribution_curl[j], C_G_J.C)
                + eps_kappa_2 * evaluate_a(contribution_value[i], contribution_value[j], C_G_J.G)) *
                JxW;
              cell_matrix[i][j] += part;
          }
        }
      }
      std::vector<unsigned int> local_indices;
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        local_indices.push_back(cell_dofs[i].global_index - Geometry.levels[level].inner_first_dof);
      }
      constraints->distribute_local_to_global(cell_matrix, local_indices, *matrix);
      it2++;
      cell_counter++;
    }
}

auto HSIESurface::build_fad_for_cell(CellIterator2D) -> FaceAngelingData {
  FaceAngelingData ret;
  for(unsigned int i = 0; i < ret.size(); i++) {
    ret[i].is_x_angled = false;
    ret[i].is_y_angled = false;
    ret[i].position_of_base_point = {};
  }
  return ret;
}

void HSIESurface::fill_matrix(
    dealii::PETScWrappers::SparseMatrix *mass_matrix, dealii::PETScWrappers::SparseMatrix *stiffness_matrix, NumericVectorDistributed* rhs, Constraints *constraints) {
    HSIEPolynomial::computeDandI(order + 2, k0);
    auto it = dof_h_nedelec.begin();
    auto end = dof_h_nedelec.end();

    QGauss<2> quadrature_formula(2);
    FEValues<2, 2> fe_q_values(fe_q, quadrature_formula,
                              update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    FEValues<2, 2> fe_n_values(fe_nedelec, quadrature_formula,
                              update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    std::vector<Point<2>> quadrature_points;
    auto temp_it = dof_h_nedelec.begin();
    auto temp_it2 = dof_h_q.begin();
    unsigned int dofs_per_cell = this->get_dof_data_for_cell(temp_it, temp_it2).size();
    FullMatrix<ComplexNumber> cell_stiffness_matrix(dofs_per_cell,
        dofs_per_cell);
    FullMatrix<ComplexNumber> cell_mass_matrix(dofs_per_cell,
        dofs_per_cell);
    unsigned int cell_counter = 0;
    auto it2 = dof_h_q.begin();
    for (; it != end; ++it) {
      FaceAngelingData fad = build_fad_for_cell(it);
      JacobianForCell jacobian_for_cell = {fad, b_id, additional_coordinate};
      cell_mass_matrix = 0;
      cell_stiffness_matrix = 0;
      DofDataVector cell_dofs = this->get_dof_data_for_cell(it, it2);
      std::vector<HSIEPolynomial> polynomials;
      std::vector<unsigned int> q_dofs(fe_q.dofs_per_cell);
      std::vector<unsigned int> n_dofs(fe_nedelec.dofs_per_cell);
      it2->get_dof_indices(q_dofs);
      it->get_dof_indices(n_dofs);
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        polynomials.push_back(HSIEPolynomial(cell_dofs[i], k0));
      }
      std::vector<unsigned int> local_related_fe_index;
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        if (cell_dofs[i].type == DofType::RAY || cell_dofs[i].type == DofType::IFFb) {
          for (unsigned int j = 0; j < q_dofs.size(); j++) {
            if (q_dofs[j] == cell_dofs[i].base_dof_index) {
              local_related_fe_index.push_back(j);
              break;
            }
          }
        } else {
          for (unsigned int j = 0; j < n_dofs.size(); j++) {
            if (n_dofs[j] == cell_dofs[i].base_dof_index) {
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
      std::vector<std::vector<HSIEPolynomial>> contribution_value;
      std::vector<std::vector<HSIEPolynomial>> contribution_curl;
      JacobianAndTensorData C_G_J;
      for (unsigned int q_point = 0; q_point < quadrature_points.size();
          q_point++) {
        C_G_J = jacobian_for_cell.get_C_G_and_J(quadrature_points[q_point]);
        for (unsigned int i = 0; i < cell_dofs.size(); i++) {
          DofData &u = cell_dofs[i];
          if (cell_dofs[i].type == DofType::RAY || cell_dofs[i].type == DofType::IFFb) {
            contribution_curl.push_back(
                build_curl_term_q(u.hsie_order,
                    fe_q_values.shape_grad(local_related_fe_index[i], q_point)));
            contribution_value.push_back(
                build_non_curl_term_q(u.hsie_order,
                    fe_q_values.shape_value(local_related_fe_index[i], q_point)));
          } else {
            contribution_curl.push_back(
                build_curl_term_nedelec(u.hsie_order,
                    fe_n_values.shape_grad_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_grad_component(local_related_fe_index[i],
                        q_point, 1),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 1)));
            contribution_value.push_back(
                build_non_curl_term_nedelec(u.hsie_order,
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 0),
                    fe_n_values.shape_value_component(local_related_fe_index[i],
                        q_point, 1)));
          }
        }

        double JxW = jxw_values[q_point];
        const double eps_kappa_2 = Geometry.eps_kappa_2(undo_transform(quadrature_points[q_point]));
        for (unsigned int i = 0; i < cell_dofs.size(); i++) {
          for (unsigned int j = 0; j < cell_dofs.size(); j++) {
            cell_mass_matrix[i][j] += eps_kappa_2 * evaluate_a(contribution_value[i], contribution_value[j], C_G_J.G) * JxW;
            cell_stiffness_matrix[i][j] += evaluate_a(contribution_curl[i], contribution_curl[j], C_G_J.C) * JxW;
          }
        }
      }
      std::vector<unsigned int> local_indices;
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        local_indices.push_back(cell_dofs[i].global_index);
      }
      Vector<ComplexNumber> cell_rhs(cell_dofs.size());
      cell_rhs = 0;
      constraints->distribute_local_to_global(cell_mass_matrix, cell_rhs, local_indices, *mass_matrix, *rhs);
      constraints->distribute_local_to_global(cell_stiffness_matrix, cell_rhs, local_indices, *stiffness_matrix, *rhs);
      it2++;
      cell_counter++;
    }
}

void HSIESurface::fill_matrix(
    dealii::PETScWrappers::MPI::SparseMatrix *matrix, NumericVectorDistributed* rhs, Constraints *constraints) {
    HSIEPolynomial::computeDandI(order + 2, k0);
    auto it = dof_h_nedelec.begin();
    auto end = dof_h_nedelec.end();

    QGauss<2> quadrature_formula(2);
    FEValues<2, 2> fe_q_values(fe_q, quadrature_formula,
                              update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    FEValues<2, 2> fe_n_values(fe_nedelec, quadrature_formula,
                              update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);
    std::vector<Point<2>> quadrature_points;
    const unsigned int dofs_per_cell =
        GeometryInfo<2>::vertices_per_cell * compute_dofs_per_vertex() +
        GeometryInfo<2>::lines_per_cell * compute_dofs_per_edge(false) +
        compute_dofs_per_face(false);
      FullMatrix<ComplexNumber> cell_matrix(dofs_per_cell,
          dofs_per_cell);
    unsigned int cell_counter = 0;
    auto it2 = dof_h_q.begin();
    for (; it != end; ++it) {
      FaceAngelingData fad = build_fad_for_cell(it);
      JacobianForCell jacobian_for_cell = {fad, b_id, additional_coordinate};
      cell_matrix = 0;
      DofDataVector cell_dofs = this->get_dof_data_for_cell(it, it2);
      std::vector<HSIEPolynomial> polynomials;
      std::vector<unsigned int> q_dofs(fe_q.dofs_per_cell);
      std::vector<unsigned int> n_dofs(fe_nedelec.dofs_per_cell);
      it2->get_dof_indices(q_dofs);
      it->get_dof_indices(n_dofs);
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        polynomials.push_back(HSIEPolynomial(cell_dofs[i], k0));
      }
      std::vector<unsigned int> local_related_fe_index;
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        if (cell_dofs[i].type == DofType::RAY || cell_dofs[i].type == DofType::IFFb) {
          for (unsigned int j = 0; j < q_dofs.size(); j++) {
            if (q_dofs[j] == cell_dofs[i].base_dof_index) {
              local_related_fe_index.push_back(j);
              break;
            }
          }
        } else {
          for (unsigned int j = 0; j < n_dofs.size(); j++) {
            if (n_dofs[j] == cell_dofs[i].base_dof_index) {
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
      std::vector<std::vector<HSIEPolynomial>> contribution_value;
      std::vector<std::vector<HSIEPolynomial>> contribution_curl;
      JacobianAndTensorData C_G_J;
      for (unsigned int q_point = 0; q_point < quadrature_points.size();
          q_point++) {
        C_G_J = jacobian_for_cell.get_C_G_and_J(quadrature_points[q_point]);
        for (unsigned int i = 0; i < cell_dofs.size(); i++) {
          DofData &u = cell_dofs[i];
          if (cell_dofs[i].type == DofType::RAY || cell_dofs[i].type == DofType::IFFb) {
            contribution_curl.push_back(
              build_curl_term_q(u.hsie_order, fe_q_values.shape_grad(local_related_fe_index[i], q_point)));
            contribution_value.push_back(
              build_non_curl_term_q(u.hsie_order, fe_q_values.shape_value(local_related_fe_index[i], q_point)));
          } else {
            contribution_curl.push_back(
              build_curl_term_nedelec(u.hsie_order,
                fe_n_values.shape_grad_component(local_related_fe_index[i], q_point, 0),
                fe_n_values.shape_grad_component(local_related_fe_index[i], q_point, 1),
                fe_n_values.shape_value_component(local_related_fe_index[i], q_point, 0),
                fe_n_values.shape_value_component(local_related_fe_index[i], q_point, 1)));
            contribution_value.push_back(
              build_non_curl_term_nedelec(u.hsie_order,
                fe_n_values.shape_value_component(local_related_fe_index[i], q_point, 0),
                fe_n_values.shape_value_component(local_related_fe_index[i], q_point, 1)));
          }
        }

        double JxW = jxw_values[q_point];
        const double eps_kappa_2 = Geometry.eps_kappa_2(undo_transform(quadrature_points[q_point]));
        for (unsigned int i = 0; i < cell_dofs.size(); i++) {
          for (unsigned int j = 0; j < cell_dofs.size(); j++) {
            ComplexNumber part =
                (evaluate_a(contribution_curl[i], contribution_curl[j], C_G_J.C)
                - eps_kappa_2 * evaluate_a(contribution_value[i], contribution_value[j], C_G_J.G)) *
                JxW;
              cell_matrix[i][j] += part;
          }
        }
      }
      std::vector<unsigned int> local_indices;
      for (unsigned int i = 0; i < cell_dofs.size(); i++) {
        local_indices.push_back(cell_dofs[i].global_index);
      }
      Vector<ComplexNumber> cell_rhs(cell_dofs.size());
      cell_rhs = 0;
      constraints->distribute_local_to_global(cell_matrix, cell_rhs, local_indices, *matrix, *rhs);
      it2++;
      cell_counter++;
    }
    matrix->compress(dealii::VectorOperation::add);
}

void HSIESurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constraints) {
  auto it = dof_h_nedelec.begin();
  auto end = dof_h_nedelec.end();
  const unsigned int dofs_per_cell =
      GeometryInfo<2>::vertices_per_cell * compute_dofs_per_vertex() +
      GeometryInfo<2>::lines_per_cell * compute_dofs_per_edge(false) +
      compute_dofs_per_face(false);
  auto it2 = dof_h_q.begin();
  for (; it != end; ++it) {
    DofDataVector cell_dofs = this->get_dof_data_for_cell(it, it2);
    std::vector<unsigned int> local_indices;
    for (unsigned int i = 0; i < cell_dofs.size(); i++) {
      local_indices.push_back(cell_dofs[i].global_index);
    }
    local_indices = transform_local_to_global_dofs(local_indices);
    in_constraints->add_entries_local_to_global(local_indices, *in_dsp);
    it2++;
  }
}

DofCountsStruct HSIESurface::compute_n_edge_dofs() {
  DoFHandler<2>::active_cell_iterator cell;
  DoFHandler<2>::active_cell_iterator cell2;
  DoFHandler<2>::active_cell_iterator endc;
  endc = dof_h_nedelec.end();
  DofCountsStruct ret;
  cell2 = dof_h_q.begin_active();
  Geometry.surface_meshes[b_id].clear_user_flags();
  for (cell = dof_h_nedelec.begin_active(); cell != endc; cell++) {
    for (unsigned int edge = 0; edge < GeometryInfo<2>::lines_per_cell; edge++) {
      if (!cell->line(edge)->user_flag_set()) {
        update_dof_counts_for_edge(cell, edge, ret);
        register_new_edge_dofs(cell, cell2, edge);
        cell->line(edge)->set_user_flag();
      }
    }
    cell2++;
  }
  return ret;
}

DofCountsStruct HSIESurface::compute_n_vertex_dofs() {
  std::set<unsigned int> touched_vertices;
  DoFHandler<2>::active_cell_iterator cell;
  DoFHandler<2>::active_cell_iterator endc;
  endc = dof_h_q.end();
  DofCountsStruct ret;
  for (cell = dof_h_q.begin_active(); cell != endc; cell++) {
    // for each edge
    for (unsigned int vertex = 0; vertex < GeometryInfo<2>::vertices_per_cell;
         vertex++) {
      unsigned int idx = cell->vertex_dof_index(vertex, 0);
      if (touched_vertices.end() == touched_vertices.find(idx)) {
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

DofCountsStruct HSIESurface::compute_n_face_dofs() {
  std::set<std::string> touched_faces;
  DoFHandler<2>::active_cell_iterator cell;
  DoFHandler<2>::active_cell_iterator cell2;
  DoFHandler<2>::active_cell_iterator endc;
  endc = dof_h_nedelec.end();
  DofCountsStruct ret;
  cell2 = dof_h_q.begin_active();
  for (cell = dof_h_nedelec.begin_active(); cell != endc; cell++) {
    if (touched_faces.end() == touched_faces.find(cell->id().to_string())) {
      update_dof_counts_for_face(cell, ret);
      register_new_surface_dofs(cell, cell2);
      touched_faces.insert(cell->id().to_string());
    }
    cell2++;
  }
  return ret;
}

unsigned int HSIESurface::compute_dofs_per_edge(bool only_hsie_dofs) {
  unsigned int ret = 0;
  const unsigned int INNER_REAL_DOFS_PER_LINE = fe_nedelec.dofs_per_line;

  if (!only_hsie_dofs) {
    ret += INNER_REAL_DOFS_PER_LINE;
  }

  ret += INNER_REAL_DOFS_PER_LINE * (order + 1)
      + (INNER_REAL_DOFS_PER_LINE - 1) * (order + 2);

  return ret;
}

unsigned int HSIESurface::compute_dofs_per_face(bool only_hsie_dofs) {
  unsigned int ret = 0;
  const unsigned int INNER_REAL_NEDELEC_DOFS_PER_FACE =
      fe_nedelec.dofs_per_cell -
      dealii::GeometryInfo<2>::faces_per_cell * fe_nedelec.dofs_per_face;

  ret = INNER_REAL_NEDELEC_DOFS_PER_FACE * (order + 2) * 3;
  if (only_hsie_dofs) {
    ret -= INNER_REAL_NEDELEC_DOFS_PER_FACE;
  }
  return ret;
}

unsigned int HSIESurface::compute_dofs_per_vertex() {
  unsigned int ret = order + 2;

  return ret;
}

void HSIESurface::initialize() {
  initialize_dof_handlers_and_fe();
  compute_n_edge_dofs();
  compute_n_face_dofs();
  compute_n_vertex_dofs();
}

void HSIESurface::initialize_dof_handlers_and_fe() {
  dof_h_q.distribute_dofs(fe_q);
  dof_h_nedelec.distribute_dofs(fe_nedelec);
}

void HSIESurface::update_dof_counts_for_edge(
    const CellIterator2D, unsigned int,
    DofCountsStruct &in_dof_count) {
  const unsigned int dofs_per_edge_all = compute_dofs_per_edge(false);
  const unsigned int dofs_per_edge_hsie = compute_dofs_per_edge(true);
  in_dof_count.total += dofs_per_edge_all;
  in_dof_count.hsie += dofs_per_edge_hsie;
  in_dof_count.non_hsie += dofs_per_edge_all - dofs_per_edge_hsie;
}

void HSIESurface::update_dof_counts_for_face(
    const CellIterator2D,
    DofCountsStruct &in_dof_count) {
  const unsigned int dofs_per_face_all = compute_dofs_per_face(false);
  const unsigned int dofs_per_face_hsie = compute_dofs_per_face(true);
  in_dof_count.total += dofs_per_face_all;
  in_dof_count.hsie += dofs_per_face_hsie;
  in_dof_count.non_hsie += dofs_per_face_all - dofs_per_face_hsie;
}

void HSIESurface::update_dof_counts_for_vertex(
    const CellIterator2D, unsigned int,
    unsigned int, DofCountsStruct &in_dof_count) {
  const unsigned int dofs_per_vertex_all = compute_dofs_per_vertex();

  in_dof_count.total += dofs_per_vertex_all;
  in_dof_count.hsie += dofs_per_vertex_all;
}

void HSIESurface::register_new_vertex_dofs(
    CellIterator2D cell, unsigned int dof_index,
    unsigned int vertex) {
  const int max_hsie_order = order;
  for (int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order++) {
    register_single_dof(cell->vertex_index(vertex), hsie_order, -1, DofType::RAY, vertex_dof_data, dof_index);
  }
}

void HSIESurface::register_new_edge_dofs(CellIterator2D cell_nedelec, CellIterator2D cell_q, unsigned int edge) {
  const int max_hsie_order = order;
  // EDGE Dofs
  std::vector<unsigned int> local_dofs(fe_nedelec.dofs_per_line);
  cell_nedelec->line(edge)->get_dof_indices(local_dofs);
  bool orientation = false;
  if(cell_nedelec->line(edge)->vertex_index(0) > cell_nedelec->line(edge)->vertex_index(1)) {
    orientation = get_orientation(undo_transform(cell_nedelec->line(edge)->vertex(0)), undo_transform(cell_nedelec->line(edge)->vertex(1)));
  } else {
    orientation = get_orientation(undo_transform(cell_nedelec->line(edge)->vertex(1)), undo_transform(cell_nedelec->line(edge)->vertex(0)));
  }
  
  for (int inner_order = 0; inner_order < static_cast<int>(fe_nedelec.dofs_per_line); inner_order++) {
    register_single_dof(cell_nedelec->face_index(edge), -1, inner_order + 1, DofType::EDGE, edge_dof_data, local_dofs[inner_order], orientation);
    Position bp = undo_transform(cell_nedelec->face(edge)->center(false, false));
    InterfaceDofData dof_data;
    dof_data.index = edge_dof_data[edge_dof_data.size() - 1].global_index;
    dof_data.order = inner_order;
    dof_data.base_point = bp;
    add_surface_relevant_dof(dof_data);
  }

  // INFINITE FACE Dofs Type a
  for (int inner_order = 0; inner_order < static_cast<int>(fe_nedelec.dofs_per_line); inner_order++) {
    for (int hsie_order = 0; hsie_order <= max_hsie_order; hsie_order++) {
      register_single_dof(cell_nedelec->face_index(edge), hsie_order, inner_order + 1, DofType::IFFa, edge_dof_data, local_dofs[inner_order], orientation);
    }
  }
  // INFINITE FACE Dofs Type b
  local_dofs.clear();
  local_dofs.resize(fe_q.dofs_per_line + 2 * fe_q.dofs_per_vertex);
  cell_q->line(edge)->get_dof_indices(local_dofs);
  IndexSet line_dofs(MAX_DOF_NUMBER);
  IndexSet non_line_dofs(MAX_DOF_NUMBER);
  for (unsigned int i = 0; i < local_dofs.size(); i++) {
    line_dofs.add_index(local_dofs[i]);
  }
  for (unsigned int i = 0; i < fe_q.dofs_per_vertex; i++) {
    non_line_dofs.add_index(cell_q->line(edge)->vertex_dof_index(0, i));
    non_line_dofs.add_index(cell_q->line(edge)->vertex_dof_index(1, i));
  }
  line_dofs.subtract_set(non_line_dofs);
  for (int inner_order = 0; inner_order < static_cast<int>(line_dofs.n_elements());
       inner_order++) {
    for (int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order++) {
      register_single_dof(cell_q->face_index(edge), hsie_order, inner_order, DofType::IFFb, edge_dof_data, line_dofs.nth_index_in_set(inner_order), orientation);
    }
  }
}

void HSIESurface::register_new_surface_dofs(CellIterator2D cell_nedelec, CellIterator2D cell_q) {
  const int max_hsie_order = order;
  std::vector<unsigned int> surface_dofs(fe_nedelec.dofs_per_cell);
  cell_nedelec->get_dof_indices(surface_dofs);
  IndexSet surf_dofs(MAX_DOF_NUMBER);
  IndexSet edge_dofs(MAX_DOF_NUMBER);
  for (unsigned int i = 0; i < surface_dofs.size(); i++) {
    surf_dofs.add_index(surface_dofs[i]);
  }
  for (unsigned int i = 0; i < dealii::GeometryInfo<2>::lines_per_cell; i++) {
    std::vector<unsigned int> line_dofs(fe_nedelec.dofs_per_line);
    cell_nedelec->line(i)->get_dof_indices(line_dofs);
    for (unsigned int j = 0; j < line_dofs.size(); j++) {
      edge_dofs.add_index(line_dofs[j]);
    }
  }
  surf_dofs.subtract_set(edge_dofs);
  std::string id = cell_q->id().to_string();
  const unsigned int nedelec_dof_count = dof_h_nedelec.n_dofs();
  dealii::Vector<ComplexNumber> vec_temp(nedelec_dof_count);
  // SURFACE functions
  for (unsigned int inner_order = 0; inner_order < surf_dofs.n_elements(); inner_order++) {
    register_single_dof(cell_nedelec->id().to_string(), -1, inner_order, DofType::SURFACE, face_dof_data, surf_dofs.nth_index_in_set(inner_order));
    Position bp = undo_transform(cell_nedelec->center());
    InterfaceDofData dof_data;
    dof_data.index = face_dof_data[face_dof_data.size() - 1].global_index;
    dof_data.base_point = bp;
    dof_data.order = inner_order;
    add_surface_relevant_dof(dof_data);
  }

  // SEGMENT functions a
  for (unsigned int inner_order = 0; inner_order < surf_dofs.n_elements(); inner_order++) {
    for (int hsie_order = 0; hsie_order <= max_hsie_order; hsie_order++) {
      register_single_dof(id, hsie_order, inner_order, DofType::SEGMENTa, face_dof_data, surf_dofs.nth_index_in_set(inner_order));
    }
  }

  for (unsigned int inner_order = 0; inner_order < surf_dofs.n_elements(); inner_order++) {
    for (int hsie_order = -1; hsie_order <= max_hsie_order; hsie_order++) {
      register_single_dof(id, hsie_order, inner_order, DofType::SEGMENTb, face_dof_data, surf_dofs.nth_index_in_set(inner_order));
    }
  }
}

void HSIESurface::register_single_dof( std::string in_id, const int in_hsie_order, const int in_inner_order,
    DofType in_dof_type, DofDataVector &in_vector, unsigned int in_base_dof_index) {
  DofData dd(in_id);
  dd.global_index = register_dof();
  dd.hsie_order = in_hsie_order;
  dd.inner_order = in_inner_order;
  dd.type = in_dof_type;
  dd.set_base_dof(in_base_dof_index);
  dd.update_nodal_basis_flag();
  in_vector.push_back(dd);
}

void HSIESurface::register_single_dof( unsigned int in_id, const int in_hsie_order, const int in_inner_order,
    DofType in_dof_type, DofDataVector &in_vector, unsigned int in_base_dof_index, bool orientation) {
  DofData dd(in_id);
  dd.global_index = register_dof();
  dd.hsie_order = in_hsie_order;
  dd.inner_order = in_inner_order;
  dd.type = in_dof_type;
  dd.orientation = orientation;
  dd.set_base_dof(in_base_dof_index);
  dd.update_nodal_basis_flag();
  in_vector.push_back(dd);
}

unsigned int HSIESurface::register_dof() {
  dof_counter++;
  return dof_counter - 1;
}

ComplexNumber HSIESurface::evaluate_a(std::vector<HSIEPolynomial> &u, std::vector<HSIEPolynomial> &v, Tensor<2,3,double> G) {
  ComplexNumber result(0, 0);
  for(unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      for (unsigned int k = 0; k < std::min(u[i].a.size(), v[j].a.size()); k++) {
        result += G[i][j] * u[i].a[k] * v[j].a[k];
      }
    }
  }
  return result;
}

std::vector<HSIEPolynomial> HSIESurface::build_curl_term_nedelec(
    unsigned int dof_hsie_order,
    const Tensor<1, 2> fe_shape_gradient_component_0,
    const Tensor<1, 2> fe_shape_gradient_component_1,
    const double fe_shape_value_component_0,
    const double fe_shape_value_component_1) {
  std::vector<HSIEPolynomial> ret;
  HSIEPolynomial temp = HSIEPolynomial::PsiJ(dof_hsie_order, k0);
  temp.multiplyBy(fe_shape_gradient_component_0[1]);
  temp.applyI();
  HSIEPolynomial temp2 = HSIEPolynomial::PsiJ(dof_hsie_order, k0);
  temp2.multiplyBy(-1.0 * fe_shape_gradient_component_1[0]);
  temp2.applyI();
  temp.add(temp2);
  ret.push_back(temp);

  temp = HSIEPolynomial::PsiJ(dof_hsie_order, k0);
  temp.multiplyBy(-1.0 * fe_shape_value_component_1);
  temp.applyDerivative();
  ret.push_back(temp);

  temp = HSIEPolynomial::PsiJ(dof_hsie_order, k0);
  temp.multiplyBy(fe_shape_value_component_0);
  temp.applyDerivative();
  ret.push_back(temp);

  this->transform_coordinates_in_place(&ret);
  return ret;
}

std::vector<HSIEPolynomial> HSIESurface::build_non_curl_term_nedelec(
    const unsigned int dof_hsie_order, const double fe_shape_value_component_0,
    const double fe_shape_value_component_1) {
  std::vector<HSIEPolynomial> ret;
  ret.push_back(HSIEPolynomial::ZeroPolynomial());
  HSIEPolynomial temp = HSIEPolynomial::PsiJ(dof_hsie_order, k0);
  temp.multiplyBy(fe_shape_value_component_0);
  ret.push_back(temp);
  temp = HSIEPolynomial::PsiJ(dof_hsie_order, k0);
  temp.multiplyBy(fe_shape_value_component_1);
  ret.push_back(temp);
  this->transform_coordinates_in_place(&ret);
  return ret;
}

std::vector<HSIEPolynomial> HSIESurface::build_curl_term_q(
    const unsigned int dof_hsie_order, const Tensor<1, 2> fe_gradient) {
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

std::vector<HSIEPolynomial> HSIESurface::build_non_curl_term_q(
    const unsigned int dof_hsie_order, const double fe_shape_value) {
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

void HSIESurface::transform_coordinates_in_place(
    std::vector<HSIEPolynomial> *vector) {
  // The ray direction before transformation is x. This has to be adapted.
  HSIEPolynomial temp = (*vector)[0];
  switch (this->b_id) {
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

Position HSIESurface::undo_transform(dealii::Point<2> inp) {
  Position ret;
  ret[0] = inp[0];
  ret[1] = inp[1];
  ret[2] = additional_coordinate;
  switch (b_id) {
  case 0:
    ret = Transform_5_to_0(ret);
    break;
  case 1:
    ret = Transform_5_to_1(ret);
    break;
  case 2:
    ret = Transform_5_to_2(ret);
    break;
  case 3:
    ret = Transform_5_to_3(ret);
    break;
  case 4:
    ret = Transform_5_to_4(ret);
    break;
  default:
    break;
  }
  return ret;
}

Position HSIESurface::undo_transform_for_shape_function(dealii::Point<2> inp) {
  Position ret;
  ret[0] = inp[0];
  ret[1] = inp[1];
  ret[2] = 0;
  switch (b_id) {
  case 0:
    ret = Transform_5_to_0(ret);
    break;
  case 1:
    ret = Transform_5_to_1(ret);
    break;
  case 2:
    ret = Transform_5_to_2(ret);
    break;
  case 3:
    ret = Transform_5_to_3(ret);
    break;
  case 4:
    ret = Transform_5_to_4(ret);
    break;
  default:
    break;
  }
  return ret;
}

bool is_oriented_positively(Position in_p) {
  return (in_p[0] + in_p[1] + in_p[2] > 0);
}

std::vector<InterfaceDofData> HSIESurface::get_dof_association() {
  std::sort(surface_dofs.begin(), surface_dofs.end(), compareDofBaseDataAndOrientation);
  std::vector<InterfaceDofData> ret;
  copy(surface_dofs.begin(), surface_dofs.end(), back_inserter(ret));
  return ret;
}

void HSIESurface::identify_corner_cells() {

}

bool HSIESurface::check_dof_assignment_integrity() {
  HSIEPolynomial::computeDandI(order + 2, k0);
  auto it = dof_h_nedelec.begin_active();
  auto end = dof_h_nedelec.end();
  auto it2 = dof_h_q.begin_active();
  unsigned int counter = 1;
  for (; it != end; ++it) {
    if (it->id() != it2->id()) std::cout << "Identity failure!" << std::endl;
    DofDataVector cell_dofs = this->get_dof_data_for_cell(it, it2);
    std::vector<unsigned int> q_dofs(fe_q.dofs_per_cell);
    std::vector<unsigned int> n_dofs(fe_nedelec.dofs_per_cell);
    it2->get_dof_indices(q_dofs);
    it->get_dof_indices(n_dofs);
    std::vector<unsigned int> local_related_fe_index;
    bool found = false;
    for (unsigned int i = 0; i < cell_dofs.size(); i++) {
      found = false;
      if (cell_dofs[i].type == DofType::RAY ||
          cell_dofs[i].type == DofType::IFFb) {
        for (unsigned int j = 0; j < q_dofs.size(); j++) {
          if (q_dofs[j] == cell_dofs[i].base_dof_index) {
            local_related_fe_index.push_back(j);
            found = true;
          }
        }
      } else {
        for (unsigned int j = 0; j < n_dofs.size(); j++) {
          if (n_dofs[j] == cell_dofs[i].base_dof_index) {
            local_related_fe_index.push_back(j);
            found = true;
          }
        }
      }
      if (!found) {
        std::cout << "Error in dof assignment integrity!" << std::endl;
      }
    }

    if (local_related_fe_index.size() != cell_dofs.size()) {
      std::cout << "Mismatch in cell " << counter
                << ": Found indices: " << local_related_fe_index.size()
                << " of a total " << cell_dofs.size() << std::endl;
      return false;
    }
    counter++;
    it2++;
  }

  return true;
}

bool HSIESurface::check_number_of_dofs_for_cell_integrity() {
  auto it = dof_h_nedelec.begin_active();
  auto it2 = dof_h_q.begin_active();
  auto end = dof_h_nedelec.end();
  const unsigned int dofs_per_cell = 4 * compute_dofs_per_vertex() +
                                     4 * compute_dofs_per_edge(false) +
                                     compute_dofs_per_face(false);
  unsigned int counter = 0;
  for (; it != end; ++it) {
    DofDataVector cell_dofs = this->get_dof_data_for_cell(it, it2);
    if (cell_dofs.size() != dofs_per_cell) {
      for (unsigned int i = 0; i < 7; i++) {
        unsigned int count = 0;
        for (unsigned int j = 0; j < cell_dofs.size(); ++j) {
          if (cell_dofs[j].type == i) count++;
        }
        std::cout << cell_dofs.size() << " vs. " << dofs_per_cell << std::endl;
        std::cout << "For type " << i << " I found " << count << " dofs" << std::endl;
      }
      return false;
    }
    counter++;
    it2++;
  }
  return true;
}

void HSIESurface::clear_user_flags() {
  auto it = dof_h_nedelec.begin();
  const auto end = dof_h_nedelec.end();
  while (it != end) {
    it->clear_user_flag();
    for (unsigned int i = 0; i < 4; i++) {
      it->face(i)->clear_user_flag();
    }
    it++;
  }
}

Position2D get_vertex_position_for_vertex_index_in_tria(dealii::Triangulation<2> * in_tria, unsigned int vertex_id) {
  for(auto it : *in_tria) {
    for(unsigned int i = 0; i < 4; i++) {
      if(it.vertex_index(i) == vertex_id) {
        return it.vertex(i);
      }
    }
  }
  std::cout << "There was an error locating a vertex by id." << std::endl;
  return Position2D();
}

Position2D get_line_position_for_line_index_in_tria(dealii::Triangulation<2> * in_tria, unsigned int line_id) {
  for(auto it : *in_tria) {
    for(unsigned int i = 0; i < 4; i++) {
      if(it.line_index(i) == line_id) {
        return it.line(i)->center();
      }
    }
  }
  std::cout << "There was an error locating a line by id." << std::endl;
  return Position2D();
}

std::vector<Position> HSIESurface::vertex_positions_for_ids(std::vector<unsigned int> ids) {
  std::vector<Position> ret(ids.size());
  for(unsigned int vertex_index_in_array = 0; vertex_index_in_array < ids.size(); vertex_index_in_array++) {
    Position p = undo_transform(get_vertex_position_for_vertex_index_in_tria(&Geometry.surface_meshes[b_id], ids[vertex_index_in_array]));
    ret[vertex_index_in_array] = p;
  }
  return ret;
}

std::vector<Position> HSIESurface::line_positions_for_ids(std::vector<unsigned int> ids) {
  std::vector<Position> ret(ids.size());
  for(unsigned int line_index_in_array = 0; line_index_in_array < ids.size(); line_index_in_array++) {
    Position p  = undo_transform(get_line_position_for_line_index_in_tria(&Geometry.surface_meshes[b_id], ids[line_index_in_array]));
    ret[line_index_in_array] = p;  
  }
  return ret;
}

std::vector<InterfaceDofData> HSIESurface::get_dof_association_by_boundary_id(BoundaryId in_boundary_id) {
  if (in_boundary_id == b_id) {
    return this->get_dof_association();
  } 

  if (are_opposing_sites(in_boundary_id, b_id)) {
    std::vector<InterfaceDofData> surface_dofs_unsorted(0);
    return surface_dofs_unsorted;
  } 
  std::vector<InterfaceDofData> surface_dofs_unsorted;
  std::vector<unsigned int> vertex_ids = get_vertices_for_boundary_id(in_boundary_id);
  std::vector<unsigned int> line_ids = get_lines_for_boundary_id(in_boundary_id);
  std::vector<Position> vertex_positions = vertex_positions_for_ids(vertex_ids);
  std::vector<Position> line_positions = line_positions_for_ids(line_ids);
  for(unsigned int index = 0; index < vertex_dof_data.size(); index++) {
    DofData dof = vertex_dof_data[index];
    for(unsigned int index_in_ids = 0; index_in_ids < vertex_ids.size(); index_in_ids++) {
      if(vertex_ids[index_in_ids] == vertex_dof_data[index].base_structure_id_non_face) {
        InterfaceDofData new_item;
        new_item.index = dof.global_index;
        new_item.base_point = vertex_positions[index_in_ids];
        new_item.order = (dof.inner_order+1) * (dof.nodal_basis + 1);
        surface_dofs_unsorted.push_back(new_item);
      }
    }
  }

  // Construct containers with base points, orientation and index
  for(unsigned int index = 0; index < edge_dof_data.size(); index++) {
    DofData dof = edge_dof_data[index];
    for(unsigned int index_in_ids = 0; index_in_ids < line_ids.size(); index_in_ids++) {
      if(line_ids[index_in_ids] == edge_dof_data[index].base_structure_id_non_face) {
        InterfaceDofData new_item;
        new_item.index = dof.global_index;
        new_item.base_point = line_positions[index_in_ids];
        new_item.order = (dof.inner_order+1) * (dof.nodal_basis + 1);
        surface_dofs_unsorted.push_back(new_item);
      }
    }
  }
  
  // Sort the vectors.
  std::sort(surface_dofs_unsorted.begin(), surface_dofs_unsorted.end(), compareDofBaseDataAndOrientation);
  
  return surface_dofs_unsorted;
}

unsigned int HSIESurface::get_dof_count_by_boundary_id(BoundaryId in_boundary_id) {
  unsigned int ret = 0;
  if (in_boundary_id == this->b_id) {
    return dof_counter;
  } else {
    auto it = dof_h_nedelec.begin_active();
    auto it2 = dof_h_q.begin_active();
    auto end = dof_h_nedelec.end();
    std::vector<unsigned int> vertex_indices;
    for (; it != end; ++it) {
      if (it->at_boundary()) {
        for (unsigned int edge = 0; edge < 4; edge++) {
          if (it->face(edge)->boundary_id() == in_boundary_id) {
            ret += compute_dofs_per_edge(true);
            const unsigned int first_index = it2->face(edge)->vertex_index(0);
            const unsigned int second_index = it2->face(edge)->vertex_index(1);
            auto search = find(vertex_indices.begin(), vertex_indices.end(),
                first_index);
            if (search != vertex_indices.end()) {
              ret += compute_dofs_per_vertex();
              vertex_indices.push_back(first_index);
            }
            search = find(vertex_indices.begin(), vertex_indices.end(), second_index);
            if (search != vertex_indices.end()) {
              ret += compute_dofs_per_vertex();
              vertex_indices.push_back(second_index);
            }
          }
        }
      }
      it2++;
    }
  }
  return ret;
}

void HSIESurface::add_surface_relevant_dof(InterfaceDofData dof_data) {
  surface_dofs.emplace_back(dof_data);
}

void HSIESurface::set_V0(Position in_V0) {
  V0 = in_V0;
}

void HSIESurface::compute_extreme_vertex_coordinates() {
  std::array<double, 3> upper_coordinates = {-100000, -100000, -100000};
  std::array<double, 3> lower_coordinates = {100000, 100000, 100000};
  
  for(auto it = Geometry.surface_meshes[b_id].begin(); it != Geometry.surface_meshes[b_id].end(); it++) {
    for(unsigned int ind = 0; ind < 4; ind++) {
      Position vertex_position = undo_transform(it->vertex(ind));
      for(unsigned int i = 0; i < 3; i++) {
        if(vertex_position[i] > upper_coordinates[i]) {
          upper_coordinates[i] = vertex_position[i];
        }
        if(vertex_position[i] < lower_coordinates[i]) {
          lower_coordinates[i] = vertex_position[i];
        }
      }
    }
  }
  boundary_vertex_coordinates[0] = lower_coordinates[0];
  boundary_vertex_coordinates[1] = upper_coordinates[0];
  boundary_vertex_coordinates[2] = lower_coordinates[1];
  boundary_vertex_coordinates[3] = upper_coordinates[1];
  boundary_vertex_coordinates[4] = lower_coordinates[2];
  boundary_vertex_coordinates[5] = upper_coordinates[2];
  boundary_coordinates_computed = true;
}

bool HSIESurface::is_point_at_boundary(Position2D in_p, BoundaryId in_bid) {
  if(!boundary_coordinates_computed) {
    compute_extreme_vertex_coordinates();
  }
  if(are_opposing_sites(in_bid, b_id) || in_bid == b_id) return true;
  Position full_position = undo_transform(in_p);
  unsigned int component = in_bid / 2;
  return full_position[component] == boundary_vertex_coordinates[in_bid];
}

std::vector<unsigned int> HSIESurface::get_vertices_for_boundary_id(BoundaryId in_boundary_id) {
  std::vector<unsigned int> vertices;
  for(auto it = Geometry.surface_meshes[b_id].begin_vertex(); it != Geometry.surface_meshes[b_id].end_vertex(); it++) {
    if(is_point_at_boundary(it->center(), in_boundary_id)) {
      vertices.push_back(it->index());
    }
  }
  vertices.shrink_to_fit();
  return vertices;
}

std::vector<unsigned int> HSIESurface::get_lines_for_boundary_id(BoundaryId in_boundary_id) {
  std::vector<unsigned int> edges;
  for(auto it = Geometry.surface_meshes[b_id].begin_active_face(); it != Geometry.surface_meshes[b_id].end_face(); it++) {
    if(is_point_at_boundary(it->center(), in_boundary_id)) {
      edges.push_back(it->index());
    }
  }
  edges.shrink_to_fit();
  return edges;
}

std::string HSIESurface::output_results(const dealii::Vector<ComplexNumber> & , std::string) { 
  return "";
}

DofCount HSIESurface::compute_n_locally_owned_dofs() {
    return 0;
}

DofCount HSIESurface::compute_n_locally_active_dofs() {
    return dof_counter;
}

void HSIESurface::finish_dof_index_initialization() {
  for(unsigned int surf = 0; surf < 6; surf+=2) {
    if(surf != b_id && !are_opposing_sites(surf, b_id)) {
      DofIndexVector dofs_in_global_numbering = Geometry.levels[level].surfaces[surf]->get_global_dof_indices_by_boundary_id(b_id);
      std::vector<InterfaceDofData> local_interface_data = get_dof_association_by_boundary_id(surf);
      DofIndexVector dofs_in_local_numbering(local_interface_data.size());
      for(unsigned int i = 0; i < local_interface_data.size(); i++) {
        dofs_in_local_numbering[i] = local_interface_data[i].index;
      }
      set_non_local_dof_indices(dofs_in_local_numbering, dofs_in_global_numbering);
    }
  }
}

void HSIESurface::determine_non_owned_dofs() {
  // TODO: This needs to be implemented, but I will do it once PML works.
}
