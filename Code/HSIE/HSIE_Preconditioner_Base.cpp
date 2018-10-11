#ifndef HSIEPRECONDITIONERBASE_CPP
#define HSIEPRECONDITIONERBASE_CPP
#include "HSIE_Preconditioner_Base.h"
#include <complex.h>
#include <deal.II/base/quadrature_lib.h>
#include <vector>

/*
 * HSIE_Preconditioner_Base.cpp
 *
 *  Created on: Jul 11, 2018
 *      Author: kraft
 */

/**
 * Implement:
 * - a(u,v)
 * - g(j,k)
 * - A_C,T
 * - (28)
 *
 */

const unsigned int S1IntegrationPoints = 100;
const double k0 = 0.5;

dealii::Point<3, std::complex<double>> base_fun(int i, int j,
                                                dealii::Point<2, double> x) {
  dealii::Point<3, std::complex<double>> ret =
      new dealii::Point<3, std::complex<double>>();
  ret[0] = new std::complex<double>(0, 0);
  ret[1] = new std::complex<double>(0, 0);
  ret[2] = new std::complex<double>(0, 0);
  return ret;
}

std::complex<double>* a(int in_i, int in_j) {
  dealii::Point<3, std::complex<double>> ret =
      new dealii::Point<3, std::complex<double>>();
  for (unsigned int i = 0; i < 3; i++) {
    ret[i] = new std::complex<double>(0, 0);
  }
  for (unsigned int i = 0; i < S1IntegrationPoints; i++) {
    double x = std::sin(2 * 3.14159265359 * ((double)i) /
                        ((double)S1IntegrationPoints));
    double y = std::cos(2 * 3.14159265359 * ((double)i) /
                        ((double)S1IntegrationPoints));
    dealii::Point<2, double> p1(x, y);
    dealii::Point<2, double> p2(x, -y);
    ret[0] += base_fun(in_i, 0, p1) * base_fun(in_j, 0, p2);
  }
  ret /= (double)S1IntegrationPoints;
  ret *= k0 * std::complex<double>(0, 1) / 3.14159265359;
  return ret;
}

int count_HSIE_dofs(dealii::parallel::distributed::Triangulation<3>* tria,
                    unsigned int degree, double in_z) {
  dealii::IndexSet interfacevertices(tria->n_vertices());
  dealii::IndexSet interfaceedges(tria->n_active_lines());
  dealii::parallel::distributed::Triangulation<3>::active_cell_iterator
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

template <int hsie_order>
unsigned int HSIEPreconditionerBase<hsie_order>::compute_number_of_dofs() {}

template <int hsie_order>
std::complex<double> HSIEPreconditionerBase<hsie_order>::a(
    HSIE_Dof_Type<hsie_order> u, HSIE_Dof_Type<hsie_order> v,
    bool use_curl_fomulation, dealii::Point<2, double> x) {}

template <int hsie_order>
std::complex<double> HSIEPreconditionerBase<hsie_order>::A(
    HSIE_Dof_Type<hsie_order> u, HSIE_Dof_Type<hsie_order> v,
    dealii::Tensor<2, 3, double> G, bool use_curl_fomulation) {
  dealii::QGauss<2> quadrature_formula(2);
  std::complex<double> ret(0,0);
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

#endif
