//
// Created by kraft on 23.07.19.
//

#include "RectangularMode.h"

RectangularMode::RectangularMode(double eps_in, double eps_out,
                                 double x_width_waveguide,
                                 double y_width_waveguide,
                                 double x_width_domain, double y_width_domain,
                                 double beta, unsigned int order)
    : FEM_order(order),
      fe(FE_Nedelec<3>(order), 2),
      triangulation(Triangulation<3>::MeshSmoothing(Triangulation<3>::none))
          .real(0),
      imag(3),
      dof_handler(triangulation),
{
  this->eps_in = eps_in;
  this->eps_out = eps_out;
  this->x_width_domain = x_width_domain;
  this->y_width_domain = y_width_domain;
  this->x_width_waveguide = x_width_waveguide;
  this->y_width_waveguide = y_width_waveguide;
  this->beta = beta;
};

void RectangularMode::run() {
  make_mesh();
  make_boundary_conditions();
  assemble_system();
  output_solution();
}

void RectangularMode::make_mesh() {
  Point<3> p1(-x_width_domain / 2.0, -y_width_domain / 2.0, -0.1);
  Point<3> p2(x_width_domain / 2.0, y_width_domain / 2.0, 0.1);
  std::vector<unsigned int> repetitions;
  repetitions.push_back(100);
  repetitions.push_back(100);
  repetitions.push_back(0);
  GridGenerator::fubdivided_hyper_rectangle(triangulation, repetitions, p1, p2,
                                            true);
  dof_handler.distribute_dofs(fe);
  DoFTools::make_periodicity_constraints(dof_handler, 4, 5, 2,
                                         periodic_constraints);
};

void RectangularMode::make_boundary_conditions() {
  ZeroFunction<3, double> zero(6);
  VectorTools::project_boundary_values_curl_conforming_l2(dof_handler, 0, zero,
                                                          0, constraints);
  VectorTools::project_boundary_values_curl_conforming_l2(dof_handler, 0, zero,
                                                          1, constraints);
  VectorTools::project_boundary_values_curl_conforming_l2(dof_handler, 0, zero,
                                                          2, constraints);
  VectorTools::project_boundary_values_curl_conforming_l2(dof_handler, 0, zero,
                                                          3, constraints);
  constraints.close();
};

void RectangularMode::assemble_system() {
  IndexSet repeated_Dof_indices = get_dofs_for_boundary_id(5);
};

void RectangularMode::output_solution(){

};

IndexSet RectangularMode::get_dofs_for_boundary_id(types::boundary_id in_bid) {
  IndexSet ret(dof_handler.n_dofs());
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
