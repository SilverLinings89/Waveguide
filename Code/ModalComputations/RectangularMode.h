//
// Created by kraft on 23.07.19.
//

#ifndef WAVEGUIDEPROBLEM_RECTANGULARMODE_H
#define WAVEGUIDEPROBLEM_RECTANGULARMODE_H

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_system.h>

using namespace dealii;

class RectangularMode {
  double eps_in, eps_out;
  double beta;
  double x_width_waveguide, y_width_waveguide;
  double x_width_domain, y_width_domain;
  const int FEM_order;

  FESystem<3> fe;
  FEValuesExtractors::Vector real, imag;
  AffineConstraints<ComplexNumber> constraints, periodic_constraints;
  Triangulation<3> triangulation;
  DoFHandler<3> dof_handler;

  SparseMatrix<ComplexNumber> system_matrix;

  RectangularMode(double, double, double, double, double, double, double,
                  unsigned int);
  void assemble_system();
  void make_mesh();
  void make_boundary_conditions();
  void output_solution();
  void run();
  IndexSet get_dofs_for_boundary_id(types::boundary_id);
};

#endif  // WAVEGUIDEPROBLEM_RECTANGULARMODE_H
