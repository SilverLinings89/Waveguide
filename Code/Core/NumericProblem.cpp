// Copyright 2018 Pascal Kraft
#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "NumericProblem.h"

#include <deal.II/base/std_cxx11/bind.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_matrix_array.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <algorithm>
#include <ctime>
#include <string>
#include <utility>
#include <vector>
#include "../Helpers/staticfunctions.h"
#include "../MeshGenerators/SquareMeshGenerator.h"
#include "PreconditionerSweeping.h"
#include "SolutionWeight.h"
#include "./GlobalObjects.h"
#include "../SpaceTransformations/HomogenousTransformationRectangular.h"

NumericProblem::NumericProblem()
    : fe(FE_Nedelec<3>(GlobalParams.So_ElementOrder), 2),
      triangulation(Triangulation<3>::MeshSmoothing(Triangulation<3>::none)),
      real(0),
      imag(3),
      dof_handler(triangulation) {
  mg = new SquareMeshGenerator();
  st = new HomogenousTransformationRectangular(0);
  n_dofs = 0;
}

NumericProblem::~NumericProblem() {}

bool compareConstraintPairs(ConstraintPair v1, ConstraintPair v2) {
  return (v1.left < v2.left);
}

Tensor<1, 3, std::complex<double>> NumericProblem::Conjugate_Vector(
    Tensor<1, 3, std::complex<double>> input) {
  Tensor<1, 3, std::complex<double>> ret;

  for (int i = 0; i < 3; i++) {
    ret[i].real(input[i].real());
    ret[i].imag(-input[i].imag());
  }
  return ret;
}

void NumericProblem::make_grid() {
  mg->prepare_triangulation(&triangulation);
  dof_handler.distribute_dofs(fe);
  SortDofsDownstream();
  n_dofs = dof_handler.n_dofs();
}

bool compareIndexCenterPairs(std::pair<int, double> c1,
                             std::pair<int, double> c2) {
  return c1.second < c2.second;
}

void NumericProblem::SortDofsDownstream() {
  std::vector<std::pair<int, double>> current;

  std::vector<unsigned int> lines_touched;
  std::vector<types::global_dof_index> local_line_dofs(fe.dofs_per_line);
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
      for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
        if (!(std::find(lines_touched.begin(), lines_touched.end(),
            cell->face(i)->line(j)->index()) !=
              lines_touched.end())) {
          ((cell->face(i))->line(j))->get_dof_indices(local_line_dofs);
          for (unsigned k = 0; k < local_line_dofs.size(); k++) {
            current.push_back(std::pair<int, double>(
                local_line_dofs[k], (cell->face(i))->line(j)->center()[2]));
          }
          lines_touched.push_back(cell->face(i)->line(j)->index());
        }
      }
    }
  }
  std::sort(current.begin(), current.end(), compareIndexCenterPairs);
  std::vector<unsigned int> new_numbering;
  new_numbering.resize(current.size());
  for (unsigned int i = 0; i < current.size(); i++) {
    new_numbering[current[i].first] = i;
  }
  dof_handler.renumber_dofs(new_numbering);
}

void NumericProblem::setup_system() {
  deallog.push("setup_system");

  reinit_all();

  deallog.pop();
}

void NumericProblem::reinit_all() {
  deallog.push("reinit_all");

  deallog << "reinitializing right-hand side" << std::endl;
  reinit_rhs();

  deallog << "reinitializing solutiuon" << std::endl;
  reinit_solution();

  deallog << "reinitializing system matrix" << std::endl;
  reinit_systemmatrix();

  deallog << "Done" << std::endl;
  deallog.pop();
}

void NumericProblem::reinit_systemmatrix() {
  deallog.push("reinit_systemmatrix");
  deallog << "Initializing sparsity pattern and system matrix." << std::endl;
  DynamicSparsityPattern dsp (dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  DoFTools::make_hanging_node_constraints(dof_handler, cm);
  cm.condense(dsp);
  final_sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(final_sparsity_pattern);
  deallog << "Done." << std::endl;
  deallog.pop();
}

void NumericProblem::reinit_rhs() {

}

void NumericProblem::reinit_solution() {

}

void NumericProblem::assemble_system() {
  reinit_rhs();

  QGauss<3> quadrature_formula(2);
  const FEValuesExtractors::Vector real(0);
  const FEValuesExtractors::Vector imag(3);
  FEValues<3> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values |
                            update_quadrature_points);
  std::vector<Point<3>> quadrature_points;
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  deallog << "Starting Assemblation process" << std::endl;

  FullMatrix<double> cell_matrix_real(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_prec_odd(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_prec_even(dofs_per_cell, dofs_per_cell);

  double e_temp = 1.0;
  if (!GlobalParams.C_AllOne) {
    e_temp *= GlobalParams.C_Epsilon;
  }
  double mu_temp = 1.0;
  if (!GlobalParams.C_AllOne) {
    mu_temp *= GlobalParams.C_Mu;
  }

  const double eps_in = GlobalParams.M_W_epsilonin * e_temp;
  const double eps_out = GlobalParams.M_W_epsilonout * e_temp;
  const double mu_zero = mu_temp;

  Vector<double> cell_rhs(dofs_per_cell);
  cell_rhs = 0;
  Tensor<2, 3, std::complex<double>> transformation;
  Tensor<2, 3, std::complex<double>> epsilon;
  Tensor<2, 3, std::complex<double>> mu;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::complex<double> k_a_sqr(GlobalParams.C_omega,
                               GlobalParams.So_PreconditionerDampening);
  k_a_sqr = k_a_sqr * k_a_sqr;

  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();

  for (; cell != endc; ++cell) {
    cell->get_dof_indices(local_dof_indices);
    cell_rhs.reinit(dofs_per_cell, false);
    fe_values.reinit(cell);
    quadrature_points = fe_values.get_quadrature_points();
    std::vector<types::global_dof_index> input_dofs(fe.dofs_per_line);
    IndexSet input_dofs_local_set(fe.dofs_per_cell);
    std::vector<Point<3, double>> input_dof_centers(fe.dofs_per_cell);
    std::vector<Tensor<1, 3, double>> input_dof_dirs(fe.dofs_per_cell);

    cell_matrix_real = 0;
    cell_matrix_prec_odd = 0;
    cell_matrix_prec_even = 0;
    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
      transformation = st->get_Tensor(quadrature_points[q_index]);
      mu = invert(transformation) / mu_zero;

      if (Geometry.math_coordinate_in_waveguide(quadrature_points[q_index])) {
        epsilon = transformation * eps_in;
      } else {
        epsilon = transformation * eps_out;
      }

      const double JxW = fe_values.JxW(q_index);
      for (unsigned int i = 0; i < dofs_per_cell; i++) {
        Tensor<1, 3, std::complex<double>> I_Curl;
        Tensor<1, 3, std::complex<double>> I_Val;
        for (int k = 0; k < 3; k++) {
          I_Curl[k].imag(fe_values[imag].curl(i, q_index)[k]);
          I_Curl[k].real(fe_values[real].curl(i, q_index)[k]);
          I_Val[k].imag(fe_values[imag].value(i, q_index)[k]);
          I_Val[k].real(fe_values[real].value(i, q_index)[k]);
        }

        for (unsigned int j = 0; j < dofs_per_cell; j++) {
          Tensor<1, 3, std::complex<double>> J_Curl;
          Tensor<1, 3, std::complex<double>> J_Val;
          for (int k = 0; k < 3; k++) {
            J_Curl[k].imag(fe_values[imag].curl(j, q_index)[k]);
            J_Curl[k].real(fe_values[real].curl(j, q_index)[k]);
            J_Val[k].imag(fe_values[imag].value(j, q_index)[k]);
            J_Val[k].real(fe_values[real].value(j, q_index)[k]);
          }

          std::complex<double> x =
              (mu * I_Curl) * Conjugate_Vector(J_Curl) * JxW -
              ((epsilon * I_Val) * Conjugate_Vector(J_Val)) * JxW *
                  GlobalParams.C_omega * GlobalParams.C_omega;
          cell_matrix_real[i][j] += x.real();

        }
      }

      cm.distribute_local_to_global(cell_matrix_real, cell_rhs,
                                    local_dof_indices, system_matrix,
                                    system_rhs, false);
    }
  }


  deallog << "Assembling done. L2-Norm of RHS: " << system_rhs.l2_norm()
          << std::endl;

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  deallog << "Distributing solution done." << std::endl;
}


#endif
