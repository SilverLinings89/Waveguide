// Copyright 2018 Pascal Kraft
#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "NumericProblem.h"

#include <deal.II/base/std_cxx11/bind.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <algorithm>
#include <ctime>
#include <string>
#include <utility>
#include <vector>
#include "../Core/Types.h"
#include "../Helpers/staticfunctions.h"
#include "../MeshGenerators/SquareMeshGenerator.h"
#include "SolutionWeight.h"
#include "./GlobalObjects.h"
#include "../SpaceTransformations/HomogenousTransformationRectangular.h"

NumericProblem::NumericProblem()
    :
    fe(GlobalParams.So_ElementOrder),
      triangulation(Triangulation<3>::MeshSmoothing(Triangulation<3>::none)),
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
  // mg->prepare_triangulation(&triangulation);
  std::cout << "Make Grid." << std::endl;
  const unsigned int Cells_Per_Direction = 25;
  std::vector<unsigned int> repetitions;
  repetitions.push_back(Cells_Per_Direction);
  repetitions.push_back(Cells_Per_Direction);
  repetitions.push_back(Cells_Per_Direction);
  dealii::Point<3, double> left(-1, -1, -1);
  dealii::Point<3, double> right(1, 1, 1);
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions,
      left, right, true);
  dof_handler.distribute_dofs(fe);
  SortDofsDownstream();
  n_dofs = dof_handler.n_dofs();
  std::cout << "Mesh Preparation finished. System has " << n_dofs
      << " degrees of freedom." << std::endl;
}

bool compareIndexCenterPairs(std::pair<int, double> c1,
                             std::pair<int, double> c2) {
  return c1.second < c2.second;
}

std::vector<unsigned int> NumericProblem::dofs_for_cell_around_point(
    dealii::Point<3> &in_p) {
  std::vector<unsigned int> ret(fe.dofs_per_cell);
  std::cout << "DOFS per Cell: " << fe.dofs_per_cell << std::endl;
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    if (cell->point_inside(in_p)) {
      cell->get_dof_indices(ret);
      return ret;
    }
  }
  return ret;
}

void NumericProblem::make_sparsity_pattern(
    dealii::SparsityPattern *in_pattern,
    unsigned int shift) {
  reinit_rhs();

  std::vector<Point<3>> quadrature_points;
  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();

  for (; cell != endc; ++cell) {
    cell->get_dof_indices(local_dof_indices);

    std::vector<types::global_dof_index> input_dofs(fe.dofs_per_line);
    IndexSet input_dofs_local_set(fe.dofs_per_cell);
    std::vector<Point<3, double>> input_dof_centers(fe.dofs_per_cell);
    std::vector<Tensor<1, 3, double>> input_dof_dirs(fe.dofs_per_cell);

    for (unsigned int i = 0; i < dofs_per_cell; i++) {
      for (unsigned int j = 0; j < dofs_per_cell; j++) {
        in_pattern->add(local_dof_indices[i] + shift,
            local_dof_indices[j] + shift);
      }
    }
  }
}

void NumericProblem::make_constraints() {
  dealii::Point<3> center(0.0, 0.0, 0.0);
  std::vector<unsigned int> restrained_dofs = dofs_for_cell_around_point(
      center);
  for (unsigned int i = 0; i < restrained_dofs.size(); i++) {
    cm.add_line(restrained_dofs[i]);
    cm.set_inhomogeneity(restrained_dofs[i], std::complex<double>(1.0, -1.0));
  }
  cm.close();
}

void NumericProblem::SortDofsDownstream() {
  std::vector<std::pair<int, Point<3, double>>> current;

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
            current.emplace_back(local_line_dofs[k],
                (cell->face(i))->line(j)->center());
          }
          lines_touched.push_back(cell->face(i)->line(j)->index());
        }
      }
    }
  }
  std::sort(current.begin(), current.end(), compareDofBaseData);
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
  make_constraints();

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

  deallog << "Done." << std::endl;
  deallog.pop();
}

void NumericProblem::reinit_rhs() {
  // system_rhs.reinit(dof_handler.n_dofs());
}

void NumericProblem::reinit_solution() {

}

std::vector<unsigned int> NumericProblem::get_surface_dof_vector_for_boundary_id(
    unsigned int b_id) {
  std::vector<unsigned int> ret;
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  std::vector<bool> dof_vector;
  std::set<unsigned int> b_ids;
  b_ids.insert(b_id);
  dealii::ComponentMask cm;
  dealii::DoFTools::extract_boundary_dofs(dof_handler,
      cm, dof_vector, b_ids);
  for (unsigned int i = 0; i < dof_vector.size(); i++) {
    if (dof_vector[i]) {
      ret.push_back(i);
    }
  }
  return ret;
}

void NumericProblem::assemble_system(unsigned int shift,
    dealii::SparseMatrix<std::complex<double>> *matrix,
    dealii::Vector<std::complex<double>> *rhs) {
  reinit_rhs();

  QGauss<3> quadrature_formula(2);
  FEValues<3> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values |
                            update_quadrature_points);
  std::vector<Point<3>> quadrature_points;
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  deallog << "Starting Assemblation process" << std::endl;

  FullMatrix<std::complex<double>> cell_matrix_real(dofs_per_cell,
      dofs_per_cell);

  double e_temp = 1.0;
  double mu_temp = 1.0;

  const double eps_in = GlobalParams.M_W_epsilonin * e_temp;
  const double eps_out = GlobalParams.M_W_epsilonout * e_temp;
  const double mu_zero = mu_temp;
  Vector<std::complex<double>> cell_rhs(dofs_per_cell);
  cell_rhs = 0;
  Tensor<2, 3, std::complex<double>> transformation;
  Tensor<2, 3, std::complex<double>> epsilon;
  Tensor<2, 3, std::complex<double>> mu;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      if (i == j) {
        transformation[i][j] = std::complex<double>(1, 0);
      } else {
        transformation[i][j] = std::complex<double>(0, 0);
      }
    }
  }
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();

  deallog << "Assembly loop." << std::endl;
  const dealii::Point<3> bounded_cell(0.0, 0.0, 0.0);
  const FEValuesExtractors::Vector fe_field(0);
  for (; cell != endc; ++cell) {
    if (true) {
      // if (!cell->point_inside(bounded_cell)) {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < local_dof_indices.size(); i++) {
        local_dof_indices[i] += shift;
      }
      cell_rhs.reinit(dofs_per_cell, false);
      fe_values.reinit(cell);
      quadrature_points = fe_values.get_quadrature_points();
      std::vector<types::global_dof_index> input_dofs(fe.dofs_per_line);
      IndexSet input_dofs_local_set(fe.dofs_per_cell);
      std::vector<Point<3, double>> input_dof_centers(fe.dofs_per_cell);
      std::vector<Tensor<1, 3, double>> input_dof_dirs(fe.dofs_per_cell);

      cell_matrix_real = 0;
      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {

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
          I_Curl = fe_values[fe_field].curl(i, q_index);
          I_Val = fe_values[fe_field].value(i, q_index);

          for (unsigned int j = 0; j < dofs_per_cell; j++) {
            Tensor<1, 3, std::complex<double>> J_Curl;
            Tensor<1, 3, std::complex<double>> J_Val;
            J_Curl = fe_values[fe_field].curl(j, q_index);
            J_Val = fe_values[fe_field].value(j, q_index);

            std::complex<double> x = (mu * I_Curl) * Conjugate_Vector(J_Curl)
                * JxW
                - ((epsilon * I_Val) * Conjugate_Vector(J_Val)) * JxW
                    * GlobalParams.C_omega * GlobalParams.C_omega;
            cell_matrix_real[i][j] += x;

          }
        }

        cm.distribute_local_to_global(cell_matrix_real, cell_rhs,
            local_dof_indices, *matrix, *rhs, false);
        // for (unsigned int i = 0; i < local_dof_indices.size(); i++) {
        //  for (unsigned int j = 0; j < local_dof_indices.size(); j++) {
        //    matrix->add(local_dof_indices[i], local_dof_indices[j],
        //        cell_matrix_real[i][j]);
        //  }
        // }
      }
    }
  }

  std::cout << "System Matrix l_infty norm: " << matrix->linfty_norm()
      << std::endl;
  std::cout << "System Matrix l_1 norm: " << matrix->l1_norm() << std::endl;

  std::cout << "Assembling done. L2-Norm of RHS: " << rhs->l2_norm()
      << std::endl;

  // system_matrix.compress(VectorOperation::add);
  // system_rhs.compress(VectorOperation::add);

  deallog << "Distributing solution done." << std::endl;
}


#endif
