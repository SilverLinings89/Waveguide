#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "NumericProblem.h"

#include <functional>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/vector_operation.h>
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
#include "../Helpers/PointSourceField.h"

NumericProblem::NumericProblem()
    :
      mesh_generator{},
      space_transformation{0},
    fe(GlobalParams.Nedelec_element_order),
      triangulation(Triangulation<3>::MeshSmoothing(Triangulation<3>::none)),
      dof_handler(triangulation){
  n_dofs = 0;
}

NumericProblem::~NumericProblem() {}

bool compareConstraintPairs(ConstraintPair v1, ConstraintPair v2) {
  return (v1.left < v2.left);
}

void NumericProblem::make_grid() {
  std::cout << "Make Grid." << std::endl;
  const unsigned int Cells_Per_Direction = 25;
  std::vector<unsigned int> repetitions;
  repetitions.push_back(Cells_Per_Direction);
  repetitions.push_back(Cells_Per_Direction);
  repetitions.push_back(Cells_Per_Direction);
  std::cout << "Geometry: ["<< Geometry.local_x_range.first << "," << Geometry.local_x_range.second 
    << "] x ["<< Geometry.local_y_range.first << "," << Geometry.local_y_range.second 
    << "] x ["<< Geometry.local_z_range.first << "," << Geometry.local_z_range.second << "]" << std::endl;
  Position lower(Geometry.local_x_range.first, Geometry.local_y_range.first,
      Geometry.local_z_range.first);
  Position upper(Geometry.local_x_range.second, Geometry.local_y_range.second,
      Geometry.local_z_range.second);
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions,
      lower, upper, true);
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
    dealii::DynamicSparsityPattern *in_pattern,
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

void NumericProblem::make_constraints(
    AffineConstraints<std::complex<double>> *in_constraints, unsigned int shift,
    unsigned int constraints_size) {
  std::cout << "Called Make constraints with shift = " << shift
      << " and  constraint_size" << constraints_size << std::endl;
  dealii::AffineConstraints<std::complex<double>> temp_cm;
  dealii::IndexSet is;
  is.set_size(constraints_size);
  is.add_range(0, constraints_size);
  temp_cm.reinit(is);
  PointSourceField psf;
  Position center(0.0, 0.0, 0.0);
  std::vector<unsigned int> ret(fe.dofs_per_line);
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    if (cell->point_inside(center)) {
      std::cout << "Am in cell. Its center is " << cell->center() << std::endl;
      for (unsigned int i = 0; i < 12; i++) {
        cell->line(i)->get_dof_indices(ret);
        temp_cm.add_line(ret[0] + shift);
        Position p0 = cell->line(i)->vertex(0);
        Position p1 = cell->line(i)->vertex(1);
        dealii::Vector<ComplexNumber> value;
        value.reinit(3);
        psf.vector_value(cell->line(i)->center(), value);
        ComplexNumber dof_value = { 0, 0 };
        for (unsigned int j = 0; j < 3; j++) {
          dof_value += value[j] * (p1[j] - p0[j]);
        }
        temp_cm.set_inhomogeneity(ret[0] + shift, dof_value);
        for (unsigned int j = 1; j < fe.dofs_per_line; j++) {
          temp_cm.add_line(ret[j] + shift);
          temp_cm.set_inhomogeneity(ret[j] + shift, ComplexNumber(0, 0));
        }
      }
    }
  }
  in_constraints->merge(temp_cm,
      dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins);
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

bool NumericProblem::get_orientation(const Position &vertex_1,
    const Position &vertex_2) {
  bool ret = false;
  double abs_max = -1.0;
  for (unsigned int i = 0; i < 3; i++) {
    double diff = vertex_1[i] - vertex_2[i];
    if (std::abs(diff) > abs_max) {
      ret = diff > 0;
      abs_max = std::abs(diff);
    }
  }
  return ret;
}

std::vector<DofIndexAndOrientationAndPosition> NumericProblem::get_surface_dof_vector_for_boundary_id(
    unsigned int b_id) {
  std::vector<DofIndexAndOrientationAndPosition> ret;
  for (auto cell : dof_handler.active_cell_iterators()) {
    if (cell->at_boundary(b_id)) {
      bool found_one = false;
      for (unsigned int face = 0; face < 6; face++) {
        if (cell->face(face)->boundary_id() == b_id && found_one) {
          std::cout << "Error in get_surface_dof_vector_for_boundary_id"
              << std::endl;
        }
        if (cell->face(face)->boundary_id() == b_id) {
          found_one = true;
          std::vector<DofNumber> face_dofs_indices(fe.dofs_per_face);
          cell->face(face)->get_dof_indices(face_dofs_indices);
          std::vector<DofIndexAndOrientationAndPosition> cell_dofs_and_orientations_and_points(
              fe.dofs_per_face);
          for (unsigned int i = 0; i < fe.dofs_per_face; i++) {
            cell_dofs_and_orientations_and_points[i].index =
                face_dofs_indices[i];
            cell_dofs_and_orientations_and_points[i].position =
                cell->face(face)->center();
            cell_dofs_and_orientations_and_points[i].orientation =
                get_orientation(cell->face(face)->vertex(0),
                    cell->face(face)->vertex(1));
          }
          for (unsigned int i = 0; i < 4; i++) {
            std::vector<DofNumber> line_dofs(fe.dofs_per_line);
            cell->face(face)->line(i)->get_dof_indices(line_dofs);
            for (unsigned int j = 0; j < fe.dofs_per_line; j++) {
              for (unsigned int outer_index = 0; outer_index < fe.dofs_per_face;
                  outer_index++) {
                if (face_dofs_indices[outer_index] == line_dofs[j]) {
                  cell_dofs_and_orientations_and_points[outer_index].orientation =
                      get_orientation(cell->face(face)->line(i)->vertex(0),
                          cell->face(face)->line(i)->vertex(1));
                  cell_dofs_and_orientations_and_points[outer_index].position =
                      cell->face(face)->line(i)->center();
                }
              }
            }
          }
          for (unsigned int i = 0;
              i < cell_dofs_and_orientations_and_points.size(); i++) {
            ret.push_back(cell_dofs_and_orientations_and_points[i]);
          }
        }
      }
    }
  }
  std::sort(ret.begin(), ret.end(), compareDofBaseDataAndOrientation);
  std::vector<DofIndexAndOrientationAndPosition> removed_doubles;
  if (ret.size() == 0)
    return ret;
  removed_doubles.push_back(ret[0]);
  for (unsigned int i = 1; i < ret.size(); i++) {
    if (ret[i].index != ret[i - 1].index) {
      removed_doubles.push_back(ret[i]);
    }
  }
  return removed_doubles;
}

struct CellwiseAssemblyData {
  QGauss<3> quadrature_formula; 
  FEValues<3> fe_values;
  std::vector<Point<3>> quadrature_points;
  const unsigned int dofs_per_cell;
  const unsigned int n_q_points;
  FullMatrix<ComplexNumber> cell_matrix_real;
  const double eps_in;
  const double eps_out;
  const double mu_zero;
  Vector<ComplexNumber> cell_rhs;
  MaterialTensor transformation;
  MaterialTensor epsilon;
  MaterialTensor mu;
  std::vector<DofNumber> local_dof_indices;
  DofHandler3D::active_cell_iterator cell;
  DofHandler3D::active_cell_iterator end_cell;
  const Position bounded_cell;
  const FEValuesExtractors::Vector fe_field;
  CellwiseAssemblyData(dealii::FE_NedelecSZ<3> * fe, DofHandler3D * dof_handler):
  quadrature_formula(2),
  fe_values(*fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values |
                            update_quadrature_points),
  dofs_per_cell(fe->dofs_per_cell),
  n_q_points(quadrature_formula.size()),
  cell_matrix_real(dofs_per_cell,dofs_per_cell),
  eps_in(GlobalParams.Epsilon_R_in_waveguide),
  eps_out(GlobalParams.Epsilon_R_outside_waveguide),
  mu_zero(1.0),
  cell_rhs(dofs_per_cell),
  local_dof_indices(dofs_per_cell),
  bounded_cell(0.0,0.0,0.0),
  fe_field(0)
  {
    cell_rhs = 0;
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        if (i == j) {
          transformation[i][j] = ComplexNumber(1, 0);
        } else {
          transformation[i][j] = ComplexNumber(0, 0);
        }
      }
    }
    cell = dof_handler->begin_active();
    end_cell = dof_handler->end();
  };

  void prepare_for_current_q_index(unsigned int q_index) {
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

        cell_matrix_real[i][j] += (mu * I_Curl) * Conjugate_Vector(J_Curl)
            * JxW
            - ((epsilon * I_Val) * Conjugate_Vector(J_Val)) * JxW
                * GlobalParams.Omega * GlobalParams.Omega;
      }
    }
  }

  /**
   * This function calculates the complex conjugate of every vector entry and
   * returns the result in a copy. Similar to Conjugate_Tensor(Tensor<2,3,
   * std::complex<double>> input) this function does not operate in place - it
   * operates on a copy and hence returns a new object.
   */
  Tensor<1, 3, std::complex<double>> Conjugate_Vector(
      Tensor<1, 3, std::complex<double>> input) {
    Tensor<1, 3, std::complex<double>> ret;

    for (int i = 0; i < 3; i++) {
      ret[i].real(input[i].real());
      ret[i].imag(-input[i].imag());
    }
    return ret;
  }

};

void NumericProblem::assemble_system(unsigned int shift,
    dealii::AffineConstraints<ComplexNumber> * constraints,
    dealii::PETScWrappers::SparseMatrix *matrix,
    NumericVectorDistributed *rhs) {
  reinit_rhs();

  CellwiseAssemblyData cell_data(&fe, &dof_handler);

  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    if (!cell_data.cell->point_inside(cell_data.bounded_cell)) {
      cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
      for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
        cell_data.local_dof_indices[i] += shift;
      }
      cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
      cell_data.fe_values.reinit(cell_data.cell);
      cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
      std::vector<types::global_dof_index> input_dofs(fe.dofs_per_line);
      IndexSet input_dofs_local_set(fe.dofs_per_cell);
      std::vector<Position> input_dof_centers(fe.dofs_per_cell);
      std::vector<Tensor<1, 3, double>> input_dof_dirs(fe.dofs_per_cell);

      cell_data.cell_matrix_real = 0;
      for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
        cell_data.prepare_for_current_q_index(q_index);
        
        constraints->distribute_local_to_global(cell_data.cell_matrix_real, cell_data.cell_rhs,
            cell_data.local_dof_indices,*matrix, *rhs, false);
      }
    }
  }
  matrix->compress(dealii::VectorOperation::add);
  write_matrix_and_rhs_metrics(matrix, rhs);
  
  deallog << "Distributing solution done." << std::endl;
}

void NumericProblem::assemble_system(unsigned int shift,
    dealii::AffineConstraints<ComplexNumber> * constraints,
    dealii::PETScWrappers::MPI::SparseMatrix *matrix,
    NumericVectorDistributed *rhs) {
  reinit_rhs();

  CellwiseAssemblyData cell_data(&fe, &dof_handler);

  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    if (!cell_data.cell->point_inside(cell_data.bounded_cell)) {
      cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
      for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
        cell_data.local_dof_indices[i] += shift;
      }
      cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
      cell_data.fe_values.reinit(cell_data.cell);
      cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
      std::vector<types::global_dof_index> input_dofs(fe.dofs_per_line);
      IndexSet input_dofs_local_set(fe.dofs_per_cell);
      std::vector<Position> input_dof_centers(fe.dofs_per_cell);
      std::vector<Tensor<1, 3, double>> input_dof_dirs(fe.dofs_per_cell);

      cell_data.cell_matrix_real = 0;
      for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
        cell_data.prepare_for_current_q_index(q_index);
        
        constraints->distribute_local_to_global(cell_data.cell_matrix_real, cell_data.cell_rhs,
            cell_data.local_dof_indices,*matrix, *rhs, false);
      }
    }
  }
  matrix->compress(dealii::VectorOperation::add);
  write_matrix_and_rhs_metrics(matrix, rhs);
  
  deallog << "Distributing solution done." << std::endl;
}

void NumericProblem::write_matrix_and_rhs_metrics(dealii::PETScWrappers::MatrixBase * matrix, NumericVectorDistributed *rhs) {
  std::cout << "System Matrix l_infty norm: " << matrix->linfty_norm()
      << std::endl;
  std::cout << "System Matrix l_1 norm: " << matrix->l1_norm() << std::endl;

  std::cout << "Assembling done. L2-Norm of RHS: " << rhs->l2_norm()
      << std::endl;
}

#endif
