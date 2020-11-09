#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "NumericProblem.h"

#include <functional>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>
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
#include <set>
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
  local_constraints_made = false;
}

NumericProblem::~NumericProblem() {}

bool compareConstraintPairs(ConstraintPair v1, ConstraintPair v2) {
  return (v1.left < v2.left);
}

void NumericProblem::make_grid() {
  std::cout << "Make Grid." << std::endl;
  // const unsigned int Cells_Per_Direction = 3;
  std::vector<unsigned int> repetitions;
  repetitions.push_back(GlobalParams.Cells_in_x);
  repetitions.push_back(GlobalParams.Cells_in_y);
  repetitions.push_back(GlobalParams.Cells_in_z);
  std::cout << "Cells: " << GlobalParams.Cells_in_x << " x " << GlobalParams.Cells_in_y << " x " << GlobalParams.Cells_in_z <<std::endl;
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
    Position &in_p) {
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
    unsigned int shift,
    dealii::AffineConstraints<ComplexNumber> *constraints) {
  
  auto end = dof_handler.end();
  std::vector<DofNumber> cell_dof_indices(fe.dofs_per_cell);
  for(auto cell = dof_handler.begin_active(); cell != end; cell++) {
    cell->get_dof_indices(cell_dof_indices);
    if(shift!=0) {
      for(unsigned int i = 0; i < fe.dofs_per_cell; i++) {
        cell_dof_indices[i] += shift; 
      }
    }
    constraints->add_entries_local_to_global(cell_dof_indices, *in_pattern);
  }
}

auto NumericProblem::get_central_cells(double radius) -> std::set<std::string> {
  std::set<std::string> ret;
  for(auto it = dof_handler.begin_active(); it != dof_handler.end(); it++) {
    if( std::abs(it->center()[0]) < radius &&  std::abs(it->center()[1]) < radius && std::abs(it->center()[2]) < radius  ) {
      ret.insert(it->id().to_string());
    }
  }
  return ret;
}

auto NumericProblem::get_outer_constrained_faces() -> std::set<unsigned int> {
  std::set<unsigned int> constrained_faces;
  for(auto it = dof_handler.begin_active(); it != dof_handler.end(); it++) {
    if(constrained_cells.contains(it->id().to_string())) {
      for(unsigned int i = 0; i < 6; i++) {
        if(constrained_faces.contains(it->face_index(i))) {
          constrained_faces.erase(it->face_index(i));
        } else {
          constrained_faces.insert(it->face_index(i));
        }
      }
    }
  }
  return constrained_faces;
}

void NumericProblem::make_constraints() {
  PointSourceField psf;
  double inner_cell_radius = 1.0/11.0;
  psf.set_cell_diameter(inner_cell_radius/2.0);
  constrained_cells = get_central_cells(inner_cell_radius+0.0001);
  outer_constrained_faces = get_outer_constrained_faces();
  std::cout << "Constrained cell count: " << constrained_cells.size() << std::endl;
  std::vector<DofCount> local_line_indices(fe.dofs_per_line);
  std::vector<DofCount> local_face_indices(fe.dofs_per_face);
  local_dof_indices.set_size(n_dofs);
  local_dof_indices.add_range(0, n_dofs);
  local_constraints.reinit(local_dof_indices);

  // Constrain the outer ones correctly
  for(auto it = dof_handler.begin_active(); it != dof_handler.end(); it++) {
    if(constrained_cells.contains(it->id().to_string())) {
      for (unsigned int face = 0; face < dealii::GeometryInfo<3>::faces_per_cell; face++) {
        it->face(face)->get_dof_indices(local_face_indices);
        if(outer_constrained_faces.contains(it->face_index(face))) {
          for(unsigned int line = 0; line < dealii::GeometryInfo<3>::lines_per_face; line++) {
            it->face(face)->line(line)->get_dof_indices(local_line_indices);
            Position p0 = it->face(face)->line(line)->vertex(0);
            Position p1 = it->face(face)->line(line)->vertex(1);
            NumericVectorLocal value(3);
            psf.vector_value(it->face(face)->line(line)->center(), value);
            ComplexNumber dof_value = { 0, 0 };
            for (unsigned int j = 0; j < 3; j++) {
              dof_value += value[j] * (p0[j] - p1[j]);
            }
            if(std::abs(dof_value) >= 1000.0) {
              dof_value = 1000.0 * (dof_value / std::abs(dof_value));
            }
            local_constraints.add_line(local_line_indices[0]);
            local_constraints.set_inhomogeneity(local_line_indices[0], dof_value);
            for(unsigned int j = 1; j < fe.dofs_per_line; j++) {
              local_constraints.add_line(local_line_indices[j]);
              local_constraints.set_inhomogeneity(local_line_indices[j], ComplexNumber(0,0));
            }
          }
        }
      }
    }
  }
  local_constraints_made = true;
}

void NumericProblem::make_constraints(
    AffineConstraints<ComplexNumber> *in_constraints, unsigned int shift, IndexSet local_constraints_indices) {
  if(!local_constraints_made) {
    make_constraints();
  }
  dealii::AffineConstraints<ComplexNumber> temp(local_constraints_indices);
  for(auto index : local_dof_indices) {
    if(local_constraints.is_constrained(index)){
      temp.add_line(index + shift);
      for(auto entry : *local_constraints.get_constraint_entries(index)) {
        temp.add_entry(index+shift, entry.first + shift, entry.second);
      }
      if(local_constraints.is_inhomogeneously_constrained(index)) {
        temp.set_inhomogeneity(index+shift, local_constraints.get_inhomogeneity(index));
      }
    }
  }
  in_constraints->merge(temp, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
}

void NumericProblem::SortDofsDownstream() {
  std::cout << "Start Dof Sorting" << std::endl;
  triangulation.clear_user_flags();
  std::vector<std::pair<DofNumber, Position>> current;
  std::vector<types::global_dof_index> local_line_dofs(fe.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe.dofs_per_face);
  std::set<DofNumber> face_set;
  std::vector<DofNumber> local_cell_dofs(fe.dofs_per_cell);
  std::set<DofNumber> cell_set;
  print_info("Sorting", "Mark 1");
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    cell->get_dof_indices(local_cell_dofs);
    cell_set.clear();
    cell_set.insert(local_cell_dofs.begin(), local_cell_dofs.end());
    for(unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; face++) {
      cell->face(face)->get_dof_indices(local_face_dofs);
      face_set.clear();
      face_set.insert(local_face_dofs.begin(), local_face_dofs.end());
      for(auto firstit : face_set) {
        cell_set.erase(firstit);
      }
      for(unsigned int line = 0; line < GeometryInfo<3>::lines_per_face; line++) {
        cell->face(face)->line(line)->get_dof_indices(local_line_dofs);
        line_set.clear();
        line_set.insert(local_line_dofs.begin(), local_line_dofs.end());
        for(auto firstit : line_set) {
          face_set.erase(firstit);
        }
        if(!cell->face(face)->line(line)->user_flag_set()){
          for(auto dof: line_set) {
            current.emplace_back(dof, cell->face(face)->line(line)->center());
          }
          cell->face(face)->line(line)->set_user_flag();
        }
      }
      if(!cell->face(face)->user_flag_set()){
        for(auto dof: face_set) {
          current.emplace_back(dof, cell->face(face)->center());
        }
        cell->face(face)->set_user_flag();
      }
    }
    for(auto dof: cell_set) {
      current.emplace_back(dof, cell->center());
    }
  }
  print_info("Sorting", "Mark 2");
  std::sort(current.begin(), current.end(), compareDofBaseData);
  std::vector<unsigned int> new_numbering;
  new_numbering.resize(current.size());
  for (unsigned int i = 0; i < current.size(); i++) {
    new_numbering[current[i].first] = i;
  }
  print_info("Sorting", "Mark 3");
  dof_handler.renumber_dofs(new_numbering);
  std::cout << "End Dof Sorting" << std::endl;
}

std::vector<DofIndexAndOrientationAndPosition> NumericProblem::get_surface_dof_vector_for_boundary_id(
    unsigned int b_id) {
  std::vector<DofIndexAndOrientationAndPosition> ret;
  std::vector<types::global_dof_index> local_line_dofs(fe.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe.dofs_per_face);
  std::set<DofNumber> face_set;
  triangulation.clear_user_flags();
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
          face_set.clear();
          face_set.insert(face_dofs_indices.begin(), face_dofs_indices.end());
          std::vector<DofIndexAndOrientationAndPosition> cell_dofs_and_orientations_and_points;
          for (unsigned int i = 0; i < dealii::GeometryInfo<3>::lines_per_face; i++) {
            std::vector<DofNumber> line_dofs(fe.dofs_per_line);
            cell->face(face)->line(i)->get_dof_indices(line_dofs);
            line_set.clear();
            line_set.insert(line_dofs.begin(), line_dofs.end());
            for(auto erase_it: line_set) {
              face_set.erase(erase_it);
            }
            if(!cell->face(face)->line(i)->user_flag_set()) {
              for (unsigned int j = 0; j < fe.dofs_per_line; j++) {
                DofIndexAndOrientationAndPosition new_item;
                new_item.index = line_dofs[j];
                new_item.position = cell->face(face)->line(i)->center();
                new_item.orientation = get_orientation(cell->face(face)->line(i)->vertex(0),
                            cell->face(face)->line(i)->vertex(1));
                cell_dofs_and_orientations_and_points.emplace_back(new_item);
              }
              cell->face(face)->line(i)->set_user_flag();
            }
          }
          for (auto item: face_set) {
            DofIndexAndOrientationAndPosition new_item;

            new_item.index = item;
            new_item.position = cell->face(face)->center();
            new_item.orientation = get_orientation(cell->face(face)->vertex(0),
                    cell->face(face)->vertex(1));
            cell_dofs_and_orientations_and_points.emplace_back(new_item);
          }
          for (auto item: cell_dofs_and_orientations_and_points) {
            ret.push_back(item);
          }
        }
      }
    }
  }
  std::sort(ret.begin(), ret.end(), compareDofBaseDataAndOrientation);
  return ret;
}

struct CellwiseAssemblyData {
  QGauss<3> quadrature_formula; 
  FEValues<3> fe_values;
  std::vector<Position> quadrature_points;
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
  quadrature_formula(GlobalParams.Nedelec_element_order + 2),
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
    mu = invert(transformation);

    if (Geometry.math_coordinate_in_waveguide(quadrature_points[q_index])) {
      epsilon = transformation * eps_in;
    } else {
      epsilon = transformation * eps_out;
    }

    const double JxW = fe_values.JxW(q_index);
    for (unsigned int i = 0; i < dofs_per_cell; i++) {
      Tensor<1, 3, ComplexNumber> I_Curl;
      Tensor<1, 3, ComplexNumber> I_Val;
      I_Curl = fe_values[fe_field].curl(i, q_index);
      I_Val = fe_values[fe_field].value(i, q_index);

      for (unsigned int j = 0; j < dofs_per_cell; j++) {
        Tensor<1, 3, ComplexNumber> J_Curl;
        Tensor<1, 3, ComplexNumber> J_Val;
        J_Curl = fe_values[fe_field].curl(j, q_index);
        J_Val = fe_values[fe_field].value(j, q_index);

        cell_matrix_real[i][j] += I_Curl * Conjugate_Vector(J_Curl)* JxW
            - ( I_Val * Conjugate_Vector(J_Val)) * JxW;
      }
    }
  }

  Tensor<1, 3, ComplexNumber> Conjugate_Vector(
      Tensor<1, 3, ComplexNumber> input) {
    Tensor<1, 3, ComplexNumber> ret;

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
  CellwiseAssemblyData cell_data(&fe, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
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
  matrix->compress(dealii::VectorOperation::add);
  write_matrix_and_rhs_metrics(matrix, rhs);
}

void NumericProblem::assemble_system(unsigned int shift,
    dealii::AffineConstraints<ComplexNumber> * constraints,
    dealii::PETScWrappers::MPI::SparseMatrix * matrix,
    NumericVectorDistributed *rhs) {
  CellwiseAssemblyData cell_data(&fe, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
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
  // matrix->compress(dealii::VectorOperation::add);
  // write_matrix_and_rhs_metrics(matrix, rhs);
}

void NumericProblem::write_matrix_and_rhs_metrics(dealii::PETScWrappers::MatrixBase * matrix, NumericVectorDistributed *rhs) {
  std::cout << "System Matrix l_infty norm: " << matrix->linfty_norm() << std::endl;
  std::cout << "System Matrix l_1 norm: " << matrix->l1_norm() << std::endl;
  std::cout << "Assembling done. L2-Norm of RHS: " << rhs->l2_norm() << std::endl;
}

#endif
