#include "InnerDomain.h"

#include <functional>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_project.h>
#include <deal.II/numerics/vector_tools_project.templates.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
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
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/PointSourceField.h"
#include "../Solutions/ExactSolutionRamped.h"
#include "../Solutions/ExactSolutionConjugate.h"
#include "../SpaceTransformations/SpaceTransformation.h"

InnerDomain::InnerDomain(unsigned int in_level)
    :
      mesh_generator{},
    fe(GlobalParams.Nedelec_element_order),
      triangulation(Triangulation<3>::MeshSmoothing(Triangulation<3>::none)),
      dof_handler(triangulation){
  level = in_level;
}

InnerDomain::~InnerDomain() {}

bool compareConstraintPairs(ConstraintPair v1, ConstraintPair v2) {
  return (v1.left < v2.left);
}

void InnerDomain::make_grid() {
  Triangulation<3> temp_tria;
  std::vector<unsigned int> repetitions;
  repetitions.push_back(GlobalParams.Cells_in_x);
  repetitions.push_back(GlobalParams.Cells_in_y);
  repetitions.push_back(GlobalParams.Cells_in_z);
  std::string m = "Cells: " + std::to_string(GlobalParams.Cells_in_x) + " x " + std::to_string(GlobalParams.Cells_in_y) + " x " + std::to_string(GlobalParams.Cells_in_z) + " Geometry: [" + std::to_string(Geometry.local_x_range.first) + "," + std::to_string(Geometry.local_x_range.second);
  m += "] x [" + std::to_string(Geometry.local_y_range.first) + "," + std::to_string(Geometry.local_y_range.second) + "] x [" + std::to_string(Geometry.local_z_range.first) + "," + std::to_string(Geometry.local_z_range.second) + "]";
  print_info("InnerDomain::make_grid", m, false, LoggingLevel::PRODUCTION_ALL);
  Position lower(Geometry.local_x_range.first, Geometry.local_y_range.first, Geometry.local_z_range.first);
  Position upper(Geometry.local_x_range.second, Geometry.local_y_range.second, Geometry.local_z_range.second);
  dealii::GridGenerator::subdivided_hyper_rectangle(temp_tria, repetitions, lower, upper, true);
  triangulation = reforge_triangulation(&temp_tria);
  dof_handler.distribute_dofs(fe);
  print_info("InnerDomain::make_grid", "Mesh Preparation finished. System has " + std::to_string(dof_handler.n_dofs()) + " degrees of freedom.", false, LoggingLevel::PRODUCTION_ONE);
  print_info("InnerDomain::make_grid", "end");
}

bool compareIndexCenterPairs(std::pair<int, double> c1,
                             std::pair<int, double> c2) {
  return c1.second < c2.second;
}

void InnerDomain::fill_sparsity_pattern( dealii::DynamicSparsityPattern *in_pattern, Constraints *in_constraints) {
  auto end = dof_handler.end();
  std::vector<DofNumber> cell_dof_indices(fe.dofs_per_cell);
  for(auto cell = dof_handler.begin_active(); cell != end; cell++) {
    cell->get_dof_indices(cell_dof_indices);
    cell_dof_indices = transform_local_to_global_dofs(cell_dof_indices);
    in_constraints->add_entries_local_to_global(cell_dof_indices, *in_pattern);
  }
}

std::vector<InterfaceDofData> InnerDomain::get_surface_dof_vector_for_boundary_id(unsigned int b_id) {
  std::vector<InterfaceDofData> ret;
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
          print_info("InnerDomain::get_surface_dof_vector_for_boundary_id", "There was an error!", false, LoggingLevel::PRODUCTION_ALL);
        }
        if (cell->face(face)->boundary_id() == b_id) {
          found_one = true;
          std::vector<DofNumber> face_dofs_indices(fe.dofs_per_face);
          cell->face(face)->get_dof_indices(face_dofs_indices);
          face_set.clear();
          face_set.insert(face_dofs_indices.begin(), face_dofs_indices.end());
          std::vector<InterfaceDofData> cell_dofs_and_orientations_and_points;
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
                InterfaceDofData new_item;
                new_item.index = line_dofs[j];
                new_item.base_point = cell->face(face)->line(i)->center();
                new_item.order = j;
                cell_dofs_and_orientations_and_points.push_back(new_item);
              }
              cell->face(face)->line(i)->set_user_flag();
            }
          }
          unsigned int index = 0;
          for (auto item: face_set) {
            InterfaceDofData new_item;
            new_item.index = item;
            new_item.base_point = cell->face(face)->center();
            new_item.order = 0;
            cell_dofs_and_orientations_and_points.push_back(new_item);
            index++;
          }
          for (auto item: cell_dofs_and_orientations_and_points) {
            ret.push_back(item);
          }
        }
      }
    }
  }
  ret.shrink_to_fit();
  std::sort(ret.begin(), ret.end(), compareDofBaseDataAndOrientation);
  return ret;
}

struct CellwiseAssemblyDataNP {
  QGauss<3> quadrature_formula; 
  FEValues<3> fe_values;
  std::vector<Position> quadrature_points;
  const unsigned int dofs_per_cell;
  const unsigned int n_q_points;
  FullMatrix<ComplexNumber> cell_matrix;
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
  ExactSolutionRamped exact_solution_ramped;
  bool has_input_interface = false;
  const FEValuesExtractors::Vector fe_field;
  Vector<ComplexNumber> incoming_wave_field;
  IndexSet constrained_dofs;

  CellwiseAssemblyDataNP(dealii::FE_NedelecSZ<3> * fe, DofHandler3D * dof_handler):
  quadrature_formula(GlobalParams.Nedelec_element_order + 2),
  fe_values(*fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points),
  dofs_per_cell(fe->dofs_per_cell),
  n_q_points(quadrature_formula.size()),
  cell_matrix(dofs_per_cell,dofs_per_cell),
  eps_in(GlobalParams.Epsilon_R_in_waveguide),
  eps_out(GlobalParams.Epsilon_R_outside_waveguide),
  mu_zero(1.0),
  cell_rhs(dofs_per_cell),
  local_dof_indices(dofs_per_cell),
  exact_solution_ramped(true, false),
  fe_field(0)
  { 
    // has_input_interface = GlobalParams.Index_in_z_direction == 0;
    has_input_interface = false;
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
    if(has_input_interface) {
      ExactSolution es(true, false);
      IndexSet owned_dofs(dof_handler->n_dofs());
      owned_dofs.add_range(0, dof_handler->n_dofs());
      AffineConstraints<ComplexNumber> constraints_local(owned_dofs);
      VectorTools::project_boundary_values_curl_conforming_l2(*dof_handler, 0, es, 4, constraints_local);
      incoming_wave_field.reinit(owned_dofs.size());
      constrained_dofs.set_size(dof_handler->n_dofs());
      for(unsigned int i = 0; i < owned_dofs.size(); i++) {
        if(constraints_local.is_constrained(i)) {
          incoming_wave_field[i] = constraints_local.get_inhomogeneity(i);
          constrained_dofs.add_index(i);
        } else {
          incoming_wave_field[i] = 0;
        }
      }
    }
  };

  void prepare_for_current_q_index(unsigned int q_index) {
    if(GlobalParams.Use_Predefined_Shape) {
      transformation = GlobalSpaceTransformation->get_Space_Transformation_Tensor(quadrature_points[q_index]);
    }
    mu = invert(transformation);
    const double eps_kappa_2 = Geometry.eps_kappa_2(quadrature_points[q_index]);
    if (Geometry.math_coordinate_in_waveguide(quadrature_points[q_index])) {
      epsilon = transformation * eps_in;
    } else {
      epsilon = transformation * eps_out;
    }
    
    std::vector<unsigned int> dof_indices(dofs_per_cell);
    cell->get_dof_indices(dof_indices);
    const double JxW = fe_values.JxW(q_index);
    for (unsigned int i = 0; i < dofs_per_cell; i++) {
      Tensor<1, 3, ComplexNumber> I_Curl;
      Tensor<1, 3, ComplexNumber> I_Val;
      I_Curl = fe_values[fe_field].curl(i, q_index);
      I_Val = fe_values[fe_field].value(i, q_index);
      if(GlobalParams.Signal_coupling_method == SignalCouplingMethod::Jump) {
        if(cell->at_boundary(4) && has_input_interface) {
          if(!constrained_dofs.is_element(dof_indices[i])) {
            for(unsigned int j = 0; j < dofs_per_cell; j++) {
              if(constrained_dofs.is_element(dof_indices[j])) {
                Tensor<1, 3, ComplexNumber> J_Curl;
                Tensor<1, 3, ComplexNumber> J_Val;
                J_Curl = fe_values[fe_field].curl(j, q_index);
                J_Val = fe_values[fe_field].value(j, q_index);
                cell_rhs[i] += incoming_wave_field[dof_indices[j]] * (I_Curl * Conjugate_Vector(J_Curl) * JxW - (eps_kappa_2 * ( I_Val * Conjugate_Vector(J_Val)) * JxW));
              }
            }
          }
        }
      }
      for (unsigned int j = 0; j < dofs_per_cell; j++) {
        Tensor<1, 3, ComplexNumber> J_Curl;
        Tensor<1, 3, ComplexNumber> J_Val;
        J_Curl = fe_values[fe_field].curl(j, q_index);
        J_Val = fe_values[fe_field].value(j, q_index);
        cell_matrix[i][j] += I_Curl * Conjugate_Vector(J_Curl) * JxW - (eps_kappa_2 * ( I_Val * Conjugate_Vector(J_Val)) * JxW);
      }
    }
  }

  Tensor<1, 3, ComplexNumber> Conjugate_Vector(Tensor<1, 3, ComplexNumber> input) {
    Tensor<1, 3, ComplexNumber> ret;
    for (int i = 0; i < 3; i++) {
      ret[i].real(input[i].real());
      ret[i].imag(-input[i].imag());
    }
    return ret;
  }

};

void InnerDomain::assemble_system(Constraints * constraints,
    dealii::PETScWrappers::SparseMatrix *matrix,
    NumericVectorDistributed *rhs) {
  CellwiseAssemblyDataNP cell_data(&fe, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    cell_data.local_dof_indices = transform_local_to_global_dofs(cell_data.local_dof_indices);
    cell_data.cell_matrix = 0;
    cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
    cell_data.fe_values.reinit(cell_data.cell);
    cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
    for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
      cell_data.prepare_for_current_q_index(q_index);
    }
    constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, true);
  }
  matrix->compress(dealii::VectorOperation::add);
  rhs->compress(dealii::VectorOperation::add);
}

void InnerDomain::assemble_system(Constraints * constraints,
    dealii::SparseMatrix<ComplexNumber> *matrix) {
  CellwiseAssemblyDataNP cell_data(&fe, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    cell_data.cell_matrix = 0;
    cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
    cell_data.fe_values.reinit(cell_data.cell);
    cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
    for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
      cell_data.prepare_for_current_q_index(q_index);
    }
    constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.local_dof_indices, *matrix);
  }
  matrix->compress(dealii::VectorOperation::add);
}

void InnerDomain::assemble_system(Constraints * constraints,
    dealii::PETScWrappers::MPI::SparseMatrix * matrix,
    NumericVectorDistributed *rhs) {
  CellwiseAssemblyDataNP cell_data(&fe, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    cell_data.local_dof_indices = transform_local_to_global_dofs(cell_data.local_dof_indices);
    cell_data.cell_matrix = 0;
    cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
    cell_data.fe_values.reinit(cell_data.cell);
    cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
    for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
      cell_data.prepare_for_current_q_index(q_index);
    }
    constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, true);
  }
  matrix->compress(dealii::VectorOperation::add);
  rhs->compress(dealii::VectorOperation::add);
}

void InnerDomain::write_matrix_and_rhs_metrics(dealii::PETScWrappers::MatrixBase * matrix, NumericVectorDistributed *rhs) {
  print_info("InnerDomain::write_matrix_and_rhs_metrics", "Start", LoggingLevel::DEBUG_ALL);
  print_info("InnerDomain::write_matrix_and_rhs", "System Matrix l_1 norm: " + std::to_string(matrix->l1_norm()), false, LoggingLevel::PRODUCTION_ALL);
  print_info("InnerDomain::write_matrix_and_rhs", "RHS L_2 norm:  " + std::to_string(rhs->l2_norm()), false, LoggingLevel::PRODUCTION_ALL);
  print_info("InnerDomain::write_matrix_and_rhs_metrics", "End");
}

std::string InnerDomain::output_results(std::string in_filename, NumericVectorLocal in_solution) {
  print_info("InnerDomain::output_results()", "Start");
  data_out.clear();
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(in_solution, "Solution");
  std::string filename = GlobalOutputManager.get_numbered_filename(in_filename, GlobalParams.MPI_Rank, "vtu");
  std::ofstream outputvtu(filename);
  
  Function<3,ComplexNumber> * esc;
  esc = GlobalParams.source_field;
  dealii::IndexSet local_indices(n_locally_active_dofs);
  local_indices.add_range(0,n_locally_active_dofs);
  Constraints local_constraints(local_indices);
  local_constraints.close();
  dealii::Vector<ComplexNumber> interpolated_exact_solution(in_solution.size());
  VectorTools::project(dof_handler, local_constraints, dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2), *esc, interpolated_exact_solution);
  
  data_out.add_data_vector(interpolated_exact_solution, "Exact_Solution");
  
  data_out.build_patches();
  data_out.write_vtu(outputvtu);

  print_info("InnerDomain::output_results()", "End");
  return filename;
}


DofCount InnerDomain::compute_n_locally_owned_dofs() {
  IndexSet set_of_locally_owned_dofs(dof_handler.n_dofs());
  set_of_locally_owned_dofs.add_range(0,dof_handler.n_dofs());
  IndexSet dofs_to_remove(dof_handler.n_dofs());
  for(unsigned int surf = 0; surf < 6; surf += 2) {
    if(Geometry.levels[level].surface_type[surf] == SurfaceType::NEIGHBOR_SURFACE) {
      std::vector<InterfaceDofData> dofs = get_surface_dof_vector_for_boundary_id(surf);
      for(unsigned int i = 0; i < dofs.size(); i++) {
        dofs_to_remove.add_index(dofs[i].index);
      }
    }
  }
  set_of_locally_owned_dofs.subtract_set(dofs_to_remove);
  return set_of_locally_owned_dofs.n_elements();
}

DofCount InnerDomain::compute_n_locally_active_dofs() {
  return dof_handler.n_dofs();
}

void InnerDomain::determine_non_owned_dofs() {
  for(unsigned int i = 0; i < 6; i += 2) {
    if(Geometry.levels[level].surface_type[i] == SurfaceType::NEIGHBOR_SURFACE) {
      std::vector<InterfaceDofData> dof_data = get_surface_dof_vector_for_boundary_id(i);
      std::vector<unsigned int> local_dof_indices(dof_data.size());
      for(unsigned int j = 0; j < dof_data.size(); j++) {
        local_dof_indices[j] = dof_data[j].index;
      }
      mark_local_dofs_as_non_local(local_dof_indices);
    }
  }
}
