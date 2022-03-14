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
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
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
  exact_solution_is_initialized = false;
}

InnerDomain::~InnerDomain() {}

void InnerDomain::load_exact_solution() {
  if(!exact_solution_is_initialized) {
    dealii::IndexSet local_indices(n_locally_active_dofs);
    local_indices.add_range(0,n_locally_active_dofs);
    Constraints local_constraints(local_indices);
    local_constraints.close();
    exact_solution_interpolated.reinit(n_locally_active_dofs);
    VectorTools::project(dof_handler, local_constraints, dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2), *GlobalParams.source_field, exact_solution_interpolated);
    exact_solution_is_initialized = true;
    print_info("InnerDomain::load_exact_solution", "Norm of interpolated mode signal is " + std::to_string(exact_solution_interpolated.l2_norm()));
  }
}

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
  print_info("InnerDomain::make_grid", m, LoggingLevel::PRODUCTION_ALL);
  Position lower(Geometry.local_x_range.first, Geometry.local_y_range.first, Geometry.local_z_range.first);
  Position upper(Geometry.local_x_range.second, Geometry.local_y_range.second, Geometry.local_z_range.second);
  dealii::GridGenerator::subdivided_hyper_rectangle(temp_tria, repetitions, lower, upper, true);
  triangulation = reforge_triangulation(&temp_tria);
  dof_handler.distribute_dofs(fe);
  print_info("InnerDomain::make_grid", "Mesh Preparation finished. System has " + std::to_string(dof_handler.n_dofs()) + " degrees of freedom.", LoggingLevel::PRODUCTION_ONE);
}

bool compareIndexCenterPairs(std::pair<int, double> c1, std::pair<int, double> c2) {
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
          print_info("InnerDomain::get_surface_dof_vector_for_boundary_id", "There was an error!", LoggingLevel::PRODUCTION_ALL);
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
  bool has_input_interface = false;
  const FEValuesExtractors::Vector fe_field;
  Vector<ComplexNumber> incoming_wave_field;
  IndexSet constrained_dofs;
  Tensor<1,3,ComplexNumber> J;
  ExactSolution * es_for_j;

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

  void set_es_pointer(ExactSolution * in_es) {
    es_for_j = in_es;
  }

  void prepare_for_current_q_index(unsigned int q_index) {
    if(GlobalParams.Use_Predefined_Shape) {
      transformation = GlobalSpaceTransformation->get_Space_Transformation_Tensor(quadrature_points[q_index]);
    }
    mu = invert(transformation);
    const double kappa_2 = Geometry.kappa_2();
    if (Geometry.math_coordinate_in_waveguide(quadrature_points[q_index])) {
      epsilon = transformation * eps_in;
    } else {
      epsilon = transformation * eps_out;
    }
    bool compute_rhs = false;
    if(GlobalParams.Signal_coupling_method == SignalCouplingMethod::Tapering && GlobalParams.Index_in_z_direction == 0 && quadrature_points[q_index][2] < GlobalParams.tapering_max_z && quadrature_points[q_index][2] >= GlobalParams.tapering_min_z ) {
      compute_rhs = true;
    }
    if(compute_rhs) {
      J = evaluate_J_at(quadrature_points[q_index]);
    }
    std::vector<unsigned int> dof_indices(dofs_per_cell);
    cell->get_dof_indices(dof_indices);
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
        cell_matrix[i][j] += I_Curl * (mu * Conjugate_Vector(J_Curl)) * JxW - (kappa_2 * ( (epsilon * I_Val) * Conjugate_Vector(J_Val)) * JxW);
      }
      if(compute_rhs) {
        cell_rhs[i] += JxW * J * Conjugate_Vector(I_Val);
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

  Tensor<1,3,ComplexNumber> evaluate_J_at(Position p) {
    Tensor<1,3,ComplexNumber> ret;
    const ComplexNumber im(0,1);
    ComplexNumber phi, d_phi_dz, d_phi_dzz;
    const double omega_sq_eps = Geometry.eps_kappa_2(p);
    for(unsigned int i = 0; i < 3; i++) {
      ret[i] = 0;
    }
    if(p[2] < GlobalParams.tapering_min_z - FLOATING_PRECISION || p[2] >= GlobalParams.tapering_max_z) {
      return ret;
    }
    Position2D p2d;
    p2d[0] = p[0];
    p2d[1] = p[1];
    J_derivative_terms f_and_h_terms = es_for_j->get_derivative_terms(p2d);
    const double phi_coord_z = (p[2] - GlobalParams.tapering_min_z)/(GlobalParams.tapering_max_z - GlobalParams.tapering_min_z);
    phi = 2*(phi_coord_z*phi_coord_z*phi_coord_z) - 3*(phi_coord_z*phi_coord_z) + 1;
    const double zmax_minus_zmin = (GlobalParams.tapering_max_z - GlobalParams.tapering_min_z);
    d_phi_dz = 6 * (p[2] - GlobalParams.tapering_min_z) * (p[2] - GlobalParams.tapering_max_z) / (zmax_minus_zmin * zmax_minus_zmin * zmax_minus_zmin);
    d_phi_dzz = 12 * (p[2] - GlobalParams.tapering_min_z) / (zmax_minus_zmin * zmax_minus_zmin * zmax_minus_zmin) - 6/(zmax_minus_zmin * zmax_minus_zmin);
    ret[0] = phi * ( - f_and_h_terms.d_f_dyy + f_and_h_terms.beta * f_and_h_terms.beta * f_and_h_terms.f + im*f_and_h_terms.beta*f_and_h_terms.d_h_dx) 
      - f_and_h_terms.f*(d_phi_dzz + 2.0 * im * f_and_h_terms.beta * d_phi_dz) + d_phi_dz * f_and_h_terms.d_h_dx +  omega_sq_eps * f_and_h_terms.f;
    ret[1] = phi * (f_and_h_terms.d_f_dxy + im*f_and_h_terms.beta*f_and_h_terms.d_h_dy) + d_phi_dz * f_and_h_terms.d_h_dy ;
    ret[2] = phi * (im*f_and_h_terms.beta*f_and_h_terms.d_f_dx - f_and_h_terms.d_h_dxx - f_and_h_terms.d_h_dyy) + d_phi_dz*f_and_h_terms.d_f_dx + omega_sq_eps*f_and_h_terms.h;
    
    const ComplexNumber phase = std::exp(im * f_and_h_terms.beta * p[2]);
    for(unsigned int i = 0; i < 3; i++) {
      ret[i] *= -phase;
    }
    
    return ret;
  }
};

void InnerDomain::assemble_system(Constraints * constraints, dealii::PETScWrappers::MPI::SparseMatrix * matrix, NumericVectorDistributed *rhs) {
  CellwiseAssemblyDataNP cell_data(&fe, &dof_handler);
  load_exact_solution();
  ExactSolution * esp;
  if(GlobalParams.Index_in_z_direction == 0 && GlobalParams.Signal_coupling_method == SignalCouplingMethod::Tapering) {
      esp = new ExactSolution();
      cell_data.set_es_pointer(esp);
  }
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    cell_data.local_dof_indices = transform_local_to_global_dofs(cell_data.local_dof_indices);
    cell_data.cell_matrix = 0;
    cell_data.cell_rhs.reinit(cell_data.dofs_per_cell);
    cell_data.cell_rhs = 0;
    cell_data.fe_values.reinit(cell_data.cell);
    cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
    for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
      cell_data.prepare_for_current_q_index(q_index);
    }
    constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, true);
  }
  matrix->compress(dealii::VectorOperation::add);
  rhs->compress(dealii::VectorOperation::add);
  if(GlobalParams.Index_in_z_direction == 0 && GlobalParams.Signal_coupling_method == SignalCouplingMethod::Tapering) {
    delete esp;
  }
}

void InnerDomain::write_matrix_and_rhs_metrics(dealii::PETScWrappers::MatrixBase * matrix, NumericVectorDistributed *rhs) {
  print_info("InnerDomain::write_matrix_and_rhs_metrics", "Start", LoggingLevel::DEBUG_ALL);
  print_info("InnerDomain::write_matrix_and_rhs", "System Matrix l_1 norm: " + std::to_string(matrix->l1_norm()), LoggingLevel::PRODUCTION_ALL);
  print_info("InnerDomain::write_matrix_and_rhs", "RHS L_2 norm:  " + std::to_string(rhs->l2_norm()), LoggingLevel::PRODUCTION_ALL);
  print_info("InnerDomain::write_matrix_and_rhs_metrics", "End");
}

std::string InnerDomain::output_results(std::string in_filename, NumericVectorLocal in_solution, bool apply_transformation) {
  print_info("InnerDomain::output_results()", "Start");
  const unsigned int n_cells = dof_handler.get_triangulation().n_active_cells();
  unsigned int counter = 0;
  dealii::Vector<double> eps_abs(n_cells);
  for(auto it = dof_handler.begin_active(); it != dof_handler.end(); it++) {
    Position p = it->center();
    MaterialTensor transformation;
    if(apply_transformation) {
      for(unsigned int i = 0 ; i < 3; i++) {
        for(unsigned int j = 0; j < 3; j++) {
          if(i == j) {
            transformation[i][j] = ComplexNumber(1,0);
          } else {
            transformation[i][j] = ComplexNumber(0,0);
          }
        }
      }
    } else {
      transformation = GlobalSpaceTransformation->get_Space_Transformation_Tensor(p);
    }
    MaterialTensor epsilon;
    if (Geometry.math_coordinate_in_waveguide(p)) {
      epsilon = transformation * GlobalParams.Epsilon_R_in_waveguide;
    } else {
      epsilon = transformation * GlobalParams.Epsilon_R_outside_waveguide;
    }
    eps_abs[counter] = epsilon.norm();
    counter++;
  }
  if(apply_transformation) {
    GlobalSpaceTransformation->switch_application_mode(true);
    dealii::GridTools::transform(*GlobalSpaceTransformation, triangulation);
  }
  dealii::Vector<ComplexNumber> interpolated_exact_solution(in_solution.size());
  
  data_out.clear();
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(in_solution, "Solution");
  
  data_out.add_data_vector(eps_abs, "Epsilon");
  dealii::Vector<double> index_x(n_cells), index_y(n_cells), index_z(n_cells);
  for(unsigned int i = 0; i < n_cells; i++) {
    index_x[i] = GlobalParams.Index_in_x_direction;
    index_y[i] = GlobalParams.Index_in_y_direction;
    index_z[i] = GlobalParams.Index_in_z_direction;
  }
  data_out.add_data_vector(index_x, "IndexX");
  data_out.add_data_vector(index_y, "IndexY");
  data_out.add_data_vector(index_z, "IndexZ");
  std::string filename = GlobalOutputManager.get_numbered_filename(in_filename, GlobalParams.MPI_Rank, "vtu");
  std::ofstream outputvtu(filename);
  
  Function<3,ComplexNumber> * esc;
  if(!apply_transformation) {
    data_out.add_data_vector(exact_solution_interpolated, "Exact_Solution");
  } else {
    esc = GlobalParams.source_field;
    dealii::IndexSet local_indices(n_locally_active_dofs);
    local_indices.add_range(0,n_locally_active_dofs);
    Constraints local_constraints(local_indices);
    local_constraints.close();
    if(GlobalParams.Point_Source_Type == 0 || GlobalParams.Point_Source_Type == 3) {
      VectorTools::project(dof_handler, local_constraints, dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2), *esc, interpolated_exact_solution);
      data_out.add_data_vector(interpolated_exact_solution, "Exact_Solution");
    }
  }
  
  dealii::Vector<ComplexNumber> error_vector(in_solution.size());
  for(unsigned int i = 0; i < in_solution.size(); i++) {
    error_vector[i] = in_solution[i] - exact_solution_interpolated[i];
  }
  data_out.add_data_vector(error_vector, "SolutionError");

  data_out.build_patches();
  data_out.write_vtu(outputvtu);
  if(apply_transformation) {
    delete esc;
    GlobalSpaceTransformation->switch_application_mode(false);
    dealii::GridTools::transform(*GlobalSpaceTransformation, triangulation);
  }
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

ComplexNumber InnerDomain::compute_signal_strength(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution) {
  ComplexNumber ret(0,0);
  if(GlobalParams.Index_in_z_direction == GlobalParams.Blocks_in_z_direction - 1) {
    NumericVectorLocal local_solution;
    local_solution.reinit(n_locally_active_dofs);
    for(unsigned int i = 0; i < n_locally_active_dofs; i++) {
      local_solution[i] = in_solution->operator[](global_index_mapping[i]);
    }
    Vector<ComplexNumber> fe_evaluation(3);
    Vector<ComplexNumber> mode(3);
    std::vector<Position> quadrature_points;
    for(auto cell : triangulation) {
      if(cell.at_boundary()) {
        for(unsigned int i = 0; i < 6; i++) {
          if(cell.face(i)->boundary_id() == 5) {
            quadrature_points.push_back(cell.face(i)->center());
          }
        } 
      }
    }
    for(unsigned int index = 0; index < quadrature_points.size(); index++) {
      quadrature_points[index][2] = quadrature_points[index][2] - 2 * FLOATING_PRECISION; // This is only to make sure that even on large mesges, there are no rounding errors that lead the code to throw an error because the position isnt "inside" the mesh.
    }
    for(unsigned int index = 0; index < quadrature_points.size(); index++) {
      VectorTools::point_value(dof_handler, local_solution, quadrature_points[index], fe_evaluation);
      GlobalParams.source_field->vector_value(quadrature_points[index], mode);
      for(unsigned int comp = 0; comp < 3; comp++) {
        mode[comp] = conjugate(mode[comp]);
      }
      ret += fe_evaluation[0]*mode[0] + fe_evaluation[1]*mode[1] + fe_evaluation[2] * mode[2];
    }
    ret /= (quadrature_points.size());
    return ret;
  } 
  return ret;
}

ComplexNumber InnerDomain::compute_mode_strength() {
  ComplexNumber ret(0,0);
  if(GlobalParams.Index_in_z_direction == GlobalParams.Blocks_in_z_direction - 1) {
    Vector<ComplexNumber> mode_a(3), mode_b(3);
    std::vector<Position> quadrature_points;
    for(auto cell : triangulation) {
      if(cell.at_boundary()) {
        for(unsigned int i = 0; i < 6; i++) {
          if(cell.face(i)->boundary_id() == 5) {
            quadrature_points.push_back(cell.face(i)->center());
          }
        } 
      }
    }
    for(unsigned int index = 0; index < quadrature_points.size(); index++) {
      quadrature_points[index][2] = quadrature_points[index][2] - 2 * FLOATING_PRECISION;
    }
    for(unsigned int index = 0; index < quadrature_points.size(); index++) {
      GlobalParams.source_field->vector_value(quadrature_points[index], mode_a);
      for(unsigned int comp = 0; comp < 3; comp++) {
        mode_b[comp] = conjugate(mode_a[comp]);
      }
      ret += mode_a[0]*mode_b[0] + mode_a[1]*mode_b[1] + mode_a[2] * mode_b[2];
    }
    ret /= (quadrature_points.size());
    return ret;
  } 
  return ret;
}

FEErrorStruct InnerDomain::compute_errors(dealii::LinearAlgebra::distributed::Vector<ComplexNumber> * in_solution) {
  FEErrorStruct ret;
  dealii::Vector<double> cell_vector (triangulation.n_active_cells());
  QGauss<3> q(GlobalParams.Nedelec_element_order + 2);
  NumericVectorLocal local_solution(n_locally_active_dofs);
  for(unsigned int i =0 ; i < n_locally_active_dofs; i++) {
    local_solution[i] = in_solution->operator[](global_index_mapping[i]);
  }
  VectorTools::integrate_difference(dof_handler, local_solution, *GlobalParams.source_field, cell_vector, q, dealii::VectorTools::NormType::L2_norm);
  ret.L2 = VectorTools::compute_global_error(triangulation, cell_vector, dealii::VectorTools::NormType::L2_norm);
  ret.L2 /= in_solution->l2_norm();
  VectorTools::integrate_difference(dof_handler, local_solution, *GlobalParams.source_field, cell_vector, q, dealii::VectorTools::NormType::Linfty_norm);
  ret.Linfty = VectorTools::compute_global_error(triangulation, cell_vector, dealii::VectorTools::NormType::Linfty_norm);
  ret.Linfty /= in_solution->linfty_norm();
  return ret;
}

std::vector<std::vector<ComplexNumber>> InnerDomain::evaluate_at_positions(std::vector<Position> in_positions, NumericVectorLocal in_solution) {
  std::vector<std::vector<ComplexNumber>> ret;
  QGauss<3> q(GlobalParams.Nedelec_element_order + 2);
  for(unsigned int i = 0; i < in_positions.size(); i++) {
    Vector<ComplexNumber> fe_evaluation(3);
    VectorTools::point_value(dof_handler, in_solution, in_positions[i], fe_evaluation);
    std::vector<ComplexNumber> point_val;
    point_val.push_back(fe_evaluation[0]);
    point_val.push_back(fe_evaluation[1]);
    point_val.push_back(fe_evaluation[2]);
    ret.push_back(point_val);
  }
  return ret;
}

std::vector<FEAdjointEvaluation> InnerDomain::compute_local_shape_gradient_data(NumericVectorLocal & in_solution, NumericVectorLocal & in_adjoint) {
  std::vector<FEAdjointEvaluation> ret;
  QGauss<3> quadrature_formula(1); 
  const FEValuesExtractors::Vector fe_field(0);
  FEValues<3> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);
  std::vector<unsigned int> local_dof_indices(fe.n_dofs_per_cell());
  for (DofHandler3D::active_cell_iterator cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    fe_values.reinit(cell);
    auto q_points = fe_values.get_quadrature_points();
    for(unsigned int q_index = 0; q_index < quadrature_formula.size(); q_index++) {
      cell->get_dof_indices(local_dof_indices);
      Position p = q_points[q_index];
      FEAdjointEvaluation item;
      item.x = p;
      for(unsigned int i = 0; i < 3; i++) {
        item.primal_field[i] = 0;
      }
      for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++) {
        Tensor<1, 3, ComplexNumber> I_Val;
        I_Val = fe_values[fe_field].value(i, q_index);
        item.primal_field += I_Val * in_solution[local_dof_indices[i]];
        item.adjoint_field += I_Val * in_adjoint[local_dof_indices[i]];
      }
      ret.push_back(item);
    }
  }
  return ret;
}
