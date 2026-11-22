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
#include <array>
#include <cmath>
#include <map>
#include <memory>
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

/**
 * @brief Working data for the assembly of the system matrix and right-hand side of the inner domain, cell by cell.
 *
 * InnerDomain::assemble_system() advances the member cell, reinitializes fe_values and then calls prepare_for_current_q_index() for every quadrature point, which adds to cell_matrix and cell_rhs.
 */
struct CellwiseAssemblyDataNP {
  /** @brief Gauss quadrature with Nedelec_element_order + 2 points per direction. */
  QGauss<3> quadrature_formula; 
  /** @brief Values, gradients, JxW values and quadrature points of the shape functions on the current cell. */
  FEValues<3> fe_values;
  /** @brief The quadrature points of the current cell in the coordinates of the mesh. Set by the assembly loop. */
  std::vector<Position> quadrature_points;
  /** @brief Number of dofs of the finite element per cell. */
  const unsigned int dofs_per_cell;
  /** @brief Number of quadrature points per cell. */
  const unsigned int n_q_points;
  /** @brief Contributions of the current cell to the system matrix. */
  FullMatrix<ComplexNumber> cell_matrix;
  /** @brief Relative permittivity inside the waveguide core. */
  const double eps_in;
  /** @brief Relative permittivity outside the waveguide core. */
  const double eps_out;
  /** @brief Set to 1. Not used. */
  const double mu_zero;
  /** @brief Contributions of the current cell to the right-hand side. */
  Vector<ComplexNumber> cell_rhs;
  /** @brief Tensor of the space transformation at the current quadrature point. */
  MaterialTensor transformation;
  /** @brief Permittivity tensor at the current quadrature point: transformation times eps_in or eps_out. */
  MaterialTensor epsilon;
  /** @brief Inverse of transformation. It is the tensor in the curl-curl term, i.e. the inverse permeability tensor. */
  MaterialTensor mu;
  /** @brief Dof indices of the current cell. The assembly loop stores the indices in the global numbering of the level here. */
  std::vector<DofNumber> local_dof_indices;
  /** @brief The current cell. */
  DofHandler3D::active_cell_iterator cell;
  /** @brief End iterator of the cell loop. */
  DofHandler3D::active_cell_iterator end_cell;
  /** @brief Not used. */
  bool has_input_interface = false;
  /** @brief Extractor for the vector valued field, which starts at component 0. */
  const FEValuesExtractors::Vector fe_field;
  /** @brief Not used. */
  Vector<ComplexNumber> incoming_wave_field;
  /** @brief Not used. */
  IndexSet constrained_dofs;
  /** @brief Mode profile used for the tapered source term. If it is a null pointer, no right-hand side is assembled. */
  ExactSolution * es_for_j = nullptr;

  /**
   * @brief Sets up quadrature and FEValues, sets the transformation tensor to the identity and the cell iterators to the range of active cells.
   *
   * @param fe The finite element of the inner domain.
   * @param dof_handler The dof handler of the inner domain.
   */
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

  /**
   * @brief Sets the mode profile used for the tapered source term. This enables the assembly of the right-hand side.
   *
   * @param in_es Pointer to the mode profile. The object must outlive the assembly.
   */
  void set_es_pointer(ExactSolution * in_es) {
    es_for_j = in_es;
  }

  /**
   * @brief Adds the contributions of one quadrature point to cell_matrix and cell_rhs.
   *
   * The matrix entry (i,j) receives \f$\left(\operatorname{curl}\phi_i \cdot \mu^{-1} \overline{\operatorname{curl}\phi_j} - \omega^2 (\epsilon \phi_i) \cdot \overline{\phi_j}\right) JxW\f$. If a mode profile is set and the point is inside the taper region (tapering_min_z <= z < tapering_max_z), the source term of evaluate_taper_source_at() is added to cell_rhs.
   *
   * @param q_index Index of the quadrature point on the current cell.
   */
  void prepare_for_current_q_index(unsigned int q_index) {
    // The material tensors of transformation optics. For a straight waveguide this is the identity. It has to be
    // applied for every kind of shape (predefined shapes as well as shapes described by shape functions in
    // optimization runs) and it has to be the same tensor the PML domains use.
    transformation = GlobalSpaceTransformation->get_Space_Transformation_Tensor(quadrature_points[q_index]);
    mu = invert(transformation);
    const double kappa_2 = Geometry.kappa_2();
    if (Geometry.math_coordinate_in_waveguide(quadrature_points[q_index])) {
      epsilon = transformation * eps_in;
    } else {
      epsilon = transformation * eps_out;
    }
    bool compute_rhs = false;
    if(es_for_j != nullptr && quadrature_points[q_index][2] < GlobalParams.tapering_max_z && quadrature_points[q_index][2] >= GlobalParams.tapering_min_z ) {
      compute_rhs = true;
    }
    Tensor<1,3,ComplexNumber> source_for_values, source_for_curls;
    if(compute_rhs) {
      evaluate_taper_source_at(quadrature_points[q_index], source_for_values, source_for_curls);
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
        cell_rhs[i] += JxW * (source_for_values * Conjugate_Vector(I_Val) + source_for_curls * Conjugate_Vector(I_Curl));
      }
    }
  }

  /**
   * @brief Computes the complex conjugate of every component of a vector.
   *
   * @param input The vector.
   * @return The conjugated vector.
   */
  Tensor<1, 3, ComplexNumber> Conjugate_Vector(Tensor<1, 3, ComplexNumber> input) {
    Tensor<1, 3, ComplexNumber> ret;
    for (int i = 0; i < 3; i++) {
      ret[i].real(input[i].real());
      ret[i].imag(-input[i].imag());
    }
    return ret;
  }

  /**
   * Tapered signal coupling. We want the solution to be u = psi(z) * E, where E is the mode of the waveguide and
   * psi = 1 - phi ramps from 0 (z <= tapering_min_z) to 1 (z >= tapering_max_z). Below the taper only the
   * backward travelling (scattered) field remains, which is absorbed by the PML on the input side.
   * With the operator L u = curl(mu^-1 curl u) - omega^2 eps u and L E = 0 one finds
   *
   *   F = L(psi E) = grad(psi) x (mu^-1 curl E) + curl( mu^-1 (grad(psi) x E) ).
   *
   * Tested with a function v and integrated by parts (the boundary terms vanish because grad(psi) = 0 at both ends
   * of the taper) this gives the right hand side
   *
   *   (F, v) = int [grad(psi) x (mu^-1 curl E)] . conj(v) + [mu^-1 (grad(psi) x E)] . conj(curl v) dV.
   *
   * Compared to evaluating F pointwise (see the chapter on tapered coupling in the dissertation) this form only
   * requires first derivatives of the tabulated mode profile and it remains valid across the jump of the
   * permittivity at the surface of the waveguide core.
   *
   * @param p The quadrature point.
   * @param source_for_values is set to grad(psi) x (mu^-1 curl E).
   * @param source_for_curls is set to mu^-1 (grad(psi) x E).
   */
  void evaluate_taper_source_at(const Position & p, Tensor<1,3,ComplexNumber> & source_for_values, Tensor<1,3,ComplexNumber> & source_for_curls) {
    const double taper_length = GlobalParams.tapering_max_z - GlobalParams.tapering_min_z;
    const double t = (p[2] - GlobalParams.tapering_min_z) / taper_length;
    // psi(t) = 1 - (2t^3 - 3t^2 + 1) = 3t^2 - 2t^3
    const double d_psi_dz = 6.0 * t * (1.0 - t) / taper_length;
    const Tensor<1,3,ComplexNumber> E = es_for_j->val(p);
    const Tensor<1,3,ComplexNumber> mu_inverse_curl_E = mu * es_for_j->curl(p);
    // grad(psi) = (0, 0, d_psi_dz), so grad(psi) x a = d_psi_dz * (-a_y, a_x, 0).
    source_for_values[0] = - d_psi_dz * mu_inverse_curl_E[1];
    source_for_values[1] =   d_psi_dz * mu_inverse_curl_E[0];
    source_for_values[2] = 0;
    Tensor<1,3,ComplexNumber> grad_psi_cross_E;
    grad_psi_cross_E[0] = - d_psi_dz * E[1];
    grad_psi_cross_E[1] =   d_psi_dz * E[0];
    grad_psi_cross_E[2] = 0;
    source_for_curls = mu * grad_psi_cross_E;
  }
};

void InnerDomain::assemble_system(Constraints * constraints, dealii::PETScWrappers::MPI::SparseMatrix * matrix, NumericVectorDistributed *rhs) {
  CellwiseAssemblyDataNP cell_data(&fe, &dof_handler);
  load_exact_solution();
  // The mode profile is only required if the signal is coupled in by a tapered source term. The source term is
  // then assembled on every process whose cells intersect the taper region.
  std::unique_ptr<ExactSolution> esp;
  if(GlobalParams.Signal_coupling_method == SignalCouplingMethod::Tapering) {
      esp = std::make_unique<ExactSolution>();
      cell_data.set_es_pointer(esp.get());
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
    bool is_skeq_sym = true;
    for(unsigned int i = 0; i < cell_data.cell_matrix.n_rows(); i++) {
      for(unsigned int j = 0; j < i; j++) {
        if(!(std::abs(cell_data.cell_matrix[i][j] - std::conj(cell_data.cell_matrix[j][i])) < FLOATING_PRECISION)) {
          is_skeq_sym = false;
        }
      }
    }
    if(!is_skeq_sym) std::cout << "Not fulfilled!" << std::endl;
    constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, true);
  }
  matrix->compress(dealii::VectorOperation::add);
  rhs->compress(dealii::VectorOperation::add);
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
        mode[comp] = std::conj(mode[comp]);
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
        mode_b[comp] = std::conj(mode_a[comp]);
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
  FEValues<3> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points);
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
        item.adjoint_field[i] = 0;
        item.primal_field_curl[i] = 0;
        item.adjoint_field_curl[i] = 0;
      }
      for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++) {
        Tensor<1, 3, ComplexNumber> I_Val;
        I_Val = fe_values[fe_field].value(i, q_index);
        Tensor<1, 3, ComplexNumber> I_Curl;
        I_Curl = fe_values[fe_field].curl(i, q_index);
        item.primal_field += I_Val * in_solution[local_dof_indices[i]];
        item.adjoint_field += I_Val * in_adjoint[local_dof_indices[i]];
        item.primal_field_curl += I_Curl * in_solution[local_dof_indices[i]];
        item.adjoint_field_curl += I_Curl * in_adjoint[local_dof_indices[i]];
      }
      ret.push_back(item);
    }
  }
  return ret;
}

ComplexNumber InnerDomain::compute_kappa(NumericVectorLocal & in_solution) {
  ComplexNumber ret;
  QGauss<2> quadrature_formula(1); 
  const FEValuesExtractors::Vector fe_field(0);
  FEFaceValues<3> fe_values(fe, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
  std::vector<unsigned int> local_dof_indices(fe.n_dofs_per_cell());
  for (DofHandler3D::active_cell_iterator cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    for(unsigned int face = 0; face < 6; face++) {
      if(std::abs(cell->face(face)->center()[2] - Geometry.global_z_range.second) < FLOATING_PRECISION) {
        fe_values.reinit(cell,face);
        double JxW;
        auto q_points = fe_values.get_quadrature_points();
        for(unsigned int q_index = 0; q_index < quadrature_formula.size(); q_index++) {
          cell->get_dof_indices(local_dof_indices);
          JxW = fe_values.get_JxW_values()[q_index];
          Position p = q_points[q_index];
          Tensor<1,3, ComplexNumber> E0;
          for(unsigned int i = 0; i < 3; i++) {
            E0[i] = GlobalParams.source_field->value(p,i);
          }
          for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++) {
            // std::cout << in_solution[local_dof_indices[i]] << " and " <<JxW << " and " << E0.norm() << " and " << fe_values[fe_field].value(i, q_index).norm() << std::endl;
            for(unsigned int j = 0; j < 3; j++) {
              ret += std::conj((in_solution[local_dof_indices[i]] * fe_values[fe_field].value(i, q_index))[j]) * E0[j] * JxW;
            }
          }
        }
      }
    }
  }
  return Utilities::MPI::sum(ret, MPI_COMM_WORLD);
}

void InnerDomain::set_rhs_for_adjoint_problem(NumericVectorLocal & in_solution, NumericVectorDistributed * in_rhs) {
  ComplexNumber kappa = compute_kappa(in_solution);
  print_info("InnerDomain::Set_rhs_for_adjoing_problem", "Abs Kappa is " + std::to_string(std::abs(kappa)));
  QGauss<2> quadrature_formula(1); 
  const FEValuesExtractors::Vector fe_field(0);
  FEFaceValues<3> fe_values(fe, quadrature_formula, update_values | update_JxW_values  | update_quadrature_points);
  std::vector<unsigned int> local_dof_indices(fe.n_dofs_per_cell());
  for (DofHandler3D::active_cell_iterator cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    for(unsigned int face =0; face < 6; face++) {
      if(std::abs(cell->face(face)->center()[2] - Geometry.global_z_range.second) < FLOATING_PRECISION) {
        fe_values.reinit(cell,face);
        double JxW;
        auto q_points = fe_values.get_quadrature_points();
        for(unsigned int q_index = 0; q_index < quadrature_formula.size(); q_index++) {
          cell->get_dof_indices(local_dof_indices);
          JxW = fe_values.get_JxW_values()[q_index];
          Position p = q_points[q_index];
          Tensor<1,3, ComplexNumber> E0;
          for(unsigned int i = 0; i < 3; i++) {
            E0[i] = GlobalParams.source_field->value(p,i);
          }
          for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++) {
            Tensor<1,3,ComplexNumber> Ival = std::conj(in_solution[local_dof_indices[i]]) * fe_values[fe_field].value(i, q_index);
            (*in_rhs)[local_dof_indices[i]] -= 2 * (kappa * (E0 * Ival) * JxW).real();
          }
        }
      }
    }
  }
}
double InnerDomain::assemble_adjoint_rhs_for_signal_strength(NumericVectorDistributed * in_rhs) {
  double mode_norm = 0;
  if(GlobalParams.Index_in_z_direction != GlobalParams.Blocks_in_z_direction - 1) {
    return mode_norm;
  }
  // The same quadrature (one point at the face centers) as in compute_signal_strength.
  QGauss<2> quadrature_formula(1);
  const FEValuesExtractors::Vector fe_field(0);
  FEFaceValues<3> fe_values(fe, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
  std::vector<DofNumber> local_dof_indices(fe.n_dofs_per_cell());
  for (DofHandler3D::active_cell_iterator cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    for(unsigned int face = 0; face < 6; face++) {
      if(cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == 5) {
        fe_values.reinit(cell, face);
        cell->get_dof_indices(local_dof_indices);
        const std::vector<DofNumber> global_dof_indices = transform_local_to_global_dofs(local_dof_indices);
        for(unsigned int q_index = 0; q_index < quadrature_formula.size(); q_index++) {
          const double JxW = fe_values.JxW(q_index);
          Position p = fe_values.quadrature_point(q_index);
          p[2] -= 2 * FLOATING_PRECISION;
          Tensor<1,3,ComplexNumber> mode_conjugate;
          for(unsigned int i = 0; i < 3; i++) {
            mode_conjugate[i] = std::conj(GlobalParams.source_field->value(p, i));
            mode_norm += std::norm(mode_conjugate[i]) * JxW;
          }
          for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++) {
            const Tensor<1,3,ComplexNumber> shape_value = fe_values[fe_field].value(i, q_index);
            const ComplexNumber entry = (mode_conjugate * shape_value) * JxW;
            in_rhs->add(1, &global_dof_indices[i], &entry);
          }
        }
      }
    }
  }
  return mode_norm;
}

std::vector<ComplexNumber> InnerDomain::compute_shape_gradient_integrals(NumericVectorLocal & in_solution, NumericVectorLocal & in_adjoint) {
  const unsigned int n_shape_dofs = GlobalSpaceTransformation->n_free_dofs();
  std::vector<ComplexNumber> ret(n_shape_dofs, ComplexNumber(0, 0));
  const double step = 1e-5;
  QGauss<3> quadrature_formula(GlobalParams.Nedelec_element_order + 2);
  const FEValuesExtractors::Vector fe_field(0);
  FEValues<3> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points);
  std::vector<DofNumber> local_dof_indices(fe.n_dofs_per_cell());
  const double kappa_2 = Geometry.kappa_2();
  const bool has_taper = (GlobalParams.Signal_coupling_method == SignalCouplingMethod::Tapering);
  std::unique_ptr<ExactSolution> mode;
  if(has_taper) {
    mode = std::make_unique<ExactSolution>();
  }
  for (DofHandler3D::active_cell_iterator cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    for(unsigned int q_index = 0; q_index < quadrature_formula.size(); q_index++) {
      Position p = fe_values.quadrature_point(q_index);
      const double JxW = fe_values.JxW(q_index);
      Tensor<1,3,ComplexNumber> u, curl_u, lambda, curl_lambda;
      for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++) {
        const Tensor<1,3,ComplexNumber> value = fe_values[fe_field].value(i, q_index);
        const Tensor<1,3,ComplexNumber> curl = fe_values[fe_field].curl(i, q_index);
        u += value * in_solution[local_dof_indices[i]];
        curl_u += curl * in_solution[local_dof_indices[i]];
        lambda += value * in_adjoint[local_dof_indices[i]];
        curl_lambda += curl * in_adjoint[local_dof_indices[i]];
      }
      const double eps = Geometry.math_coordinate_in_waveguide(p) ? GlobalParams.Epsilon_R_in_waveguide : GlobalParams.Epsilon_R_outside_waveguide;
      const bool in_taper = has_taper && p[2] < GlobalParams.tapering_max_z && p[2] >= GlobalParams.tapering_min_z;
      Tensor<1,3,ComplexNumber> mode_value, mode_curl;
      double d_psi_dz = 0;
      if(in_taper) {
        const double taper_length = GlobalParams.tapering_max_z - GlobalParams.tapering_min_z;
        const double t = (p[2] - GlobalParams.tapering_min_z) / taper_length;
        d_psi_dz = 6.0 * t * (1.0 - t) / taper_length;
        mode_value = mode->val(p);
        mode_curl = mode->curl(p);
      }
      for(unsigned int j = 0; j < n_shape_dofs; j++) {
        // Central difference of the transformation tensor and its inverse with respect to shape parameter j.
        const double old_value = GlobalSpaceTransformation->get_free_dof(j);
        GlobalSpaceTransformation->set_free_dof(j, old_value + step);
        const Tensor<2,3,double> tensor_plus = GlobalSpaceTransformation->get_Space_Transformation_Tensor(p);
        GlobalSpaceTransformation->set_free_dof(j, old_value - step);
        const Tensor<2,3,double> tensor_minus = GlobalSpaceTransformation->get_Space_Transformation_Tensor(p);
        GlobalSpaceTransformation->set_free_dof(j, old_value);
        const Tensor<2,3,ComplexNumber> d_tensor = (tensor_plus - tensor_minus) / (2 * step);
        const Tensor<2,3,ComplexNumber> d_inverse = (invert(tensor_plus) - invert(tensor_minus)) / (2 * step);
        // lambda^T dA/dp u
        ComplexNumber integrand = (d_inverse * curl_u) * curl_lambda - kappa_2 * eps * ((d_tensor * u) * lambda);
        if(in_taper) {
          // - lambda^T db/dp, see evaluate_taper_source_at in the assembly.
          const Tensor<1,3,ComplexNumber> a = d_inverse * mode_curl;
          Tensor<1,3,ComplexNumber> d_source_for_values;
          d_source_for_values[0] = - d_psi_dz * a[1];
          d_source_for_values[1] =   d_psi_dz * a[0];
          Tensor<1,3,ComplexNumber> grad_psi_cross_E;
          grad_psi_cross_E[0] = - d_psi_dz * mode_value[1];
          grad_psi_cross_E[1] =   d_psi_dz * mode_value[0];
          const Tensor<1,3,ComplexNumber> d_source_for_curls = d_inverse * grad_psi_cross_E;
          integrand -= d_source_for_values * lambda + d_source_for_curls * curl_lambda;
        }
        ret[j] += integrand * JxW;
      }
    }
  }
  return ret;
}

void InnerDomain::write_field_slices(const NumericVectorLocal & in_solution, const std::string & in_name) {
  const double plane_coordinates[3] = {
    0.5 * (Geometry.global_x_range.first + Geometry.global_x_range.second),
    0.5 * (Geometry.global_y_range.first + Geometry.global_y_range.second),
    0.5 * (Geometry.global_z_range.first + Geometry.global_z_range.second)};
  const std::string plane_names[3] = {"x", "y", "z"};
  std::ofstream files[3];
  for(unsigned int plane = 0; plane < 3; plane++) {
    files[plane].open(GlobalOutputManager.get_numbered_filename(in_name + "_" + plane_names[plane], GlobalParams.MPI_Rank, "txt"));
    files[plane] << "# x y z x_physical y_physical z_physical re_Ex im_Ex re_Ey im_Ey re_Ez im_Ez in_core" << std::endl;
  }
  QMidpoint<3> quadrature_formula;
  const FEValuesExtractors::Vector fe_field(0);
  FEValues<3> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);
  std::vector<DofNumber> local_dof_indices(fe.n_dofs_per_cell());
  for (DofHandler3D::active_cell_iterator cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    // The cell is cut by a plane if the plane lies between its lower (inclusive) and upper (exclusive) face.
    bool is_cut[3];
    bool any = false;
    for(unsigned int plane = 0; plane < 3; plane++) {
      const double lower = cell->vertex(0)[plane];
      const double upper = cell->vertex(7)[plane];
      is_cut[plane] = (lower <= plane_coordinates[plane] + FLOATING_PRECISION) && (plane_coordinates[plane] + FLOATING_PRECISION < upper);
      any = any || is_cut[plane];
    }
    if(!any) continue;
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    Tensor<1,3,ComplexNumber> value;
    for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++) {
      value += fe_values[fe_field].value(i, 0) * in_solution[local_dof_indices[i]];
    }
    const Position p = fe_values.quadrature_point(0);
    const Position p_physical = GlobalSpaceTransformation->math_to_phys(p);
    for(unsigned int plane = 0; plane < 3; plane++) {
      if(is_cut[plane]) {
        files[plane] << p[0] << " " << p[1] << " " << p[2] << " " << p_physical[0] << " " << p_physical[1] << " " << p_physical[2];
        for(unsigned int component = 0; component < 3; component++) {
          files[plane] << " " << value[component].real() << " " << value[component].imag();
        }
        files[plane] << " " << (Geometry.math_coordinate_in_waveguide(p) ? 1 : 0) << std::endl;
      }
    }
  }
}

void InnerDomain::write_signal_along_z(const NumericVectorLocal & in_solution, const std::string & in_name) {
  // The field is evaluated at the centers of the cell faces z = const. There the components E_x and E_y, which
  // carry the mode, are tangential and therefore well defined. At the cell centers the lowest order elements
  // average the values of two faces, which reduces the amplitude of a wave by cos(beta h / 2) on coarse meshes.
  // Every face is counted once: each cell contributes its lower face, the last layer also its upper face.
  std::map<long, std::array<double, 6>> layers; // z, Re S, Im S, |m|^2, |E|^2 in the core, |E|^2
  const Quadrature<3> quadrature_formula(std::vector<Point<3>>{Point<3>(0.5, 0.5, 0.0), Point<3>(0.5, 0.5, 1.0)});
  const FEValuesExtractors::Vector fe_field(0);
  FEValues<3> fe_values(fe, quadrature_formula, update_values | update_quadrature_points);
  std::vector<DofNumber> local_dof_indices(fe.n_dofs_per_cell());
  Vector<ComplexNumber> mode(3);
  for (DofHandler3D::active_cell_iterator cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    for(unsigned int q = 0; q < 2; q++) {
      const Position p = fe_values.quadrature_point(q);
      if(q == 1 && std::abs(p[2] - Geometry.global_z_range.second) > FLOATING_PRECISION) continue;
      Tensor<1,3,ComplexNumber> value;
      for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++) {
        value += fe_values[fe_field].value(i, q) * in_solution[local_dof_indices[i]];
      }
      GlobalParams.source_field->vector_value(p, mode);
      ComplexNumber overlap(0, 0);
      double mode_norm = 0, energy = 0;
      for(unsigned int component = 0; component < 3; component++) {
        overlap += value[component] * std::conj(mode[component]);
        mode_norm += std::norm(mode[component]);
        energy += std::norm(value[component]);
      }
      const long key = std::lround(p[2] * 1e6);
      if(layers.find(key) == layers.end()) {
        layers[key] = {{p[2], 0, 0, 0, 0, 0}};
      }
      std::array<double, 6> & layer = layers[key];
      layer[1] += overlap.real();
      layer[2] += overlap.imag();
      layer[3] += mode_norm;
      if(Geometry.math_coordinate_in_waveguide(p)) layer[4] += energy;
      layer[5] += energy;
    }
  }
  std::ofstream out(GlobalOutputManager.get_numbered_filename(in_name, GlobalParams.MPI_Rank, "txt"));
  out.precision(10);
  out << "# z re_overlap im_overlap mode_norm energy_in_core energy_total (sums over the centers of the cell faces z = const)" << std::endl;
  for(const auto & layer : layers) {
    for(unsigned int i = 0; i < 6; i++) out << (i > 0 ? " " : "") << layer.second[i];
    out << std::endl;
  }
}
