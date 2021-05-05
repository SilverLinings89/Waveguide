#include "./PMLSurface.h"
#include <deal.II/base/geometry_info.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include "../Core/GlobalObjects.h"
#include "../Core/NumericProblem.h"
#include "../Helpers/staticfunctions.h"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <mpi.h>
#include <string>
#include "./BoundaryCondition.h"
#include "PMLMeshTransformation.h"

PMLSurface::PMLSurface(unsigned int surface, unsigned int in_level, DofNumber in_first_own_index)
  : BoundaryCondition(surface, in_level, Geometry.surface_extremal_coordinate[surface], in_first_own_index),
  fe_nedelec(GlobalParams.Nedelec_element_order) {
     constraints_made = false; 
     mesh_is_transformed = false;
}

PMLSurface::~PMLSurface() {}

void PMLSurface::prepare_mesh() {
    Triangulation<3> temp_tria;
    dealii::GridGenerator::extrude_triangulation(Geometry.surface_meshes[b_id], GlobalParams.PML_N_Layers, GlobalParams.PML_thickness, temp_tria);
    outer_boundary_id = b_id;
    dealii::Tensor<1, 3> shift_vector;
    for(unsigned int i = 0; i < 3; i++) {
        shift_vector[i] = 0;
    }
    if(b_id % 2 == 0) {
      inner_boundary_id = b_id + 1;
    } else {
      inner_boundary_id = b_id - 1;
    }
    switch (b_id)
    {
    case 0:
      dealii::GridTools::transform(Transform_5_to_0, temp_tria);
      shift_vector[0] = additional_coordinate - GlobalParams.PML_thickness;
      break;
    case 1:
      dealii::GridTools::transform(Transform_5_to_1, temp_tria);
      shift_vector[0] = additional_coordinate;
      break;
    case 2:
      dealii::GridTools::transform(Transform_5_to_2, temp_tria);
      shift_vector[1] = additional_coordinate - GlobalParams.PML_thickness;
      break;
    case 3:
      dealii::GridTools::transform(Transform_5_to_3, temp_tria);
      shift_vector[1] = additional_coordinate;
      break;
    case 4:
      dealii::GridTools::transform(Transform_5_to_4, temp_tria);
      shift_vector[2] = additional_coordinate - GlobalParams.PML_thickness;
      break;
    case 5:
      // dont need to transform the mesh. transformation in this case is identity.
      shift_vector[2] = additional_coordinate;
      break;
    
    default:
      break;
    }
    dealii::GridTools::shift(shift_vector, temp_tria);
    if(b_id % 2 == 0) {
      fix_apply_negative_Jacobian_transformation(&temp_tria);
    }
    triangulation = reforge_triangulation(&temp_tria);
    compute_coordinate_ranges();
    transformation = PMLMeshTransformation(x_range, y_range, z_range, additional_coordinate, b_id/2, Geometry.levels[level].is_surface_truncated);
    GridTools::transform(transformation, triangulation);
    mesh_is_transformed = true;
}

unsigned int PMLSurface::cells_for_boundary_id(unsigned int boundary_id) {
    unsigned int ret = 0;
    for(auto it = triangulation.begin(); it!= triangulation.end(); it++) {
        if(it->at_boundary(boundary_id)) {
            ret++;
        }
    }
    return ret;
}

void PMLSurface::init_fe() {
    dof_h_nedelec.initialize(triangulation, fe_nedelec);
    dof_h_nedelec.distribute_dofs(fe_nedelec);
    dof_counter = dof_h_nedelec.n_dofs();
    sort_dofs();
    set_boundary_ids();
}

bool PMLSurface::is_position_at_boundary(Position in_p, BoundaryId in_bid) {
  Position p = in_p;
  if(mesh_is_transformed) {
    p = transformation.undo_transform(in_p);
  }
  bool is_at_x_interface = std::abs(p[0] - x_range.first) < FLOATING_PRECISION || std::abs(p[0] - x_range.second) < FLOATING_PRECISION;
  bool is_at_y_interface = std::abs(p[1] - y_range.first) < FLOATING_PRECISION || std::abs(p[1] - y_range.second) < FLOATING_PRECISION;
  bool is_at_z_interface = std::abs(p[2] - z_range.first) < FLOATING_PRECISION || std::abs(p[2] - z_range.second) < FLOATING_PRECISION;
  if(!is_at_x_interface && !is_at_y_interface && !is_at_z_interface) {
    return false; // Position is at none of the interfaces
  }
  switch (in_bid)
  {
  case 0:
    if(p[0] == x_range.first) return true;
    break;
  case 1:
    if(p[0] == x_range.second) return true;
    break;
  case 2:
    if(p[1] == y_range.first) return true;
    break;
  case 3:
    if(p[1] == y_range.second) return true;
    break;
  case 4:
    if(p[2] == z_range.first) return true;
    break;
  case 5:
    if(p[2] == z_range.second) return true;
    break;
  default:
    break;
  }
  return false;
}

bool PMLSurface::is_point_at_boundary(Position2D, BoundaryId) {
  return false;
}

void PMLSurface::identify_corner_cells() {
  
}

void PMLSurface::initialize() {
  prepare_mesh();
  init_fe();
  prepare_id_sets_for_boundaries();
  make_inner_constraints();
}

void PMLSurface::sort_dofs() {
  triangulation.clear_user_flags();
  std::vector<std::pair<DofNumber, Position>> current;
  std::vector<types::global_dof_index> local_line_dofs(fe_nedelec.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe_nedelec.dofs_per_face);
  std::set<DofNumber> face_set;
  std::vector<DofNumber> local_cell_dofs(fe_nedelec.dofs_per_cell);
  std::set<DofNumber> cell_set;
  auto cell = dof_h_nedelec.begin_active();
  auto endc = dof_h_nedelec.end();
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
  std::sort(current.begin(), current.end(), compareDofBaseData);
  std::vector<unsigned int> new_numbering;
  new_numbering.resize(current.size());
  for (unsigned int i = 0; i < current.size(); i++) {
    new_numbering[current[i].first] = i;
  }
  dof_h_nedelec.renumber_dofs(new_numbering);
}

void PMLSurface::prepare_id_sets_for_boundaries(){
  for(unsigned int i = 0; i < 6; i++) {
    face_ids_by_boundary_id[i].clear();
    edge_ids_by_boundary_id[i].clear();
  }
  for(auto it = dof_h_nedelec.begin(); it != dof_h_nedelec.end(); it++) {
    if(it->at_boundary()) {
      for(unsigned int face = 0; face < 6; face++) {
        if(it->face(face)->at_boundary()) {
          for(unsigned int current_boundary_id = 0; current_boundary_id < 6; current_boundary_id++) {
            if(is_position_at_boundary(it->face(face)->center(), current_boundary_id)) {
              face_ids_by_boundary_id[current_boundary_id].insert(it->face_index(face));
              for(unsigned int edge = 0; edge < 4; edge++) {
                edge_ids_by_boundary_id[current_boundary_id].insert(it->face(face)->line_index(edge));
              }
            }
          }
        }
      }
    }
  }
}

std::vector<InterfaceDofData> PMLSurface::get_dof_association_by_boundary_id(unsigned int in_bid) {
  std::vector<InterfaceDofData> ret;
  for(auto it = dof_h_nedelec.begin(); it != dof_h_nedelec.end(); it++) {
    it->clear_user_flag();
    for(auto face = 0; face < 6; face++) {
      it->face(face)->clear_user_flag();
      for(unsigned int line = 0; line < 4; line++) {
        it->face(face)->line(line)->clear_user_flag();
      }
    }
  }
  dealii::Vector<ComplexNumber> base_vector(dof_counter);
  for(auto it = dof_h_nedelec.begin(); it != dof_h_nedelec.end(); it++) {
    if(it->at_boundary()) {
      for(unsigned int face = 0; face < 6; face++) {
        if(face_ids_by_boundary_id[in_bid].contains(it->face_index(face)) && !it->face(face)->user_flag_set()) {
          std::vector<unsigned int> face_dof_indices(fe_nedelec.n_dofs_per_face());
          it->face(face)->get_dof_indices(face_dof_indices);
          for(unsigned int line = 0; line < 4; line ++) {
            std::vector<DofNumber> line_dofs(fe_nedelec.n_dofs_per_line());
            it->face(face)->line(line)->get_dof_indices(line_dofs);
            for(unsigned int i = 0; i < fe_nedelec.n_dofs_per_line(); i++) {
              for(unsigned int j = 0; j < face_dof_indices.size(); j++) {
                if(face_dof_indices[j] == line_dofs[i]) {
                  face_dof_indices.erase(face_dof_indices.begin() + j);
                }
              }
            }
            if(!it->face(face)->line(line)->user_flag_set()) {
              if(edge_ids_by_boundary_id[in_bid].contains(it->face(face)->line_index(line))) {
                for(unsigned int i = 0; i < fe_nedelec.n_dofs_per_line(); i++) {
                  InterfaceDofData entry;
                  entry.index = line_dofs[i];
                  entry.base_point = it->face(face)->line(line)->center();
                  entry.order = i;
                  ret.push_back(entry);
                }
              }
              it->face(face)->line(line)->set_user_flag();
            }
          }
          if(GlobalParams.Nedelec_element_order > 0) {
            if(face_ids_by_boundary_id[in_bid].contains(it->face_index(face)) && face_dof_indices.size() > 0) {
              for(unsigned int f = 0; f < face_dof_indices.size(); f++) {
                InterfaceDofData entry;
                entry.index = face_dof_indices[f];
                entry.base_point = it->face(face)->center();
                entry.order = f;
                ret.push_back(entry);
              }
              it->face(face)->set_user_flag();
            }
          }
        }
      }
    }
  }
  ret.shrink_to_fit();
  std::sort(ret.begin(), ret.end(), compareDofBaseDataAndOrientation);
  std::cout << "Surface first dof: " << Geometry.levels[level].surface_first_dof[b_id] << std::endl;
  shift_interface_dof_data(&ret, Geometry.levels[level].surface_first_dof[b_id]);
  return ret;
}

std::vector<InterfaceDofData> PMLSurface::get_dof_association() {
    return get_dof_association_by_boundary_id(inner_boundary_id);
}

DofCount PMLSurface::get_dof_count_by_boundary_id(BoundaryId in_bid) {
    return (get_dof_association_by_boundary_id(in_bid)).size();
}

double PMLSurface::fraction_of_pml_direction(Position in_p) {
  return std::abs(in_p[b_id/2]-additional_coordinate) / GlobalParams.PML_thickness;
}

dealii::Tensor<2,3,ComplexNumber> PMLSurface::get_pml_tensor_epsilon(Position in_p) {
    dealii::Tensor<2,3,ComplexNumber> ret = get_pml_tensor(in_p);
    ret *= Geometry.get_epsilon_for_point(in_p);
    ret *= GlobalParams.Omega * GlobalParams.Omega;
    return ret;
}

dealii::Tensor<2,3,ComplexNumber> PMLSurface::get_pml_tensor_mu(Position in_p) {
    dealii::Tensor<2,3,ComplexNumber> ret = invert(get_pml_tensor(in_p));
    return ret;
}

dealii::Tensor<2,3,ComplexNumber> PMLSurface::get_pml_tensor(Position in_p) {
  dealii::Tensor<2,3,ComplexNumber> ret;
  double fraction = fraction_of_pml_direction(in_p);
  ComplexNumber part_a = {1 , std::pow(fraction, GlobalParams.PML_skaling_order) * GlobalParams.PML_Sigma_Max};
  for(unsigned int i = 0; i < 3; i++) {
      for(unsigned int j = 0; j < 3; j++) {
          ret[i][j] = 0;
      }
  }
  for(unsigned int i = 0; i < 3; i++) {
      if(i == b_id/2) {
          ret[i][i] = 1.0 / part_a;
      } else {
          ret[i][i] = part_a;
      }
  }
  return ret;
}

void PMLSurface::make_inner_constraints() {
    IndexSet is(dof_counter);
    is.add_range(0, dof_counter);
    constraints.reinit(is);
    dealii::DoFTools::make_zero_boundary_constraints(dof_h_nedelec, outer_boundary_id, constraints);
    constraints_made = true;
}

void PMLSurface::copy_constraints(dealii::AffineConstraints<ComplexNumber> * in_constraints, unsigned int shift) {
    constraints.shift(shift);
    in_constraints->merge(constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins, true);
    constraints.shift(-shift);
}

struct CellwiseAssemblyDataPML {
  QGauss<3> quadrature_formula; 
  FEValues<3> fe_values;
  std::vector<Position> quadrature_points;
  const unsigned int dofs_per_cell;
  const unsigned int n_q_points;
  FullMatrix<ComplexNumber> cell_matrix;
  Vector<ComplexNumber> cell_rhs;
  std::vector<DofNumber> local_dof_indices;
  DofHandler3D::active_cell_iterator cell;
  DofHandler3D::active_cell_iterator end_cell;
  const FEValuesExtractors::Vector fe_field;
  CellwiseAssemblyDataPML(dealii::FE_NedelecSZ<3> * fe, DofHandler3D * dof_handler):
  quadrature_formula(GlobalParams.Nedelec_element_order + 2),
  fe_values(*fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values |
                            update_quadrature_points),
  dofs_per_cell(fe->dofs_per_cell),
  n_q_points(quadrature_formula.size()),
  cell_matrix(dofs_per_cell,dofs_per_cell),
  cell_rhs(dofs_per_cell),
  local_dof_indices(dofs_per_cell),
  fe_field(0)
  {
    cell_rhs = 0;
    cell = dof_handler->begin_active();
    end_cell = dof_handler->end();
  };

  Position get_position_for_q_index(unsigned int q_index) {
      return quadrature_points[q_index];
  }

  void prepare_for_current_q_index(unsigned int q_index, dealii::Tensor<2, 3, ComplexNumber> epsilon, dealii::Tensor<2,3,ComplexNumber> mu_inverse) {
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

        cell_matrix[i][j] += I_Curl * (mu_inverse * Conjugate_Vector(J_Curl))* JxW
            - ( ( epsilon *  I_Val * Conjugate_Vector(J_Val)) * JxW);
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

void PMLSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints) {
  auto it = dof_h_nedelec.begin_active();
  auto end = dof_h_nedelec.end();
  std::vector<DofNumber> local_indices(fe_nedelec.n_dofs_per_cell());
  for (; it != end; ++it) {
    it->get_dof_indices(local_indices);
    for(unsigned int i = 0; i < local_indices.size(); i++) {
      local_indices[i] += first_own_dof;
    }
    constraints->add_entries_local_to_global(local_indices, *in_dsp);
  }
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* , dealii::AffineConstraints<ComplexNumber> *){
    // NOT IMPLEMENTED
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints){
  CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_h_nedelec);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
      cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
      for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
          cell_data.local_dof_indices[i] += first_own_dof;
      }
      cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
      cell_data.fe_values.reinit(cell_data.cell);
      cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
      std::vector<types::global_dof_index> input_dofs(fe_nedelec.dofs_per_line);
      IndexSet input_dofs_local_set(fe_nedelec.dofs_per_cell);
      std::vector<Position> input_dof_centers(fe_nedelec.dofs_per_cell);
      std::vector<Tensor<1, 3, double>> input_dof_dirs(fe_nedelec.dofs_per_cell);
      cell_data.cell_matrix = 0;
      for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
          Position pos = cell_data.get_position_for_q_index(q_index);
          dealii::Tensor<2,3,ComplexNumber> epsilon = get_pml_tensor_epsilon(pos);
          dealii::Tensor<2,3,ComplexNumber> mu = get_pml_tensor_mu(pos);
          cell_data.prepare_for_current_q_index(q_index, epsilon, mu);
      }
      constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, false);
  }
  matrix->compress(dealii::VectorOperation::add);
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, dealii::AffineConstraints<ComplexNumber> *constraints){
  CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_h_nedelec);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
        cell_data.local_dof_indices[i] += first_own_dof;
    }
    cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
    cell_data.fe_values.reinit(cell_data.cell);
    cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
    std::vector<types::global_dof_index> input_dofs(fe_nedelec.dofs_per_line);
    IndexSet input_dofs_local_set(fe_nedelec.dofs_per_cell);
    std::vector<Position> input_dof_centers(fe_nedelec.dofs_per_cell);
    std::vector<Tensor<1, 3, double>> input_dof_dirs(fe_nedelec.dofs_per_cell);
    cell_data.cell_matrix = 0;
    for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
      Position pos = cell_data.get_position_for_q_index(q_index);
      dealii::Tensor<2,3,ComplexNumber> epsilon = get_pml_tensor_epsilon(pos);
      dealii::Tensor<2,3,ComplexNumber> mu = get_pml_tensor_mu(pos);
      cell_data.prepare_for_current_q_index(q_index, epsilon, mu);
    }
    constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, false);
  }
  matrix->compress(dealii::VectorOperation::add);
}

void PMLSurface::compute_coordinate_ranges() {
  x_range.first = 100000.0;
  y_range.first = 100000.0;
  z_range.first = 100000.0;
  x_range.second = -100000.0;
  y_range.second = -100000.0;
  z_range.second = -100000.0;
  for(auto it = triangulation.begin(); it != triangulation.end(); it++) {
    for(unsigned int i = 0; i < 6; i++) {
      Position p = it->face(i)->center();
      if(p[0] < x_range.first) {
        x_range.first = p[0];
      }
      if(p[0] > x_range.second) {
        x_range.second = p[0];
      }
      if(p[1] < y_range.first) {
        y_range.first = p[1];
      }
      if(p[1] > y_range.second) {
        y_range.second = p[1];
      }
      if(p[2] < z_range.first) {
        z_range.first = p[2];
      }
      if(p[2] > z_range.second) {
        z_range.second = p[2];
      }
    }
  }
}

void PMLSurface::set_boundary_ids() {
  for(auto it = triangulation.begin(); it != triangulation.end(); it++) {
    for(unsigned int face = 0; face < 6; face ++) {
      if(it->face(face)->at_boundary()) {
        Position p = it->face(face)->center();
        if(std::abs(p[0] - x_range.first) < 0.001) {
          it->face(face)->set_all_boundary_ids(0);
        }
        if(std::abs(p[0] - x_range.second) < 0.001) {
          it->face(face)->set_all_boundary_ids(1);
        }
        if(std::abs(p[1] - y_range.first) < 0.001) {
          it->face(face)->set_all_boundary_ids(2);
        }
        if(std::abs(p[1] - y_range.second) < 0.001) {
          it->face(face)->set_all_boundary_ids(3);
        }
        if(std::abs(p[2] - z_range.first) < 0.001) {
          it->face(face)->set_all_boundary_ids(4);
        }
        if(std::abs(p[2] - z_range.second) < 0.001) {
          it->face(face)->set_all_boundary_ids(5);
        }
      }
    }   
  }
}

Position invert_z(Position in_p) {
  Position ret = in_p;
  ret[2] = -ret[2];
  return ret;
}

double min_z_center_in_triangulation(dealii::Triangulation<3, 3> & in_tria) {
  double ret = 100000000;
  for(auto it = in_tria.begin(); it != in_tria.end(); it++) {
    if(it->at_boundary()) {
      if(it->center()[2] < ret) {
        ret = it->center()[2];
      }
    }
  }
  return ret;
}

void PMLSurface::fix_apply_negative_Jacobian_transformation(dealii::Triangulation<3> * in_tria) {
  double min_z_before = min_z_center_in_triangulation(*in_tria);
  GridTools::transform(invert_z, *in_tria);
  double min_z_after = min_z_center_in_triangulation(*in_tria);
  Tensor<1,3> shift;
  shift[0] = 0;
  shift[1] = 0;
  shift[2] = min_z_before - min_z_after;
  GridTools::shift(shift, *in_tria);
}

void PMLSurface::output_results(const dealii::Vector<ComplexNumber> & in_data, std::string in_filename) {
  print_info("PMSurface::output_results()", "Start");
  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_h_nedelec);
  data_out.add_data_vector(in_data, "Solution");
  const std::string filename = GlobalOutputManager.get_numbered_filename(in_filename + "-" + std::to_string(b_id) + "-", GlobalParams.MPI_Rank, "vtu");
  std::ofstream outputvtu(filename);
  data_out.build_patches();
  data_out.write_vtu(outputvtu);
  print_info("PMSurface::output_results()", "End");
}

void PMLSurface::fill_sparsity_pattern_for_neighbor(const BoundaryId in_bid, const unsigned int partner_index, dealii::AffineConstraints<ComplexNumber> * constraints, dealii::DynamicSparsityPattern * dsp) {
  unsigned int other_first_dof_index = first_own_dof;
  MPI_Sendrecv_replace(&other_first_dof_index, 1, MPI_UNSIGNED, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );
  std::vector<SurfaceCellData> surface_cell_data;
  std::vector<DofNumber> cell_dofs(fe_nedelec.dofs_per_cell);
  for(auto cell = dof_h_nedelec.begin(); cell != dof_h_nedelec.end(); cell++) {
    for(unsigned int face_index = 0; face_index < 6; face_index++) {
      if(is_position_at_boundary(cell->face(face_index)->center(), in_bid)) {
        cell->get_dof_indices(cell_dofs);
        SurfaceCellData new_cell_data;
        new_cell_data.surface_face_center = cell->face(face_index)->center();
        new_cell_data.dof_numbers = cell_dofs;
        surface_cell_data.push_back(new_cell_data);
      }
    }
  }
  
  std::sort(surface_cell_data.begin(), surface_cell_data.end(), compareSurfaceCellData);

  const unsigned int n_entries = surface_cell_data.size() * fe_nedelec.dofs_per_cell;
  unsigned int * face_indices = new unsigned int [n_entries];

  for(unsigned int i = 0; i < surface_cell_data.size(); i++) {
    for(unsigned int j = 0; j < fe_nedelec.dofs_per_cell; j++) {
      face_indices[i * fe_nedelec.dofs_per_cell + j] = surface_cell_data[i].dof_numbers[j] + first_own_dof;
    }
  }
  
  MPI_Sendrecv_replace(face_indices, n_entries, MPI_UNSIGNED, partner_index, 0, partner_index, 0, MPI_COMM_WORLD, 0 );

  std::vector<SurfaceCellData> other_surface_data;
  for(unsigned int i = 0; i < surface_cell_data.size(); i++) {
    SurfaceCellData scd;
    scd.surface_face_center = surface_cell_data[i].surface_face_center;
    scd.dof_numbers = std::vector<unsigned int>(fe_nedelec.dofs_per_cell);
    for(unsigned int j = 0; j < fe_nedelec.dofs_per_cell; j++) {
      scd.dof_numbers[j] = face_indices[i * fe_nedelec.dofs_per_cell + j];
    }
    other_surface_data.push_back(scd);
  }
  
  for(unsigned int i = 0; i < surface_cell_data.size(); i++) {
    std::vector<DofNumber> dof_numbers;
    for(unsigned int j = 0; j < fe_nedelec.dofs_per_cell; j++) {
      dof_numbers.push_back(surface_cell_data[i].dof_numbers[j]);
      dof_numbers.push_back(other_surface_data[i].dof_numbers[j]);
    }
    constraints->add_entries_local_to_global(dof_numbers, *dsp);
  }
}

void PMLSurface::fill_sparsity_pattern_for_boundary_id(const BoundaryId in_bid, dealii::AffineConstraints<ComplexNumber> * constraints, dealii::DynamicSparsityPattern * dsp) {
  std::vector<unsigned int> dof_numbers(fe_nedelec.dofs_per_cell);
  for(auto it = dof_h_nedelec.begin(); it != dof_h_nedelec.end(); it ++) {
    bool is_at_boundary = false;
    for(unsigned int i = 0; i < dealii::GeometryInfo<3>::faces_per_cell; i++) {
      if(is_position_at_boundary(it->face(i)->center(), in_bid)) {
        is_at_boundary = true;
        break;
      }
    }
    if(is_at_boundary) {
      it->get_dof_indices(dof_numbers);
      for(unsigned int i = 0; i <fe_nedelec.dofs_per_cell; i++) {
        dof_numbers[i] += first_own_dof;
      }
      constraints->add_entries_local_to_global(dof_numbers, *dsp);
    }
  }
}

void PMLSurface::make_surface_constraints(dealii::AffineConstraints<ComplexNumber> * constraints) {
    std::vector<InterfaceDofData> own_dof_indices = get_dof_association();
    std::vector<InterfaceDofData> inner_dof_indices = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(b_id, level);
    dealii::AffineConstraints<ComplexNumber> new_constraints = get_affine_constraints_for_InterfaceData(own_dof_indices, inner_dof_indices, Geometry.levels[level].n_total_level_dofs);
    constraints->merge(new_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
}

void PMLSurface::make_edge_constraints(dealii::AffineConstraints<ComplexNumber> * constraints, BoundaryId other_boundary) {
    std::vector<InterfaceDofData> inner_dof_indices = Geometry.levels[level].surfaces[other_boundary]->get_dof_association_by_boundary_id(b_id);
    std::vector<InterfaceDofData> own_dof_indices = get_dof_association_by_boundary_id(other_boundary);
    std::cout << "In make edge constraints: length from surface: " << inner_dof_indices.size() << " other length: " << own_dof_indices.size()<<std::endl;
    dealii::AffineConstraints<ComplexNumber> new_constraints = get_affine_constraints_for_InterfaceData(inner_dof_indices, own_dof_indices, Geometry.levels[level].n_total_level_dofs);
    constraints->merge(new_constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::right_object_wins, true);
}
