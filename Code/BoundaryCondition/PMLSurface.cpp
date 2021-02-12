#include "./PMLSurface.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include "../Core/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <string>
#include "./BoundaryCondition.h"
#include "PMLMeshTransformation.h"


PMLSurface::PMLSurface(unsigned int in_bid, double in_additional_coordinate, dealii::Triangulation<2> & in_surf_tria)
    :BoundaryCondition(in_bid, in_additional_coordinate, in_surf_tria),
    fe_nedelec(GlobalParams.Nedelec_element_order)
{
    constraints_made = false;   
}

PMLSurface::~PMLSurface() {}

void PMLSurface::prepare_mesh() {
    dealii::GridGenerator::extrude_triangulation(surface_triangulation, GlobalParams.PML_N_Layers, GlobalParams.PML_thickness, triangulation);
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
      dealii::GridTools::transform(Transform_5_to_0, triangulation);
      shift_vector[0] = additional_coordinate - GlobalParams.PML_thickness;
      break;
    case 1:
      dealii::GridTools::transform(Transform_5_to_1, triangulation);
      shift_vector[0] = additional_coordinate;
      break;
    case 2:
      dealii::GridTools::transform(Transform_5_to_2, triangulation);
      shift_vector[1] = additional_coordinate - GlobalParams.PML_thickness;
      break;
    case 3:
      dealii::GridTools::transform(Transform_5_to_3, triangulation);
      shift_vector[1] = additional_coordinate;
      break;
    case 4:
      dealii::GridTools::transform(Transform_5_to_4, triangulation);
      shift_vector[2] = additional_coordinate - GlobalParams.PML_thickness;
      break;
    case 5:
      // dont need to transform the mesh. transformation in this case is identity.
      shift_vector[2] = additional_coordinate;
      break;
    
    default:
      break;
    }
    dealii::GridTools::shift(shift_vector, triangulation);
    if(b_id % 2 == 0) {
      fix_apply_negative_Jacobian_transformation();
    }
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
    compute_coordinate_ranges();
    set_boundary_ids();
}

bool PMLSurface::is_position_at_boundary(Position in_p, BoundaryId in_bid) {
  bool is_at_x_interface = std::abs(in_p[0] - x_range.first) < 0.00001 || std::abs(in_p[0] - x_range.second) < 0.00001;
  bool is_at_y_interface = std::abs(in_p[1] - y_range.first) < 0.00001 || std::abs(in_p[1] - y_range.second) < 0.00001;
  bool is_at_z_interface = std::abs(in_p[2] - z_range.first) < 0.00001 || std::abs(in_p[2] - z_range.second) < 0.00001;
  if(!is_at_x_interface && !is_at_y_interface && !is_at_z_interface) {
    return false; // Position is at none of the interfaces
  }
  switch (in_bid)
  {
  case 0:
    if(in_p[0] == x_range.first) return true;
    break;
  case 1:
    if(in_p[0] == x_range.second) return true;
    break;
  case 2:
    if(in_p[1] == y_range.first) return true;
    break;
  case 3:
    if(in_p[1] == y_range.second) return true;
    break;
  case 4:
    if(in_p[2] == z_range.first) return true;
    break;
  case 5:
    if(in_p[2] == z_range.second) return true;
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
  make_inner_constraints();
  prepare_id_sets_for_boundaries();
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
                  if(it->face(face)->line(i)->vertex_index(0) < it->face(face)->line(line)->vertex_index(1)) {
                    entry.orientation = get_orientation(it->face(face)->line(line)->vertex(1),it->face(face)->line(line)->vertex(0));
                  } else {
                    entry.orientation = get_orientation(it->face(face)->line(line)->vertex(0),it->face(face)->line(line)->vertex(1));
                  }
                  entry.position = it->face(face)->line(line)->center();
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
                entry.orientation = get_orientation(it->face(face)->vertex(0), it->face(face)->vertex(1));
                entry.position = it->face(face)->center();
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
  return ret;
}

std::vector<InterfaceDofData> PMLSurface::get_dof_association() {
    return get_dof_association_by_boundary_id(inner_boundary_id);
}

DofCount PMLSurface::get_dof_count_by_boundary_id(BoundaryId in_bid) {
    return (get_dof_association_by_boundary_id(in_bid)).size();
}

double PMLSurface::fraction_of_pml_direction(Position in_p) {
    double delta = 0;
    if(b_id == 0 || b_id == 1) {
        delta = std::abs(in_p[0]-additional_coordinate) / GlobalParams.PML_thickness;
    }
    if(b_id == 2 || b_id == 3) {
        delta = std::abs(in_p[1]-additional_coordinate) / GlobalParams.PML_thickness;
    }
    if(b_id == 4 || b_id == 5) {
        delta = std::abs(in_p[2]-additional_coordinate) / GlobalParams.PML_thickness;
    }
    return delta;
}

dealii::Tensor<2,3,ComplexNumber> PMLSurface::get_pml_tensor_epsilon(Position in_p) {
    dealii::Tensor<2,3,ComplexNumber> ret = get_pml_tensor(in_p);
    ret *= (Geometry.math_coordinate_in_waveguide(in_p))? GlobalParams.Epsilon_R_in_waveguide : GlobalParams.Epsilon_R_outside_waveguide;
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

void PMLSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, DofNumber shift, dealii::AffineConstraints<ComplexNumber> *constraints) {
  auto it = dof_h_nedelec.begin_active();
  auto end = dof_h_nedelec.end();
  std::vector<DofNumber> local_indices(fe_nedelec.n_dofs_per_cell());
  for (; it != end; ++it) {
    it->get_dof_indices(local_indices);
    for(unsigned int i = 0; i < local_indices.size(); i++) {
      local_indices[i] += shift;
    }
    constraints->add_entries_local_to_global(local_indices, *in_dsp);

  }
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* , dealii::IndexSet , std::array<bool, 6>, dealii::AffineConstraints<ComplexNumber> *){
    // NOT IMPLEMENTED
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* , DofNumber , std::array<bool, 6> , dealii::AffineConstraints<ComplexNumber> *){
    // NOT IMPLEMENTED
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed* rhs, dealii::IndexSet in_is,  std::array<bool, 6> surfaces_hsie,  dealii::AffineConstraints<ComplexNumber> *in_constraints){
  const unsigned int shift = in_is.nth_index_in_set(0);
  fill_matrix(matrix, rhs, shift, surfaces_hsie, in_constraints);
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints){
  setup_neighbor_couplings(surfaces_hsie);
  CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_h_nedelec);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
      cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
      for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
          cell_data.local_dof_indices[i] += shift;
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
          constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, false);
      }
  }
  matrix->compress(dealii::VectorOperation::add);
    reset_neighbor_couplings(surfaces_hsie);
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, dealii::IndexSet in_is, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *in_constraints){
    const unsigned int shift = in_is.nth_index_in_set(0);
    fill_matrix(matrix, rhs, shift, surfaces_hsie, in_constraints);
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *constraints){
  setup_neighbor_couplings(surfaces_hsie);
  CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_h_nedelec);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
        cell_data.local_dof_indices[i] += shift;
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
        constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, false);
    }
  }
  matrix->compress(dealii::VectorOperation::add);
  reset_neighbor_couplings(surfaces_hsie);
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

void PMLSurface::setup_neighbor_couplings(std::array<bool, 6> in_is_neighbor_truncated) {
  std::ofstream outfileb("grid_surface" + std::to_string(b_id) + "b.vtu");
  GridOut go;
  PMLMeshTransformation::set(x_range, y_range, z_range, additional_coordinate, b_id/2, in_is_neighbor_truncated);
  GridTools::transform(&(PMLMeshTransformation::transform), triangulation);
  go.write_vtu(triangulation, outfileb);
}

void PMLSurface::reset_neighbor_couplings(std::array<bool, 6> in_is_neighbor_truncated) {
  PMLMeshTransformation::set(x_range, y_range, z_range, additional_coordinate, b_id/2, in_is_neighbor_truncated);
  GridTools::transform(&(PMLMeshTransformation::undo_transform), triangulation);
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

void PMLSurface::fix_apply_negative_Jacobian_transformation() {
  double min_z_before = min_z_center_in_triangulation(triangulation);
  GridTools::transform(invert_z, triangulation);
  double min_z_after = min_z_center_in_triangulation(triangulation);
  Tensor<1,3> shift;
  shift[0] = 0;
  shift[1] = 0;
  shift[2] = min_z_before - min_z_after;
  GridTools::shift(shift, triangulation);
}