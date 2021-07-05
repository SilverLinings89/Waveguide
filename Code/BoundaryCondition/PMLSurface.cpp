#include "./PMLSurface.h"
#include <deal.II/base/geometry_info.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include "../GlobalObjects/GlobalObjects.h"
#include "../Core/InnerDomain.h"
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
     mesh_is_transformed = false;
     outer_boundary_id = surface;
     if(surface % 2 == 0) {
       inner_boundary_id = surface + 1;
     } else {
       inner_boundary_id = surface - 1;
     }
}

PMLSurface::~PMLSurface() {}

void PMLSurface::prepare_mesh() {
    Triangulation<3> tria;
    std::vector<unsigned int> repetitions;
    repetitions.push_back(GlobalParams.Cells_in_x);
    repetitions.push_back(GlobalParams.Cells_in_y);
    repetitions.push_back(GlobalParams.Cells_in_z);
    repetitions[b_id / 2] = GlobalParams.PML_N_Layers - 1;
    Position lower_ranges;
    lower_ranges[0] = Geometry.local_x_range.first;
    lower_ranges[1] = Geometry.local_y_range.first;
    lower_ranges[2] = Geometry.local_z_range.first;
    Position higher_ranges;
    higher_ranges[0] = Geometry.local_x_range.second;
    higher_ranges[1] = Geometry.local_y_range.second;
    higher_ranges[2] = Geometry.local_z_range.second;

    if(b_id % 2 == 0) {
      lower_ranges[b_id / 2] = additional_coordinate - GlobalParams.PML_thickness;
      higher_ranges[b_id / 2] = additional_coordinate;
    } else {
      lower_ranges[b_id / 2] = additional_coordinate;
      higher_ranges[b_id / 2] = additional_coordinate + GlobalParams.PML_thickness;
    }
    
    dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, lower_ranges , higher_ranges ,true);
    triangulation = reforge_triangulation(&tria);
    
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
}

bool PMLSurface::is_position_at_boundary(Position in_p, BoundaryId in_bid) {
  Position p = in_p;
  if(mesh_is_transformed) {
    p = transformation.undo_transform(in_p);
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
  }
  return false;
}

bool PMLSurface::is_point_at_boundary(Position2D, BoundaryId) {
  return false;
}

void PMLSurface::identify_corner_cells() {
  
}

void PMLSurface::validate_meshes() {
  std::array<unsigned int, 6> cells_per_surface;
  std::array<unsigned int, 6> dofs_per_surface;
  for(unsigned int i = 0; i < 6; i++) {
    cells_per_surface[i] = cells_for_boundary_id(i);
    dofs_per_surface[i] = get_dof_association_by_boundary_id(i).size();
  }
  unsigned int inner_count = cells_per_surface[b_id];
  unsigned int other_count = 0;
  if(b_id == 0 || b_id == 1) {
    other_count = cells_per_surface[2];
  } else {
    other_count = cells_per_surface[0];
  }
  bool correct = true;
  for(unsigned int i = 0; i < 6; i++) {
    if(i == b_id || are_opposing_sites(b_id , i)) {
      if(cells_per_surface[i] != inner_count) {
        correct = false;
      }
    } else {
      if(cells_per_surface[i] != other_count) {
        correct = false;
      }
    }
  }

  inner_count = dofs_per_surface[b_id];
  other_count = 0;
  if(b_id == 0 || b_id == 1) {
    other_count = dofs_per_surface[2];
  } else {
    other_count = dofs_per_surface[0];
  }
  for(unsigned int i = 0; i < 6; i++) {
    if(i == b_id || are_opposing_sites(b_id , i)) {
      if(dofs_per_surface[i] != inner_count) {
        correct = false;
      }
    } else {
      if(dofs_per_surface[i] != other_count) {
        correct = false;
      }
    }
  }
  if(!correct) {
    std::cout << "The validation of surface " << b_id << " failed." << std::endl;
    for(unsigned int i = 0; i < 6; i++) {
      std::cout << "surf " << i << ": " << cells_per_surface[i] << " and " << dofs_per_surface[i] << std::endl;
    }
  }
}

void PMLSurface::initialize() {
  prepare_mesh();
  init_fe();
  prepare_id_sets_for_boundaries();
  // validate_meshes();
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
  std::vector<types::global_dof_index> local_line_dofs(fe_nedelec.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe_nedelec.dofs_per_face);
  std::set<DofNumber> face_set;
  triangulation.clear_user_flags();
  for (auto cell : dof_h_nedelec.active_cell_iterators()) {
    if (cell->at_boundary(in_bid)) {
      bool found_one = false;
      for (unsigned int face = 0; face < 6; face++) {
        if (cell->face(face)->boundary_id() == in_bid && found_one) {
          print_info("PMLSurface::get_dof_association_by_boundary_id", "There was an error!", false, LoggingLevel::PRODUCTION_ALL);
        }
        if (cell->face(face)->boundary_id() == in_bid) {
          found_one = true;
          std::vector<DofNumber> face_dofs_indices(fe_nedelec.dofs_per_face);
          cell->face(face)->get_dof_indices(face_dofs_indices);
          face_set.clear();
          face_set.insert(face_dofs_indices.begin(), face_dofs_indices.end());
          std::vector<InterfaceDofData> cell_dofs_and_orientations_and_points;
          for (unsigned int i = 0; i < dealii::GeometryInfo<3>::lines_per_face; i++) {
            std::vector<DofNumber> line_dofs(fe_nedelec.dofs_per_line);
            cell->face(face)->line(i)->get_dof_indices(line_dofs);
            line_set.clear();
            line_set.insert(line_dofs.begin(), line_dofs.end());
            for(auto erase_it: line_set) {
              face_set.erase(erase_it);
            }
            if(!cell->face(face)->line(i)->user_flag_set()) {
              for (unsigned int j = 0; j < fe_nedelec.dofs_per_line; j++) {
                InterfaceDofData new_item;
                new_item.index = line_dofs[j] + first_own_dof;
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
            new_item.index = item + first_own_dof;
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

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* , Constraints *){
    // NOT IMPLEMENTED
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed* rhs, Constraints *constraints){
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

void PMLSurface::fill_matrix(dealii::SparseMatrix<ComplexNumber> * matrix, Constraints *constraints){
  CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_h_nedelec);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
      cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
      for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
          cell_data.local_dof_indices[i] += first_own_dof - Geometry.levels[level].inner_first_dof;
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
      constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.local_dof_indices,*matrix);
  }
  matrix->compress(dealii::VectorOperation::add);
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, Constraints *constraints){
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
        for(unsigned int i = 0; i < 6; i++) {
          if(is_position_at_boundary(p, i)) {
            it->face(face)->set_all_boundary_ids(i);
          }
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
  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_h_nedelec);
  data_out.add_data_vector(in_data, "Solution");
  const std::string filename = GlobalOutputManager.get_numbered_filename(in_filename + "-" + std::to_string(b_id) + "-", GlobalParams.MPI_Rank, "vtu");
  std::ofstream outputvtu(filename);
  data_out.build_patches();
  data_out.write_vtu(outputvtu);
}

void PMLSurface::make_surface_constraints(Constraints * in_constraints) {
    std::vector<InterfaceDofData> own_dof_indices = get_dof_association();
    std::vector<InterfaceDofData> inner_dof_indices = Geometry.inner_domain->get_surface_dof_vector_for_boundary_id_and_level(b_id, level);
    Constraints new_constraints = get_affine_constraints_for_InterfaceData(own_dof_indices, inner_dof_indices, Geometry.levels[level].n_total_level_dofs);
    std::vector<InterfaceDofData> outer_dofs = get_dof_association_by_boundary_id(outer_boundary_id);
    for(unsigned int i = 0; i < outer_dofs.size(); i++) {
      new_constraints.add_line(outer_dofs[i].index);
      new_constraints.set_inhomogeneity(outer_dofs[i].index, ComplexNumber(0,0));
    }
    in_constraints->merge(new_constraints, Constraints::MergeConflictBehavior::right_object_wins, true);
}

void PMLSurface::make_edge_constraints(Constraints * constraints, BoundaryId other_boundary) {
    std::vector<InterfaceDofData> other_boundary_id = Geometry.levels[level].surfaces[other_boundary]->get_dof_association_by_boundary_id(b_id);
    std::vector<InterfaceDofData> own_dof_indices = get_dof_association_by_boundary_id(other_boundary);
    Constraints new_constraints = get_affine_constraints_for_InterfaceData(other_boundary_id, own_dof_indices, Geometry.levels[level].n_total_level_dofs);
    constraints->merge(new_constraints, Constraints::MergeConflictBehavior::right_object_wins, true);
}

std::vector<SurfaceCellData> PMLSurface::get_surface_cell_data(BoundaryId in_bid) {
  std::vector<SurfaceCellData> ret;
  std::vector<unsigned int> dof_indices(fe_nedelec.dofs_per_cell);
  for(auto it = dof_h_nedelec.begin(); it != dof_h_nedelec.end(); it++) {
    if(it->at_boundary(in_bid)) {
      SurfaceCellData new_cell;
      it->get_dof_indices(dof_indices);
      for(unsigned int i = 0; i < 6; i++) {
        if(is_position_at_boundary(it->face(i)->center(), in_bid)) {
          new_cell.surface_face_center = it->face(i)->center();
        }
      }
      for(unsigned int i = 0; i < fe_nedelec.dofs_per_cell; i++) {
        new_cell.dof_numbers.push_back(dof_indices[i] + first_own_dof);
      }
      ret.push_back(new_cell);
    }
  }
  std::sort(ret.begin(), ret.end(), compareSurfaceCellData);
  for(unsigned int i = 0; i < ret.size(); i++) {
    for(unsigned int j = 1; j < ret[i].dof_numbers.size(); j++) {
      if(ret[i].dof_numbers[j] == ret[i].dof_numbers[j-1]) {
        std::cout << "Error in PMLSurface::get_surface_cell_data" << std::endl;
      }
    }
  }
  return ret;
}

std::vector<SurfaceCellData> PMLSurface::get_inner_surface_cell_data() {
  return get_surface_cell_data(inner_boundary_id);
}

void PMLSurface::fill_internal_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constraints) {
  std::vector<unsigned int> cell_dofs(fe_nedelec.dofs_per_cell);
  for(auto it: dof_h_nedelec) {
    it.get_dof_indices(cell_dofs);
    for(unsigned int i = 0; i < fe_nedelec.dofs_per_cell; i++) {
      cell_dofs[i] += first_own_dof;
    }
    in_constraints->add_entries_local_to_global(cell_dofs, *in_dsp);
  }
}

std::vector<SurfaceCellData> PMLSurface::get_corner_surface_cell_data(BoundaryId main_boundary, BoundaryId secondary_boundary) {
  std::vector<SurfaceCellData> ret;
  std::vector<unsigned int> dof_indices(fe_nedelec.dofs_per_cell);
  for(auto it : dof_h_nedelec) {
    if(it.at_boundary(main_boundary) && it.at_boundary(secondary_boundary)) {
      SurfaceCellData scd;
      for(unsigned int i = 0; i < 6; i++) {
        for(unsigned int j = 0; j < 4; j++) {
          Position p = it.face(i)->line(j)->center();
          if(is_position_at_boundary(p, main_boundary) && is_position_at_boundary(p, secondary_boundary)) {
            scd.surface_face_center = p;
          }
        }
      }
      it.get_dof_indices(dof_indices);
      for(unsigned int i = 0; i < fe_nedelec.dofs_per_cell; i++) {
        scd.dof_numbers.push_back(dof_indices[i] + first_own_dof);
      }
      ret.push_back(scd);
    }
  }
  std::sort(ret.begin(), ret.end(), compareSurfaceCellData);
  return ret;
}