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
#include "./NeighborSurface.h"
#include "PMLMeshTransformation.h"

double get_surface_coordinate_for_bid(BoundaryId my_bid) {
  double surface_coordinate = 0;
  switch (my_bid)
  {
    case 0:
      surface_coordinate = Geometry.local_x_range.first;
      break;
    case 1:
      surface_coordinate = Geometry.local_x_range.second;
      break;
    case 2:
      surface_coordinate = Geometry.local_y_range.first;
      break;
    case 3:
      surface_coordinate = Geometry.local_y_range.second;
      break;
    case 4:
      surface_coordinate = Geometry.local_z_range.first;
      break;
    case 5:
      surface_coordinate = Geometry.local_z_range.second;
      break;
  }
  return surface_coordinate;
}

PMLSurface::PMLSurface(unsigned int surface, unsigned int in_level)
  : BoundaryCondition(surface, in_level, Geometry.surface_extremal_coordinate[surface]),
  fe_nedelec(GlobalParams.Nedelec_element_order),
  surface_coordinate(get_surface_coordinate_for_bid(surface)) {
     outer_boundary_id = surface;
     non_pml_layer_thickness = GlobalParams.PML_thickness / GlobalParams.PML_N_Layers;
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
  repetitions[b_id / 2] = GlobalParams.PML_N_Layers;
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
  CubeSurfaceTruncationState truncation_state;

  compute_coordinate_ranges(&tria);
  for(unsigned int i = 0; i < adjacent_boundaries.size(); i++) {
    dealii::Triangulation<3> t;
    bool built = mg_process_edge(&t , adjacent_boundaries[i]);
    if(built) {
      dealii::GridGenerator::merge_triangulations(tria,t,tria);
    }
  }
  for(unsigned int i = 0; i < adjacent_boundaries.size(); i++) {
    for(unsigned int j = i+1; j < adjacent_boundaries.size(); j++) {
      if(!are_opposing_sites(adjacent_boundaries[i], adjacent_boundaries[j])) {
        dealii::Triangulation<3> t;
        bool built = mg_process_corner(&t, adjacent_boundaries[i], adjacent_boundaries[j]);
        if(built) {
          dealii::GridGenerator::merge_triangulations(tria,t,tria);
        }
      }
    }
  }
  triangulation = reforge_triangulation(&tria);
  set_boundary_ids();
}

unsigned int PMLSurface::cells_for_boundary_id(unsigned int in_boundary_id) {
    unsigned int ret = 0;
    for(auto it = triangulation.begin(); it!= triangulation.end(); it++) {
      if(it->at_boundary()) {
        for(unsigned int i = 0; i < 6; i++) {
          if(it->face(i)->boundary_id() == in_boundary_id) {
            ret++;
          }
        }
      }
    }
    return ret;
}

void PMLSurface::init_fe() {
    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(fe_nedelec);
    dof_counter = dof_handler.n_dofs();
}

bool PMLSurface::is_position_at_boundary(const Position in_p, const BoundaryId in_bid) {
  switch (in_bid)
  {
    case 0:
      if(std::abs(in_p[0] - x_range.first) < FLOATING_PRECISION) return true;
      break;
    case 1:
      if(std::abs(in_p[0] - x_range.second) < FLOATING_PRECISION) return true;
      break;
    case 2:
      if(std::abs(in_p[1] - y_range.first) < FLOATING_PRECISION) return true;
      break;
    case 3:
      if(std::abs(in_p[1] - y_range.second) < FLOATING_PRECISION) return true;
      break;
    case 4:
      if(std::abs(in_p[2] - z_range.first) < FLOATING_PRECISION) return true;
      break;
    case 5:
      if(std::abs(in_p[2] - z_range.second) < FLOATING_PRECISION) return true;
      break;
  }
  return false;
}

bool PMLSurface::is_position_at_extended_boundary(const Position in_p, const BoundaryId in_bid) {
  if(std::abs(in_p[b_id / 2] - surface_coordinate) < FLOATING_PRECISION) {
    
    switch(b_id / 2) {
      case 0:
        return false;
        break;
      case 1:
        if((in_bid / 2) == 0) {
          if(in_p[0] < Geometry.local_x_range.first && in_bid == 0) {
            
            return true;
          }
          if(in_p[0] > Geometry.local_x_range.second && in_bid == 1) {
            return true;
          }
        } else {
          return false;
        }
        break;
      case 2:
        if(in_bid == 3) {
          return in_p[1] > Geometry.local_y_range.second;
        }
        if(in_bid == 2) {
          return in_p[1] < Geometry.local_y_range.first;
        }
        if(in_bid == 1) {
          bool not_y = in_p[1] <= Geometry.local_y_range.second && in_p[1] >= Geometry.local_y_range.first;
          if(not_y) {
            return in_p[0] > Geometry.local_x_range.second;
          } else {
            return false;
          }
        }
        if(in_bid == 0) {
          bool not_y = in_p[1] <= Geometry.local_y_range.second && in_p[1] >= Geometry.local_y_range.first;
          if(not_y) {
            return in_p[0] < Geometry.local_x_range.first;
          } else {
            return false;
          }
        }
        break;
    }
  } else {
    return b_id == in_bid;
  }
}

bool PMLSurface::is_point_at_boundary(Position2D, BoundaryId) {
  return false;
}

void PMLSurface::validate_meshes() {
  bool all_correct = true;
  for(auto other_boundary: adjacent_boundaries) {
    if(Geometry.levels[level].surface_type[other_boundary] == SurfaceType::ABC_SURFACE) {
      unsigned int own_cells = cells_for_boundary_id(other_boundary);
      unsigned int other_side = Geometry.levels[level].surfaces[other_boundary]->cells_for_boundary_id(b_id);
      std::vector<InterfaceDofData> own_dof_data = get_dof_association_by_boundary_id(other_boundary);
      std::vector<InterfaceDofData> other_dof_data = Geometry.levels[level].surfaces[other_boundary]->get_dof_association_by_boundary_id(b_id);
      if(own_cells != other_side || own_dof_data.size() != other_dof_data.size()) {
        all_correct = false;
        std::cout << "On " << std::to_string(b_id) << " for " << std::to_string(other_boundary) << " (" << own_cells << " vs " << other_side << ") and ("  << own_dof_data.size() << " vs " << other_dof_data.size() << ")." << std::endl;
      }
    }
  }
  if(!all_correct) {
    std::cout << "On " << std::to_string(b_id) << " some boundaries are not OK." << std::endl;
  }
}

void PMLSurface::initialize() {
  prepare_mesh();
  init_fe();
  prepare_dof_associations();
}

void PMLSurface::prepare_dof_associations() {
  std::array<IndexSet, 6> added_indices;
  std::array<std::vector<InterfaceDofData>, 6> temp_storage;
  unsigned int n_dofs = dof_handler.n_dofs();
  for(unsigned int i = 0; i < 6; i++) {
    added_indices[i].set_size(n_dofs);
    dof_associations[i].clear();
  }
  
  IndexSet face_set(n_dofs);
  IndexSet line_set(n_dofs);

  for (auto cell : dof_handler.active_cell_iterators()) {
    if (cell->at_boundary()) {
      for (unsigned int face = 0; face < 6; face++) {
        for (unsigned int boundary_id = 0; boundary_id < 6; boundary_id++) {
          if (cell->face(face)->boundary_id() == boundary_id) {
            std::vector<DofNumber> face_dofs_indices(fe_nedelec.dofs_per_face);
            cell->face(face)->get_dof_indices(face_dofs_indices);
            face_set.clear();
            line_set.clear();
            face_set.add_indices(face_dofs_indices.begin(), face_dofs_indices.end());
            std::vector<InterfaceDofData> cell_dofs_and_orientations_and_points;
            for (unsigned int i = 0; i < dealii::GeometryInfo<3>::lines_per_face; i++) {
              std::vector<DofNumber> line_dofs(fe_nedelec.dofs_per_line);
              cell->face(face)->line(i)->get_dof_indices(line_dofs);
              line_set.add_indices(line_dofs.begin(), line_dofs.end());
              for (unsigned int j = 0; j < fe_nedelec.dofs_per_line; j++) {
                InterfaceDofData new_item;
                new_item.index = line_dofs[j];
                new_item.base_point = cell->face(face)->line(i)->center();
                new_item.order = j;
                cell_dofs_and_orientations_and_points.push_back(new_item);
              }
            }
            for(unsigned int i = 0; i < fe_nedelec.dofs_per_face; i++) {
              if(!line_set.is_element(face_set.nth_index_in_set(i))) {
                InterfaceDofData new_item;
                new_item.index = face_set.nth_index_in_set(i);
                new_item.base_point = cell->face(face)->center();
                new_item.order = i;
                cell_dofs_and_orientations_and_points.push_back(new_item);
              }
            }
            for (auto item: cell_dofs_and_orientations_and_points) {
              temp_storage[boundary_id].push_back(item);
            }
          }
        }
      }
    }
  }

  for(unsigned int i = 0; i < 6; i++) {
    for(unsigned int j = 0; j < temp_storage[i].size(); j++) {
      if(!added_indices[i].is_element(temp_storage[i][j].index)) {
        dof_associations[i].push_back(temp_storage[i][j]);
        added_indices[i].add_index(temp_storage[i][j].index);
      }
    }
    std::sort(dof_associations[i].begin(), dof_associations[i].end(), compareDofBaseDataAndOrientation);
  }

  std::cout << "Dofs by boundary_id on surface " << b_id << ": " << dof_associations[0].size() << ", " << dof_associations[1].size() << ", " << dof_associations[2].size() << ", " << dof_associations[3].size() << ", " << dof_associations[4].size() << ", " << dof_associations[5].size() << "." <<std::endl;
}

std::vector<InterfaceDofData> PMLSurface::get_dof_association_by_boundary_id(unsigned int in_bid) {
  return dof_associations[in_bid];
  /**
  std::vector<InterfaceDofData> ret;
  std::vector<types::global_dof_index> local_line_dofs(fe_nedelec.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe_nedelec.dofs_per_face);
  std::set<DofNumber> face_set;
  triangulation.clear_user_flags();
  for (auto cell : dof_handler.active_cell_iterators()) {
    if (cell->at_boundary()) {
      for (unsigned int face = 0; face < 6; face++) {
        if (cell->face(face)->boundary_id() == in_bid) {
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
                new_item.index = line_dofs[j];
                new_item.base_point = cell->face(face)->line(i)->center();
                new_item.order = j;
                cell_dofs_and_orientations_and_points.push_back(new_item);
              }
              cell->face(face)->line(i)->set_user_flag();
            }
          }
          for (auto item: face_set) {
            InterfaceDofData new_item;
            new_item.index = item;
            new_item.base_point = cell->face(face)->center();
            new_item.order = 0;
            cell_dofs_and_orientations_and_points.push_back(new_item);
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
  **/
}

std::vector<InterfaceDofData> PMLSurface::get_dof_association() {
    return get_dof_association_by_boundary_id(inner_boundary_id);
}

std::array<double, 3> PMLSurface::fraction_of_pml_direction(Position in_p) {
  std::array<double, 3> ret;
  if(in_p[0] < Geometry.local_x_range.first) {
    ret[0] = (Geometry.local_x_range.first - in_p[0] + non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
  } else {
    if(in_p[0] > Geometry.local_x_range.second) {
      ret[0] = (in_p[0] - Geometry.local_x_range.second - non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
    } else {
      ret[0] = 0.0;
    }
  }
  if(in_p[1] < Geometry.local_y_range.first) {
    ret[1] = (Geometry.local_y_range.first - in_p[1] + non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
  } else {
    if(in_p[1] > Geometry.local_y_range.second) {
      ret[1] = (in_p[1] - Geometry.local_y_range.second - non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
    } else {
      ret[1] = 0.0;
    }
  }
  if(in_p[2] < Geometry.local_z_range.first) {
    ret[2] = (Geometry.local_z_range.first - in_p[2] + non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
  } else {
    if(in_p[2] > Geometry.local_z_range.second) {
      ret[2] = (in_p[2] - Geometry.local_z_range.second - non_pml_layer_thickness) / (GlobalParams.PML_thickness - non_pml_layer_thickness);
    } else {
      ret[2] = 0.0;
    }
  }
  return ret;
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
  const std::array<double, 3> fractions = fraction_of_pml_direction(in_p);
  ComplexNumber sx = {1 , std::pow(fractions[0], GlobalParams.PML_skaling_order) * GlobalParams.PML_Sigma_Max};
  ComplexNumber sy = {1 , std::pow(fractions[1], GlobalParams.PML_skaling_order) * GlobalParams.PML_Sigma_Max};
  ComplexNumber sz = {1 , std::pow(fractions[2], GlobalParams.PML_skaling_order) * GlobalParams.PML_Sigma_Max};
  for(unsigned int i = 0; i < 3; i++) {
      for(unsigned int j = 0; j < 3; j++) {
          ret[i][j] = 0;
      }
  }
  ret[0][0] = sy*sz/sx;
  ret[1][1] = sx*sz/sy;
  ret[2][2] = sx*sy/sz;
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

void PMLSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, Constraints * in_constraints) {
  std::vector<unsigned int> local_indices(fe_nedelec.dofs_per_cell);
  for(auto it = dof_handler.begin_active(); it != dof_handler.end(); it++) {
    it->get_dof_indices(local_indices);
    local_indices = transform_local_to_global_dofs(local_indices);
    in_constraints->add_entries_local_to_global(local_indices, *in_dsp);
  }
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed* rhs, Constraints *constraints){
  CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
      cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
      cell_data.local_dof_indices = transform_local_to_global_dofs(cell_data.local_dof_indices);
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
      constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, true);
  }
  matrix->compress(dealii::VectorOperation::add);
}

void PMLSurface::fill_matrix(dealii::SparseMatrix<ComplexNumber> * matrix, Constraints *constraints){
  CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
      cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
      cell_data.local_dof_indices = transform_local_to_global_dofs(cell_data.local_dof_indices);
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
  CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_handler);
  for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
    cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
    cell_data.local_dof_indices = transform_local_to_global_dofs(cell_data.local_dof_indices);
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
    constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, true);
  }
  matrix->compress(dealii::VectorOperation::add);
}

void PMLSurface::compute_coordinate_ranges(dealii::Triangulation<3> * in_tria) {
  x_range.first = 100000.0;
  y_range.first = 100000.0;
  z_range.first = 100000.0;
  x_range.second = -100000.0;
  y_range.second = -100000.0;
  z_range.second = -100000.0;
  for(auto it = in_tria->begin(); it != in_tria->end(); it++) {
    for(unsigned int i = 0; i < 6; i++) {
      if(it->face(i)->at_boundary()) {
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
}

bool PMLSurface::extend_mesh_in_direction(BoundaryId in_bid) {
  if(Geometry.levels[level].surface_type[in_bid] != SurfaceType::ABC_SURFACE) {
    return false;
  }
  if(b_id == 4 || b_id == 5) {
    return true;
  }
  if(b_id == 0 || b_id == 1) {
    return false;
  }
  if(b_id == 2 || b_id == 3) {
    return in_bid < b_id;
  }
  return false;
}

void PMLSurface::set_boundary_ids() {
  // first set all to outer_boundary_id
  for(auto it = triangulation.begin(); it != triangulation.end(); it++) {
    for(unsigned int face = 0; face < 6; face ++) {
      if(it->face(face)->at_boundary()) {
        it->face(face)->set_all_boundary_ids(outer_boundary_id);
      }
    }
  }
  // then locate all the faces connecting to the inner domain
  for(auto it = triangulation.begin(); it != triangulation.end(); it++) {
    for(unsigned int face = 0; face < 6; face ++) {
      if(it->face(face)->at_boundary()) {
        Position p = it->face(face)->center();
        // Have to use outer_boundary_id here because direction 4 of the pml (-z) is at the boundary 5 of the inner domain (+z)
        if(p[b_id/2] == get_surface_coordinate_for_bid(outer_boundary_id)) {
          bool is_located_properly = true;
          is_located_properly &= p[0] > Geometry.local_x_range.first - FLOATING_PRECISION;
          is_located_properly &= p[0] < Geometry.local_x_range.second + FLOATING_PRECISION;
          is_located_properly &= p[1] > Geometry.local_y_range.first - FLOATING_PRECISION;
          is_located_properly &= p[1] < Geometry.local_y_range.second + FLOATING_PRECISION;
          is_located_properly &= p[2] > Geometry.local_z_range.first - FLOATING_PRECISION;
          is_located_properly &= p[2] < Geometry.local_z_range.second + FLOATING_PRECISION;
          if(is_located_properly) {
            it->face(face)->set_all_boundary_ids(inner_boundary_id);
          }
        }
        
      }
    }
  }

  // then check all of the other boundary ids.
  std::array<unsigned int, 6> boundary_counts;
  for(unsigned int i = 0; i< 6; i++) {
    boundary_counts[i] = 0;
  }
  for(auto it = triangulation.begin(); it != triangulation.end(); it++) {
    for(unsigned int face = 0; face < 6; face ++) {
      if(it->face(face)->at_boundary()) {
        Position p = it->face(face)->center();
        for(unsigned int i = 0; i < 6; i++) {
          if(i != b_id && !are_opposing_sites(i,b_id)) {
            bool is_at_boundary = false;
            if(extend_mesh_in_direction(i)) {
              is_at_boundary = is_position_at_extended_boundary(p,i);
            } else {
              is_at_boundary = is_position_at_boundary(p,i);
            }
            if(is_at_boundary) {
              it->face(face)->set_all_boundary_ids(i);
              boundary_counts[i]++;
            }
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

std::string PMLSurface::output_results(const dealii::Vector<ComplexNumber> & in_data, std::string in_filename) {
  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(in_data, "Solution");
  dealii::Vector<ComplexNumber> zero = dealii::Vector<ComplexNumber>(in_data.size());
  for(unsigned int i = 0; i < in_data.size(); i++) {
    zero[i] = 0;
  }
  data_out.add_data_vector(zero, "Exact_Solution");
  const std::string filename = GlobalOutputManager.get_numbered_filename(in_filename + "-" + std::to_string(b_id) + "-", GlobalParams.MPI_Rank, "vtu");
  std::ofstream outputvtu(filename);
  data_out.build_patches();
  data_out.write_vtu(outputvtu);
  return filename;
}

DofCount PMLSurface::compute_n_locally_owned_dofs() {
  IndexSet non_owned_dofs = compute_non_owned_dofs();
  return dof_counter - non_owned_dofs.n_elements();
}

DofCount PMLSurface::compute_n_locally_active_dofs() {
  return dof_counter;
}

void PMLSurface::finish_dof_index_initialization() {
  for(unsigned int surf = 0; surf < 6; surf++) {
    if(surf != b_id && !are_opposing_sites(surf, b_id)) {
      if(!are_edge_dofs_owned[surf] && Geometry.levels[level].surface_type[surf] != SurfaceType::NEIGHBOR_SURFACE) {
        DofIndexVector dofs_in_global_numbering = Geometry.levels[level].surfaces[surf]->get_global_dof_indices_by_boundary_id(b_id);
        std::vector<InterfaceDofData> local_interface_data = get_dof_association_by_boundary_id(surf);
        DofIndexVector dofs_in_local_numbering(local_interface_data.size());
        for(unsigned int i = 0; i < local_interface_data.size(); i++) {
          dofs_in_local_numbering[i] = local_interface_data[i].index;
        }
        set_non_local_dof_indices(dofs_in_local_numbering, dofs_in_global_numbering);
      }
    }
  }
  // Do the same for the inner interface
  std::vector<InterfaceDofData> global_interface_data = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(b_id);
  std::vector<InterfaceDofData> local_interface_data = get_dof_association_by_boundary_id(inner_boundary_id);
  DofIndexVector dofs_in_local_numbering(local_interface_data.size());
  DofIndexVector dofs_in_global_numbering(local_interface_data.size());
  
  for(unsigned int i = 0; i < local_interface_data.size(); i++) {
    dofs_in_local_numbering[i] = local_interface_data[i].index;
    dofs_in_global_numbering[i] = Geometry.levels[level].inner_domain->global_index_mapping[global_interface_data[i].index];
  }
  set_non_local_dof_indices(dofs_in_local_numbering, dofs_in_global_numbering);
}

void PMLSurface::determine_non_owned_dofs() {
  // validate_meshes();
  IndexSet non_owned_dofs = compute_non_owned_dofs();
  const unsigned int n_dofs = non_owned_dofs.n_elements();
  std::vector<unsigned int> local_dofs(n_dofs);
  for(unsigned int i = 0; i < n_dofs; i++) {
    local_dofs[i] = non_owned_dofs.nth_index_in_set(i);
  }
  mark_local_dofs_as_non_local(local_dofs);
}

bool PMLSurface::finish_initialization(DofNumber index) {
  std::vector<InterfaceDofData> dofs = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(b_id);
  std::vector<InterfaceDofData> own = get_dof_association();
  std::vector<unsigned int> local_indices, global_indices;
  if(own.size() != dofs.size()) {
    std::cout << "Size mismatch in finish initialization: " << own.size() << " != " << dofs.size() << std::endl;
  }
  for(unsigned int i = 0; i < dofs.size(); i++) {
    local_indices.push_back(own[i].index);
    global_indices.push_back(dofs[i].index);
  }
  set_non_local_dof_indices(local_indices, global_indices);
  return FEDomain::finish_initialization(index);
}

Constraints PMLSurface::make_constraints() {
  IndexSet global_indices = IndexSet(Geometry.levels[level].n_total_level_dofs);
  global_indices.add_range(0, Geometry.levels[level].n_total_level_dofs);
  Constraints ret(global_indices);
  std::vector<InterfaceDofData> dofs = get_dof_association_by_boundary_id(outer_boundary_id);
  for(auto dof : dofs) {
		const unsigned int local_index = dof.index;
		const unsigned int global_index = global_index_mapping[local_index];
		ret.add_line(global_index);
		ret.set_inhomogeneity(global_index, ComplexNumber(0,0));
	}
  return ret;
}

dealii::IndexSet PMLSurface::compute_non_owned_dofs() {
  IndexSet non_owned_dofs(dof_counter);
  std::vector<unsigned int> non_locally_owned_surfaces;
  for(auto surf : adjacent_boundaries) {
    if(!are_edge_dofs_owned[surf]) {
      non_locally_owned_surfaces.push_back(surf);
    }
  }
  non_locally_owned_surfaces.push_back(inner_boundary_id);

  std::vector<unsigned int> local_indices(fe_nedelec.dofs_per_face);
  // The non owned surfaces are the one towards the inner domain and the surfaces 0,1 and 2 if they are false in the input.
  for(auto it = dof_handler.begin_active(); it != dof_handler.end(); it++) {
    for(unsigned int face = 0; face < 6; face++) {
      if(it->face(face)->at_boundary()) {
        for(auto surf: non_locally_owned_surfaces) {
          if(it->face(face)->boundary_id() == surf) {
            it->face(face)->get_dof_indices(local_indices);
            for(unsigned int i = 0; i < fe_nedelec.dofs_per_face; i++) {
              non_owned_dofs.add_index(local_indices[i]);
            }
          }
        }
      }
    }
  }
  return non_owned_dofs;
}

bool PMLSurface::mg_process_edge(dealii::Triangulation<3> * tria, BoundaryId other_bid) {
  // This line checks if the domain even exists
  bool domain_exists = Geometry.levels[level].is_surface_truncated[other_bid];
  // the next step checks if this boundary generates it. For b_id 4 and 5, this is always the case. For 2 and 3 it is only true if the other b_id 
  bool is_owned = false;
  if(b_id == 4 || b_id == 5) {
    is_owned = true;
  }
  if(b_id == 2 || b_id == 3) {
    is_owned = (other_bid == 0 || other_bid == 1);
  }
  if(domain_exists && is_owned) {
    std::vector<unsigned int> subdivisions(3);
    Position p1, p2;
    if(b_id / 2 != 0 && other_bid /2 != 0) {
      subdivisions[0] = GlobalParams.Cells_in_x;
      p1[0] = Geometry.local_x_range.first;
      p2[0] = Geometry.local_x_range.second;
    } else {
      subdivisions[0] = GlobalParams.PML_N_Layers;
      if(b_id == 0 || other_bid == 0) {
        p1[0] = Geometry.local_x_range.first - GlobalParams.PML_thickness;
        p2[0] = Geometry.local_x_range.first;
      } else {
        p1[0] = Geometry.local_x_range.second;
        p2[0] = Geometry.local_x_range.second + GlobalParams.PML_thickness;
      }
    }
    if(b_id / 2 != 1 && other_bid /2 != 1) {
      subdivisions[1] = GlobalParams.Cells_in_y;
      p1[1] = Geometry.local_y_range.first;
      p2[1] = Geometry.local_y_range.second;
    } else {
      subdivisions[1] = GlobalParams.PML_N_Layers;
      if(b_id == 2 || other_bid == 2) {
        p1[1] = Geometry.local_y_range.first - GlobalParams.PML_thickness;
        p2[1] = Geometry.local_y_range.first;
      } else {
        p1[1] = Geometry.local_y_range.second;
        p2[1] = Geometry.local_y_range.second + GlobalParams.PML_thickness;
      }
    }
    if(b_id / 2 != 2 && other_bid /2 != 2) {
      subdivisions[2] = GlobalParams.Cells_in_z;
      p1[2] = Geometry.local_z_range.first;
      p2[2] = Geometry.local_z_range.second;
    } else {
      subdivisions[2] = GlobalParams.PML_N_Layers;
      if(b_id == 4 || other_bid == 4) {
        p1[2] = Geometry.local_z_range.first - GlobalParams.PML_thickness;
        p2[2] = Geometry.local_z_range.first;
      } else {
        p1[2] = Geometry.local_z_range.second;
        p2[2] = Geometry.local_z_range.second + GlobalParams.PML_thickness;
      }
    }
    dealii::GridGenerator::subdivided_hyper_rectangle(*tria,subdivisions, p1, p2);
    return true;
  }
  return false;
}

bool PMLSurface::mg_process_corner(dealii::Triangulation<3> * tria, BoundaryId first_bid, BoundaryId second_bid) {
  if(b_id == 4 || b_id == 5) {
    bool generate_this_part = Geometry.levels[level].is_surface_truncated[first_bid] && Geometry.levels[level].is_surface_truncated[second_bid];
    if(generate_this_part) {
      // Do the generation.
      GridGenerator::subdivided_hyper_cube(*tria,GlobalParams.PML_N_Layers,0,GlobalParams.PML_thickness);
      dealii::Tensor<1,3> shift;
      bool lower_x = first_bid == 0 || second_bid == 0;
      bool lower_y = first_bid == 2 || second_bid == 2;
      if(lower_x) {
        shift[0] = - GlobalParams.PML_thickness + Geometry.local_x_range.first;
      } else {
        shift[0] = Geometry.local_x_range.second;
      }
      if(lower_y) {
        shift[1] = - GlobalParams.PML_thickness + Geometry.local_y_range.first;
      } else {
        shift[1] = Geometry.local_y_range.second;
      }
      if(b_id == 4) {
        shift[2] = - GlobalParams.PML_thickness + Geometry.local_z_range.first;
      } else {
        shift[2] = Geometry.local_z_range.second;
      }
      dealii::GridTools::shift(shift, *tria);
      return true;
    }
  }
  return false;
}
