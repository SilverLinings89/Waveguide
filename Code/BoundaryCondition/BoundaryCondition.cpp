#include "./BoundaryCondition.h"
#include "../Core/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include <algorithm>

using namespace dealii;

BoundaryCondition::BoundaryCondition(unsigned int in_bid, unsigned int in_level, double in_additional_coordinate, DofNumber in_first_own_index):
  b_id(in_bid),
  level(in_level),
  additional_coordinate(in_additional_coordinate),
  first_own_dof(in_first_own_index) {

}


void BoundaryCondition::identify_corner_cells() {
  auto it = Geometry.surface_meshes[b_id].begin_active();
  auto end = Geometry.surface_meshes[b_id].end();
  for(; it != end; ++it) {
    unsigned int outside_edges = 0;
    for(unsigned int i = 0; i< dealii::GeometryInfo<2>::faces_per_cell; ++i) {
      if(it->face(i)->at_boundary()) outside_edges++;
    }
    if(outside_edges == 2) {
      this->corner_cell_ids.push_back(it->index());
    }
  }
}

void BoundaryCondition::set_mesh_boundary_ids() {
    auto it = Geometry.surface_meshes[b_id].begin_active();
    std::vector<double> x;
    std::vector<double> y;
    while(it != Geometry.surface_meshes[b_id].end()){
      if(it->at_boundary()) {
        for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell; ++face) {
          if (it->face(face)->at_boundary()) {
            dealii::Point<2, double> c;
            c = it->face(face)->center();
            x.push_back(c[0]);
            y.push_back(c[1]);
          }
        }
      }
      ++it;
    }
    double x_max = *max_element(x.begin(), x.end());
    double y_max = *max_element(y.begin(), y.end());
    double x_min = *min_element(x.begin(), x.end());
    double y_min = *min_element(y.begin(), y.end());
    it = Geometry.surface_meshes[b_id].begin_active();
    while(it != Geometry.surface_meshes[b_id].end()){
    if (it->at_boundary()) {
      for (unsigned int face = 0; face < dealii::GeometryInfo<2>::faces_per_cell;
          ++face) {
        Point<2, double> center;
        center = it->face(face)->center();
        if (std::abs(center[0] - x_min) < 0.0001) {
          it->face(face)->set_all_boundary_ids(
              edge_to_boundary_id[this->b_id][0]);
        }
        if (std::abs(center[0] - x_max) < 0.0001) {
          it->face(face)->set_all_boundary_ids(
              edge_to_boundary_id[this->b_id][1]);
        }
        if (std::abs(center[1] - y_min) < 0.0001) {
          it->face(face)->set_all_boundary_ids(
              edge_to_boundary_id[this->b_id][2]);
        }
        if (std::abs(center[1] - y_max) < 0.0001) {
          it->face(face)->set_all_boundary_ids(
              edge_to_boundary_id[this->b_id][3]);
        }
        }
    }
    ++it;
  }
}

std::vector<unsigned int> BoundaryCondition::get_boundary_ids() {
    return (Geometry.surface_meshes[b_id].get_boundary_ids());
}

std::vector<unsigned int> dof_indices_from_surface_cell_data(std::vector<SurfaceCellData> in_data) {
  std::vector<unsigned int> ret;
  std::sort(in_data.begin(), in_data.end(), &compareSurfaceCellData);
  for(auto cell : in_data) {
    for(auto index : cell.dof_numbers) {
      ret.push_back(index);
    }
  }
  return ret;
}

void fill_sparsity_pattern_with_surface_data_vectors(std::vector<SurfaceCellData> first_data_vector, std::vector<SurfaceCellData> second_data_vector, dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints) {
  for(unsigned int i = 0; i < first_data_vector.size(); i++) {
    std::move(first_data_vector[i].dof_numbers.begin(), first_data_vector[i].dof_numbers.end(), std::back_inserter(second_data_vector[i].dof_numbers));
    constraints->add_entries_local_to_global(second_data_vector[i].dof_numbers, in_dsp);
  }
}

void BoundaryCondition::fill_sparsity_pattern_for_inner_surface(dealii::DynamicSparsityPattern *in_dsp, dealii::AffineConstraints<ComplexNumber> *constraints) {
  std::vector<SurfaceCellData> from_surface = get_inner_surface_cell_data();
  std::vector<SurfaceCellData> from_inner_domain = Geometry.inner_domain->get_surface_cell_data_for_boundary_id_and_level(b_id, level);
  if(from_surface.size() != from_inner_domain.size()) {
    std::cout << "Sizes incompatible in fill_sparsity_pattern_for_inner_surface" << std::endl;
  }
  fill_sparsity_pattern_with_surface_data_vectors(from_surface, from_inner_domain, in_dsp, constraints);
}

void BoundaryCondition::fill_sparsity_pattern(dealii::DynamicSparsityPattern * in_dsp, dealii::AffineConstraints<ComplexNumber> * in_constraints) {
  fill_sparsity_pattern_for_inner_surface(in_dsp, in_constraints);
  if(b_id == 4 || b_id == 5) {
    fill_sparsity_pattern_for_boundary_id(2, in_constraints, in_dsp);
    fill_sparsity_pattern_for_boundary_id(0, in_constraints, in_dsp);
    fill_sparsity_pattern_for_boundary_id(1, in_constraints, in_dsp);
    fill_sparsity_pattern_for_boundary_id(3, in_constraints, in_dsp);
  }
  if(b_id == 2 || b_id == 3) {
    fill_sparsity_pattern_for_boundary_id(0, in_constraints, in_dsp);
    fill_sparsity_pattern_for_boundary_id(1, in_constraints, in_dsp);
  }
}