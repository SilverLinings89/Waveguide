#include "./BoundaryCondition.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include "../Core/InnerDomain.h"
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

std::vector<unsigned int> BoundaryCondition::dof_indices_from_surface_cell_data(std::vector<SurfaceCellData> in_data) {
  std::vector<unsigned int> ret;
  std::sort(in_data.begin(), in_data.end(), &compareSurfaceCellData);
  for(auto cell : in_data) {
    for(auto index : cell.dof_numbers) {
      ret.push_back(index);
    }
  }
  return ret;
}

void BoundaryCondition::fill_sparsity_pattern_with_surface_data_vectors(std::vector<SurfaceCellData> first_data_vector, std::vector<SurfaceCellData> second_data_vector, dealii::DynamicSparsityPattern *in_dsp, Constraints *constraints) {
  for(unsigned int i = 0; i < first_data_vector.size(); i++) {
    std::vector<unsigned int> indices;
    for(unsigned int j = 0; j < first_data_vector[i].dof_numbers.size(); j++) {
      indices.push_back(first_data_vector[i].dof_numbers[j]);
    }
    for(unsigned int j = 0; j < second_data_vector[i].dof_numbers.size(); j++) {
      indices.push_back(second_data_vector[i].dof_numbers[j]);
    }
    std::sort(indices.begin(), indices.end());
    constraints->add_entries_local_to_global(indices, *in_dsp);
  }
}

void BoundaryCondition::fill_sparsity_pattern_for_inner_surface(dealii::DynamicSparsityPattern *in_dsp, Constraints *constraints) {
  std::vector<SurfaceCellData> from_surface = get_inner_surface_cell_data();
  std::vector<SurfaceCellData> from_inner_domain = Geometry.inner_domain->get_surface_cell_data_for_boundary_id_and_level(b_id, level);
  if(from_surface.size() != from_inner_domain.size()) {
    std::cout << "Sizes incompatible in fill_sparsity_pattern_for_inner_surface for surface " << b_id << std::endl;
    std::cout << from_surface.size() << " versus " << from_inner_domain.size() << std::endl;
    for(unsigned int i = 0; i < 6; i++) {
      std::cout << get_surface_cell_data(i).size() << " on bid " << i << std::endl;
    }
  }
  fill_sparsity_pattern_with_surface_data_vectors(from_surface, from_inner_domain, in_dsp, constraints);
}

void BoundaryCondition::fill_sparsity_pattern(dealii::DynamicSparsityPattern * in_dsp, Constraints * in_constraints) {
  fill_internal_sparsity_pattern(in_dsp, in_constraints);
  fill_sparsity_pattern_for_inner_surface(in_dsp, in_constraints);
  for(unsigned int i = 0; i < 6; i++) {
    if(!are_opposing_sites(i, b_id) && (i != b_id)) {
      make_edge_sparsity_pattern(i, in_constraints, in_dsp);
    }    
  }
}

void BoundaryCondition::make_edge_sparsity_pattern(const BoundaryId in_bid, Constraints * in_constraints, dealii::DynamicSparsityPattern * in_dsp) {
  if(Geometry.levels[level].surface_type[b_id] != SurfaceType::OPEN_SURFACE && Geometry.levels[level].surface_type[in_bid] != SurfaceType::OPEN_SURFACE) {
    std::vector<SurfaceCellData> own_vector = get_surface_cell_data(in_bid);
    std::vector<SurfaceCellData> other_vector = Geometry.levels[level].surfaces[in_bid]->get_surface_cell_data(b_id);
    fill_sparsity_pattern_with_surface_data_vectors(own_vector, other_vector, in_dsp, in_constraints);
  }
}
