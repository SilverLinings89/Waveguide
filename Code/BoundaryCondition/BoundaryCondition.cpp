#include "./BoundaryCondition.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include "../Core/InnerDomain.h"
#include <algorithm>

using namespace dealii;

BoundaryCondition::BoundaryCondition(unsigned int in_bid, unsigned int in_level, double in_additional_coordinate):
  b_id(in_bid),
  level(in_level),
  additional_coordinate(in_additional_coordinate) {
    is_isolated_boundary = false;
  is_surface_owned[0] = {{true , false , false, true,  false, true }};
  is_surface_owned[1] = {{false , true , false, true,  false, true }};
  is_surface_owned[2] = {{true,  true,  true , false , false, true }};
  is_surface_owned[3] = {{false, false, false , true , false, true }};
  is_surface_owned[4] = {{true,  true,  true,  true,  true , false }};
  is_surface_owned[5] = {{false, false, false, false, false , true }};
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

std::vector<DofNumber> BoundaryCondition::get_global_dof_indices_by_boundary_id(BoundaryId in_boundary_id) {
  std::vector<InterfaceDofData> dof_data = get_dof_association_by_boundary_id(in_boundary_id);
  std::vector<DofNumber> ret;
  for(unsigned int i = 0; i < dof_data.size(); i++) {
    ret.push_back(dof_data[i].index);
  }
  ret = transform_local_to_global_dofs(ret);
  return ret;
}

void BoundaryCondition::send_up_inner_dofs() {
  std::cout << "BoundaryCondition::send_up_inner_dofs() should not be called. It should only be called in derived classes!" << std::endl; 
}

void BoundaryCondition::receive_from_below_dofs() {
  std::cout << "BoundaryCondition::receive_from_below_dofs() should not be called. It should only be called in derived classes!" << std::endl;
}

void BoundaryCondition::finish_dof_index_initialization() {

}

std::vector<DofNumber> BoundaryCondition::receive_boundary_dofs(unsigned int) {
  std::vector<DofNumber> ret;
  std::cout << "BoundaryCondition::receive_boundary_dofs got called but never should." << std::endl;
  return ret;
}

Constraints BoundaryCondition::make_constraints() {
  Constraints ret(global_dof_indices);
  return ret;
}

void BoundaryCondition::mark_as_isolated() {
  this->is_isolated_boundary = true;
}