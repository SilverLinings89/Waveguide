#include "./BoundaryCondition.h"

using namespace dealii;

BoundaryCondition::BoundaryCondition(unsigned int in_bid, double in_additional_coordinate, const dealii::Triangulation<2> & in_surf_tria):
  b_id(in_bid), 
  additional_coordinate(in_additional_coordinate) {
    surface_triangulation.copy_triangulation(in_surf_tria);
}


void BoundaryCondition::identify_corner_cells() {
  auto it = surface_triangulation.begin_active();
  auto end = surface_triangulation.end();
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
    auto it = this->surface_triangulation.begin_active();
    std::vector<double> x;
    std::vector<double> y;
    while(it != this->surface_triangulation.end()){
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
    it = this->surface_triangulation.begin_active();
    while(it != this->surface_triangulation.end()){
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
    return (this->surface_triangulation.get_boundary_ids());
}