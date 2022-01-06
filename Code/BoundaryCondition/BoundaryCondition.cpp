#include "./BoundaryCondition.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include "../Core/InnerDomain.h"
#include <algorithm>

using namespace dealii;

BoundaryCondition::BoundaryCondition(unsigned int in_bid, unsigned int in_level, double in_additional_coordinate):
  b_id(in_bid),
  level(in_level),
  additional_coordinate(in_additional_coordinate),
  adjacent_boundaries(get_adjacent_boundary_ids(in_bid)) {
    for(unsigned int i = 0; i < 6; i++) {
      are_edge_dofs_owned[i] = false;
    }
    for(auto surf: adjacent_boundaries) {
      are_edge_dofs_owned[surf] = are_edge_dofs_locally_owned(b_id, surf, level);
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

std::vector<DofNumber> BoundaryCondition::get_global_dof_indices_by_boundary_id(BoundaryId in_boundary_id) {
  std::vector<InterfaceDofData> dof_data = get_dof_association_by_boundary_id(in_boundary_id);
  std::vector<DofNumber> ret;
  for(unsigned int i = 0; i < dof_data.size(); i++) {
    ret.push_back(dof_data[i].index);
  }
  
  ret = transform_local_to_global_dofs(ret);
  return ret;
}

void BoundaryCondition::finish_dof_index_initialization() {

}

Constraints BoundaryCondition::make_constraints() {
  Constraints ret(global_dof_indices);
  return ret;
}

double BoundaryCondition::boundary_norm(NumericVectorDistributed * in_v) {
  double ret = 0;
  for(unsigned int i = 0; i < global_index_mapping.size(); i++) {
    ret += norm_squared(in_v->operator()(global_index_mapping[i]));
  }
  return std::sqrt(ret);
}

double BoundaryCondition::boundary_surface_norm(NumericVectorDistributed * in_v, BoundaryId in_bid) {
  double ret = 0;
  auto dofs = get_dof_association_by_boundary_id(in_bid);
  for(auto it : dofs) {
    ret += norm_squared(in_v->operator()(it.index));
  }
  return std::sqrt(ret);
}

unsigned int BoundaryCondition::cells_for_boundary_id(unsigned int boundary_id) {
  return 0;
}

void BoundaryCondition::print_dof_validation() {
  unsigned int n_invalid_dofs = 0;
  for(unsigned int i = 0; i < n_locally_active_dofs; i++) {
    if(global_index_mapping[i] >= Geometry.levels[level].n_total_level_dofs) {
      n_invalid_dofs++;
    }
  }
  if(n_invalid_dofs > 0) {
    std::cout << "On process " << GlobalParams.MPI_Rank << " surface " << b_id << " has " << n_invalid_dofs << " invalid dofs." << std::endl;
    for(unsigned int surf = 0; surf < 6; surf++) {
      if(surf != b_id && !are_opposing_sites(b_id, surf)) {
        unsigned int invalid_dof_count = 0;
        unsigned int owned_invalid = 0;
        auto dofs = get_dof_association_by_boundary_id(surf);
        for(auto dof:dofs) {
          if(global_index_mapping[dof.index] >= Geometry.levels[level].n_total_level_dofs) {
            invalid_dof_count++;
            if(is_dof_owned[dof.index]) {
              owned_invalid++;
            }
          }
        }
        if(invalid_dof_count > 0) {
          std::cout << "On process " << GlobalParams.MPI_Rank << " surface " << b_id << " there were "<< invalid_dof_count << "(" << owned_invalid << ") invalid dofs towards "<< surf << std::endl;
        }
      }
    }
  }
}

void BoundaryCondition::force_validation() {
  if(Geometry.levels[level].surface_type[b_id] != SurfaceType::NEIGHBOR_SURFACE) {

    
    for(unsigned int surf = 0; surf < 6; surf++) {
        if(surf != b_id && !are_opposing_sites(b_id, surf)) {
   //       std::cout << "A" << std::endl;
          std::vector<InterfaceDofData> d = get_dof_association_by_boundary_id(surf);
     //     std::cout << "B" << std::endl;
          bool one_is_invalid = false;
          unsigned int count_before = 0;
          unsigned int count_after = 0;
          for(unsigned int index = 0; index < d.size(); index++) {
            if(!is_dof_owned[d[index].index]) {
              if(global_index_mapping[d[index].index] >= Geometry.levels[level].n_total_level_dofs) {
                one_is_invalid = true;
                count_before ++;
              }
            }
          }
       //   std::cout << "C" << std::endl;
          if(one_is_invalid) {
            std::cout << "Forcing validation on " << b_id << " for " << surf << std::endl;
            std::vector<unsigned int> local_indices(d.size());
            for(unsigned int i = 0; i < d.size(); i++) {
              local_indices[i] = d[i].index;
            }
            set_non_local_dof_indices(local_indices, Geometry.levels[level].surfaces[surf]->get_global_dof_indices_by_boundary_id(b_id));
            for(unsigned int index = 0; index < d.size(); index++) {
              if(!is_dof_owned[d[index].index]) {
                if(global_index_mapping[d[index].index] >= Geometry.levels[level].n_total_level_dofs) {
                  count_after ++;
                }
              }
            }
            std::cout << "Count before: " << count_before << " and after: " << count_after << std::endl;
          }
        }
    }
  }
}

unsigned int BoundaryCondition::n_cells() {
  return 0;
}