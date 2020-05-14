/*
 * structs.cpp
 *
 *  Created on: May 11, 2020
 *      Author: kraft
 */

#ifndef CODE_HELPERS_STRUCTS
#define CODE_HELPERS_STRUCTS

#include <deal.II/base/index_set.h>

struct DofAssociation {
  bool is_edge;
  unsigned int edge_index;
  std::string face_index;
  unsigned int dof_index_on_hsie_surface;
  dealii::Point<3> base_point;
  bool true_orientation;
};


struct DofCount {
  unsigned int hsie = 0;
  unsigned int non_hsie = 0;
  unsigned int total = 0;
};

struct LevelDofOwnershipData {
  unsigned int global_dofs;
  unsigned int owned_dofs;
  dealii::IndexSet locally_owned_dofs;
  dealii::IndexSet input_dofs;
  dealii::IndexSet output_dofs;
  dealii::IndexSet locally_relevant_dofs;

  LevelDofOwnershipData() {
    global_dofs = 0;
    owned_dofs = 0;
    locally_owned_dofs.clear();
    input_dofs.clear();
    output_dofs.clear();
    locally_relevant_dofs.clear();
  }

  LevelDofOwnershipData(unsigned int in_global) {
    global_dofs = in_global;
    owned_dofs = 0;
    locally_owned_dofs.clear();
    locally_owned_dofs.set_size(in_global);
    input_dofs.clear();
    input_dofs.set_size(in_global);
    output_dofs.clear();
    output_dofs.set_size(in_global);
    locally_relevant_dofs.clear();
    locally_relevant_dofs.set_size(in_global);
  }
};

struct ConstraintPair {
  unsigned int left;
  unsigned int right;
  bool sign;
};


#endif /* CODE_HELPERS_STRUCTS */
