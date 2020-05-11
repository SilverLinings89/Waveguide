/*
 * structs.cpp
 *
 *  Created on: May 11, 2020
 *      Author: kraft
 */

#ifndef CODE_HELPERS_STRUCTS
#define CODE_HELPERS_STRUCTS

#include <deal.II/base/index_set.h>

struct DofCount {
  unsigned int owned = 0;
  unsigned int non_owned = 0;
  unsigned int hsie = 0;
  unsigned int non_hsie = 0;
  unsigned int owned_hsie = 0;
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
  unsigned int left, right;
  bool sign;
};


#endif /* CODE_HELPERS_STRUCTS */
