#pragma once

#include "../Core/Enums.h"
#include <string>

/**
 * \struct DofData
 * 
 * \brief This is only for the storage of basic data. 
 * 
 */

struct DofData {
  DofType type;
  int hsie_order;
  int inner_order;
  bool nodal_basis;
  unsigned int global_index;
  bool got_base_dof_index;
  unsigned int
      base_dof_index;
  std::string base_structure_id_face;
  unsigned int base_structure_id_non_face;
  bool orientation = true; 

  DofData() {
    base_structure_id_face = "";
    base_structure_id_non_face = 0;
    this->got_base_dof_index = false;
  }

  void set_base_dof(unsigned int in_base_dof_index) {
    this->got_base_dof_index = true;
    this->base_dof_index = in_base_dof_index;
  }

  DofData(std::string in_id) {
    this->got_base_dof_index = false;
    base_structure_id_face = in_id;
  }

  DofData(unsigned int in_id) {
    this->got_base_dof_index = false;
    base_structure_id_non_face = in_id;
  }

  auto update_nodal_basis_flag() -> void {
    this->nodal_basis = (this->type == DofType::RAY
        || this->type == DofType::IFFb || this->type == DofType::SEGMENTb);
  }
};
