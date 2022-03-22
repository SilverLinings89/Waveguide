#pragma once

/**
 * @file DofData.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief Contains an internal data type.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../Core/Enums.h"
#include <string>

/**
 * \struct DofData
 * 
 * \brief This struct is used to store data about degrees of freedom for Hardy space infinite elements. This datatype is somewhat internal and should not require additional work.
 * 
 */

struct DofData {
  DofType type; // The type of degree of freedom.
  int hsie_order; // The Hardy space polynomial degree.
  int inner_order; // The degree of the Nedelec-Part of the element.
  bool nodal_basis; // Indicates thst the dof uses the nodal basis (i.e. is a Qlinear element on the surface)
  unsigned int global_index; // The number of this degree in global numbering
  bool got_base_dof_index; // Indicates of the dof of the dof base has been set.
  unsigned int base_dof_index; // Index of the base dof that is being used (The HSIE dofs are all either Q or Nedelec elements on the surface. If this index is set, it references the dof of the referred type)
  std::string base_structure_id_face; // For face dofs, the id is a string
  unsigned int base_structure_id_non_face;  // For other dofs the id is an unsigned int
  bool orientation = true; // defines if the dof is orientet positively.

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
