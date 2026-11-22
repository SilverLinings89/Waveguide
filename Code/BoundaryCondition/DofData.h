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
  DofType type; ///< Kind of shape function: EDGE and SURFACE are the Nedelec functions of the surface mesh (shared with the inner domain), RAY belongs to the infinite edge over a vertex, IFFa and IFFb to the infinite face over an edge, SEGMENTa and SEGMENTb to the interior of the infinite cell.
  int hsie_order; ///< Index j of the Hardy space function (see HSIEPolynomial::PsiJ and HSIEPolynomial::PhiJ). It is -1 for the function that carries the boundary value and 0 up to the HSIE polynomial degree for the others.
  int inner_order; ///< Index of the surface shape function among the functions of the same base structure (edge or cell). It starts at 1 for EDGE and IFFa dofs and at 0 otherwise. It is -1 for RAY dofs.
  bool nodal_basis; ///< True if the surface part of the shape function is a function of the nodal element (FE_Q) and not of the Nedelec element. This holds for the types RAY, IFFb and SEGMENTb, which point in the infinite direction.
  unsigned int global_index; ///< Index of this dof in the numbering of the HSIESurface it belongs to (0 to dof_counter - 1). FEDomain::transform_local_to_global_dofs() maps it to the numbering of the level.
  bool got_base_dof_index; ///< True once base_dof_index has been set by set_base_dof().
  unsigned int base_dof_index; ///< Index of the base dof that is being used (The HSIE dofs are all either Q or Nedelec elements on the surface. This is the index of that surface dof in the dof handler of the respective element, see nodal_basis)
  std::string base_structure_id_face; ///< For dofs that belong to a cell of the surface mesh (SURFACE, SEGMENTa, SEGMENTb): the id of that cell as a string (CellId::to_string()). Empty otherwise.
  unsigned int base_structure_id_non_face;  ///< For dofs that belong to an edge or a vertex of the surface mesh: the index of that edge or vertex in the surface triangulation.
  bool orientation = true; ///< For edge based dofs: result of get_orientation() for the two vertices of the edge, ordered by vertex index. It is stored but not evaluated anywhere.

  /**
   * @brief Creates an entry without base structure and without base dof.
   */
  DofData() {
    base_structure_id_face = "";
    base_structure_id_non_face = 0;
    this->got_base_dof_index = false;
  }

  /**
   * @brief Stores the index of the surface dof this dof is built on.
   *
   * @param in_base_dof_index Index of the dof in the dof handler of the surface element (nodal or Nedelec).
   */
  void set_base_dof(unsigned int in_base_dof_index) {
    this->got_base_dof_index = true;
    this->base_dof_index = in_base_dof_index;
  }

  /**
   * @brief Creates an entry for a dof that belongs to a cell of the surface mesh.
   *
   * @param in_id Id of the surface cell as a string.
   */
  DofData(std::string in_id) {
    this->got_base_dof_index = false;
    base_structure_id_face = in_id;
  }

  /**
   * @brief Creates an entry for a dof that belongs to an edge or a vertex of the surface mesh.
   *
   * @param in_id Index of the edge or vertex in the surface triangulation.
   */
  DofData(unsigned int in_id) {
    this->got_base_dof_index = false;
    base_structure_id_non_face = in_id;
  }

  /**
   * @brief Sets nodal_basis according to type. Has to be called after type was set.
   */
  auto update_nodal_basis_flag() -> void {
    this->nodal_basis = (this->type == DofType::RAY
        || this->type == DofType::IFFb || this->type == DofType::SEGMENTb);
  }
};
