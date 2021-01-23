/*
 * JacobianForCell.cpp
 *
 *  Created on: Jun 16, 2020
 *      Author: kraft
 */

#include "../Core/Types.h"
#include "JacobianForCell.h"
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include "../Helpers/staticfunctions.h"
#include <vector>

JacobianForCell::JacobianForCell(FaceAngelingData &in_fad, const BoundaryId &in_bid, double in_additional_component) {
  reinit(in_fad, in_bid, in_additional_component); 
}

void JacobianForCell::reinit(FaceAngelingData &in_fad, const BoundaryId &in_bid, double in_additional_component) {
  x                    = {"x"};
  y                    = {"y"};
  z                    = {"z"};
  z0                   = {"z0"};
  boundary_id          = in_bid;
  additional_component = in_additional_component;
  bool all_straight = true;
  for(unsigned int i = 0; i < 4; i++) {
    if(!in_fad[i].is_x_angled || !in_fad[i].is_y_angled) {
      all_straight = false;
    }
  }
  if(all_straight) {
    F[0]                   = x;
    F[1]                   = y;
    F[2]                   = (z-z0);
  } else {
    // TODO: This needs to be implemented for PSE-103 
    F[0]                   = x;
    F[1]                   = y;
    F[2]                   = (z-z0);
  }
  surface_wide_substitution_map[z0]  = MathExpression(in_additional_component);
  for(unsigned int i = 0; i <3; i++) {
    F[i] = F[i].substitute(surface_wide_substitution_map);
  }
  for(unsigned int i = 0; i < 3; i++) {
    J[i][0] = F[i].differentiate(x);
    J[i][1] = F[i].differentiate(y);
    J[i][2] = F[i].differentiate(z);
  }
}

dealii::Tensor<2,3,double> JacobianForCell::get_J_hat_for_position(const Position2D &position) const {
  dealii::Tensor<2,3,double> ret;
  dealii::Differentiation::SD::types::substitution_map substitution_map;
  substitution_map[x] = MathExpression(position[0]);
  substitution_map[y] = MathExpression(position[1]);
  substitution_map[z] = MathExpression(additional_component);
  for(unsigned int i = 0; i < 3; i++){
    for(unsigned int j = 0; j < 3; j++){
      ret[i][j] = J[i][j].substitute_and_evaluate<double>(substitution_map);
    }  
  }
  return ret;
}

JacobianAndTensorData JacobianForCell::get_C_G_and_J(Position2D in_p) {
  JacobianAndTensorData ret;
  ret.J = get_J_hat_for_position(in_p);
  const double J_norm = ret.J.norm();
  const Tensor<2,3,double> J_inverse = invert(ret.J);
  ret.G = J_norm * J_inverse * transpose(J_inverse);
  ret.C = (1.0 / J_norm) * transpose(ret.J) * ret.J;
  return ret;
}

Position JacobianForCell::transform_to_3D_space(Position2D in_position){
  Position ret= {in_position[0], in_position[1], additional_component};
  if (boundary_id == 0) {
    return Transform_5_to_0(ret);
  }
  if (boundary_id == 1) {
    return Transform_5_to_1(ret);
  }
  if (boundary_id == 2) {
    return Transform_5_to_2(ret);
  }
  if (boundary_id == 3) {
    return Transform_5_to_3(ret);
  }
  if (boundary_id == 4) {
    return Transform_5_to_4(ret);
  }
  if (boundary_id == 5) {
    return ret;
  }
  return ret;
}

std::pair<Position2D,double> JacobianForCell::split_into_triangulation_and_external_part(const Position in_point) {
  Position temp = in_point;
  if (boundary_id == 0) {
    temp = Transform_0_to_5(in_point);
  }
  if (boundary_id == 1) {
    temp = Transform_1_to_5(in_point);
  }
  if (boundary_id == 2) {
    temp = Transform_2_to_5(in_point);
  }
  if (boundary_id == 3) {
    temp = Transform_3_to_5(in_point);
  }
  if (boundary_id == 4) {
    temp = Transform_4_to_5(in_point);
  }
  return {{temp[0], temp[1]}, temp[2]};
}