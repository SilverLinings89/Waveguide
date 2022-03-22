#include "AngleWaveguideTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "../Core/Sector.h"
#include "../Helpers/staticfunctions.h"
#include "SpaceTransformation.h"

using namespace dealii;

AngleWaveguideTransformation::AngleWaveguideTransformation()
    : SpaceTransformation() {
  
}

dealii::Tensor<2, 3, double> AngleWaveguideTransformation::get_J(Position &) {
  if(!is_constant || !is_J_prepared) {
    dealii::Tensor<2, 3, double> ret;
    ret[0][0] = 1;
    ret[1][1] = 1;
    ret[2][2] = 1;
    ret[2][1] = -0.2;
    J_perm = ret;
    is_J_prepared = true;
  }
  return J_perm;
}

dealii::Tensor<2, 3, double> AngleWaveguideTransformation::get_J_inverse(Position &c) {
  if(!is_constant || !is_J_inv_prepared) {
    dealii::Tensor<2, 3, double> ret = get_J(c);
    ret = invert(ret);
    J_inv_perm = ret;
    is_J_inv_prepared = true;
  }
  return J_inv_perm;
}

double AngleWaveguideTransformation::get_det(Position c) {
  if(!is_constant || !is_det_prepared) {
    det = determinant(get_J(c));
    is_det_prepared = true;
  }
  return det;
}

AngleWaveguideTransformation::~AngleWaveguideTransformation() {}

Position AngleWaveguideTransformation::math_to_phys(Position coord) const {
  Position ret;
  ret[0] = coord[0];
  ret[1] = coord[1];
  ret[2] = coord[2] + GlobalParams.PML_Angle_Test*coord[1];
  return ret;
}

Position AngleWaveguideTransformation::phys_to_math(Position coord) const {
  Position ret;
  ret[0] = coord[0];
  ret[1] = coord[1];
  ret[2] = coord[2] - GlobalParams.PML_Angle_Test*coord[1];
  return ret;
}

Tensor<2, 3, ComplexNumber>
AngleWaveguideTransformation::get_Tensor(Position &position) {
  return get_Space_Transformation_Tensor(position);
}

void AngleWaveguideTransformation::estimate_and_initialize() {
  
}

Vector<double> AngleWaveguideTransformation::get_dof_values() const {
  Vector<double> ret;
  return ret;
}

unsigned int AngleWaveguideTransformation::n_free_dofs() const {
  return 0;
}

void AngleWaveguideTransformation::Print() const {
  std::cout << "Printing is not yet implemented." << std::endl;
}

unsigned int AngleWaveguideTransformation::n_dofs() const {
  return 0;
}

Tensor<2, 3, double>
AngleWaveguideTransformation::get_Space_Transformation_Tensor(Position &p) {
  Tensor<2, 3, double> ret;
  ret[0][0] = 1;
  ret[1][1] = 1;
  ret[2][2] = 1;
  return (get_J(p) * ret * transpose(get_J(p))) / get_det(p);
}
