#include <iostream>
#include <string>
#include <utility>
#include "PredefinedShapeTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/ShapeDescription.h"
#include "../Helpers/staticfunctions.h"
#include "SpaceTransformation.h"

using namespace dealii;

PredefinedShapeTransformation::PredefinedShapeTransformation()
    : SpaceTransformation() {
  for(unsigned int i = 0; i < 3; i++) {
    for(unsigned int j = 0; j < 3; j++) {
      if(i == j) {
        I[i][j] = 1;
      } else {
        I[i][j] = 0;
      }
    }
  }
}

PredefinedShapeTransformation::~PredefinedShapeTransformation() {}

Position PredefinedShapeTransformation::math_to_phys(Position coord) const {
  Position ret = coord;
  ret[1] += get_m(coord[2]);
  return ret;
}

Position PredefinedShapeTransformation::phys_to_math(Position coord) const {
  Position ret = coord;
  ret[1] -= get_m(coord[2]);
  return ret;
}

Tensor<2, 3, ComplexNumber>
PredefinedShapeTransformation::get_Tensor(Position &position) {
  return get_Space_Transformation_Tensor(position);
}


void PredefinedShapeTransformation::estimate_and_initialize() {
  print_info("PredefinedShapeTransformation::estimate_and_initialize", "Start");
  const ShapeDescription & shape = GlobalParams.sd;
  if(GlobalParams.MPI_Rank == 0) {
    for (unsigned int i = 0; i < shape.z.size(); i++) {
      std::string msg = "Node at z: " + std::to_string(shape.z[i]) + " (m: " + std::to_string(shape.m[i]) + " v: " + std::to_string(i < shape.v.size() ? shape.v[i] : 0.0) + ")";
      print_info("PredefinedShapeTransformation::estimate_and_initialize", msg);
    }
  }
  print_info("PredefinedShapeTransformation::estimate_and_initialize", "End");
}

double PredefinedShapeTransformation::get_m(double z_in) const {
  return GlobalParams.sd.displacement(z_in - Geometry.global_z_range.first);
}

double PredefinedShapeTransformation::get_v(double z_in) const {
  return GlobalParams.sd.slope(z_in - Geometry.global_z_range.first);
}

void PredefinedShapeTransformation::Print() const {
  std::cout << "Printing is not yet implemented." << std::endl;
}

Tensor<2, 3, double>
PredefinedShapeTransformation::get_Space_Transformation_Tensor(Position &position) {
  Tensor<2, 3, double> J_loc = get_J(position);
  Tensor<2, 3, double> ret;
  ret[0][0] = 1;
  ret[1][1] = 1;
  ret[2][2] = 1;
  return (J_loc * ret * transpose(J_loc)) / determinant(J_loc);
}

Tensor<2,3,double> PredefinedShapeTransformation::get_J(Position &in_p) {
  Tensor<2,3,double> ret = I;
  ret[1][2] = - get_v(in_p[2]);
  return ret;
}

Tensor<2,3,double> PredefinedShapeTransformation::get_J_inverse(Position &in_p) {
  Tensor<2,3,double> ret = get_J(in_p);
  return invert(ret);
}