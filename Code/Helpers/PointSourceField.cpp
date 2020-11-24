/*
 * PointSourceField.cpp
 *
 *  Created on: Jun 9, 2020
 *      Author: kraft
 */

#include "PointSourceField.h"
#include "../Helpers/staticfunctions.h"

const double PI = 3.141592653589;
const double factor = 1.0 / std::sqrt(2.0);
PointSourceField::PointSourceField(double in_k):
dealii::Function<3, ComplexNumber>(3),
ik(0, in_k) {
  k = in_k;

}

PointSourceField::~PointSourceField() {
  // TODO Auto-generated destructor stub
}

ComplexNumber PointSourceField::value(
    const Position &in_p,
    const unsigned int component) const {
  if(component == 0) return std::sin(factor * in_p[1]) * std::sin(factor * in_p[2]);
  if(component == 1) return std::sin(factor * in_p[0]) * std::sin(factor * in_p[2]);
  if(component == 2) return std::sin(factor * in_p[0]) * std::sin(factor * in_p[1]);
  return 0;
}

void PointSourceField::vector_value(const Position &in_p,
    NumericVectorLocal &vec) const {
  vec[0] = std::sin(factor * in_p[1]) * std::sin(factor * in_p[2]) ;
  vec[1] = std::sin(factor * in_p[0]) * std::sin(factor * in_p[2]) ;
  vec[2] = std::sin(factor * in_p[0]) * std::sin(factor * in_p[1]) ;
}

void PointSourceField::vector_curl(const Position &in_p,
    NumericVectorLocal &vec) {
  vec[0] = factor * (std::sin(factor * in_p[0]) * std::cos(factor * in_p[2]));
  vec[1] = factor * (std::sin(factor * in_p[1]) * std::cos(factor * in_p[0]));
  vec[2] = factor * (std::sin(factor * in_p[2]) * std::cos(factor * in_p[1]));
}

void PointSourceField::set_cell_diameter(double in_diameter) {
  cell_diameter = in_diameter;
}