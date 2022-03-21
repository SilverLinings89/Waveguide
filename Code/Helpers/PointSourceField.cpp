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
PointSourceFieldCosCos::PointSourceFieldCosCos():
dealii::Function<3, ComplexNumber>(3) {

}

PointSourceFieldCosCos::~PointSourceFieldCosCos() {
  // Nothing to do here.
}

ComplexNumber PointSourceFieldCosCos::value(
    const Position &in_p,
    const unsigned int component) const {
  if(component == 0) return std::cos(factor * in_p[1]) * std::cos(factor * in_p[2]);
  if(component == 1) return std::cos(factor * in_p[0]) * std::cos(factor * in_p[2]);
  if(component == 2) return std::cos(factor * in_p[0]) * std::cos(factor * in_p[1]);
  return 0;
}

void PointSourceFieldCosCos::vector_value(const Position &in_p,
    NumericVectorLocal &vec) const {
  vec[0] = std::cos(factor * in_p[1]) * std::cos(factor * in_p[2]) ;
  vec[1] = std::cos(factor * in_p[0]) * std::cos(factor * in_p[2]) ;
  vec[2] = std::cos(factor * in_p[0]) * std::cos(factor * in_p[1]) ;
}

void PointSourceFieldCosCos::vector_curl(const Position &in_p,
    NumericVectorLocal &vec) {
  vec[0] = factor * (std::cos(factor * in_p[0]) * std::sin(factor * in_p[2]));
  vec[1] = factor * (std::cos(factor * in_p[1]) * std::sin(factor * in_p[0]));
  vec[2] = factor * (std::cos(factor * in_p[2]) * std::sin(factor * in_p[1]));
}

PointSourceFieldHertz::PointSourceFieldHertz(double in_k):
dealii::Function<3, ComplexNumber>(3),
ik(0, in_k) {
  k = in_k;
}

PointSourceFieldHertz::~PointSourceFieldHertz() {
  // Nothing to do here.
}

ComplexNumber PointSourceFieldHertz::value(
    const Position &in_p,
    const unsigned int component) const {
  NumericVectorLocal ret(3);
  vector_value(in_p, ret);
  return ret[component];
}

void PointSourceFieldHertz::vector_value(const Position &in_p, NumericVectorLocal &vec) const {
  NumericVectorLocal x(3);
  NumericVectorLocal x_normed(3);
  NumericVectorLocal p(3);
  Position p_temp = in_p;
  if(std::abs(in_p[0]) < cell_diameter/2.0 && std::abs(in_p[1]) < cell_diameter/2.0 && std::abs(in_p[2]) < cell_diameter/2.0) {
    vec[0] = 0;
    vec[1] = 0;
    vec[2] = 0;
  }
  
  const double norm_x = p_temp.norm();

  for(unsigned int i = 0; i < 3; i++) {
    x[i] = {p_temp[i], 0};
    x_normed[i] = x[i] / norm_x;
  }
  p[0] = {0, 0};
  p[1] = {0, 0};
  p[2] = {1, 0};
  ComplexNumber k_squared = {k * k, 0.0};
  ComplexNumber factor_2 = ComplexNumber(1.0/(norm_x*norm_x), 0) - ik/norm_x;
  ComplexNumber x_times_p = x[0] * p[0] + x[1] * p[1] + x[2] * p[2];
  x_times_p *= 3.0 / norm_x;

  NumericVectorLocal term_1 = crossproduct(crossproduct(x_normed, p), x_normed);
  multiply_in_place(k_squared, term_1);

  NumericVectorLocal term_2 = x_normed;
  multiply_in_place(x_times_p, term_2);
  term_2 -= p;
  multiply_in_place(factor_2, term_2);

  term_1 += term_2;
  multiply_in_place(ComplexNumber(0, 0.25 / PI) * std::exp(ik * norm_x) / norm_x, term_1);
  vec = term_1;
}

void PointSourceFieldHertz::vector_curl(const Position &p,
    NumericVectorLocal &vec) {
  ComplexNumber factor;
  if (p.norm() < 0.01) {
    factor = { 100, -k };
  } else {
    factor = (ComplexNumber(1, 0)
        - ik * p.norm()) / p.norm();
  }

  ComplexNumber e_field = value(p, 0);
  vec[0] = 0;
  vec[1] = p[2] * e_field * factor;
  vec[2] = p[1] * e_field * factor;
}

void PointSourceFieldHertz::set_cell_diameter(double in_diameter) {
  cell_diameter = in_diameter;
}