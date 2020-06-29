/*
 * PointSourceField.cpp
 *
 *  Created on: Jun 9, 2020
 *      Author: kraft
 */

#include "PointSourceField.h"

const double PI = 3.141592653589;

PointSourceField::PointSourceField() {
  // TODO Auto-generated constructor stub

}

PointSourceField::~PointSourceField() {
  // TODO Auto-generated destructor stub
}

std::complex<double> PointSourceField::value(
    const dealii::Point<3> &p,
    const unsigned int component) const {
  if (component == 0) {
    if (p.norm() < 0.01) {
      return std::exp(std::complex<double>(0, 1) * k * 0.01) / (4 * PI * 0.01);
    }
    return std::exp(std::complex<double>(0, 1) * k * p.norm())
        / (4 * PI * p.norm());
  } else {
    return std::complex<double>(0, 0);
  }
}

void PointSourceField::vector_value(const dealii::Point<3> &p,
    dealii::Vector<std::complex<double> > &vec) const {
  vec[0] = value(p, 0);
  vec[1] = 0;
  vec[2] = 0;
}

void PointSourceField::vector_curl(const dealii::Point<3> &p,
    dealii::Vector<std::complex<double> > &vec) {
  const std::complex<double> factor;
  if (p.norm() < 0.01) {
    factor =
        (std::complex<double>(1, 0) - std::complex<double>(0, 1) * k * 0.01)
            / 0.01;
  } else {
    factor = (std::complex<double>(1, 0)
        - std::complex<double>(0, 1) * k * p.norm()) / p.norm();
  }

  const std::complex<double> e_field = value(p, 0);
  vec[0] = 0;
  vec[1] = p[2] * e_field * factor;
  vec[2] = p[1] * e_field * factor;
}
