#pragma once

#include "ExactSolutionRamped.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "../Core/Types.h"
#include "PointVal.h"

ComplexNumber ExactSolutionRamped::value(const Position &in_p, const unsigned int component) const {
  const double factor = get_ramping_factor_for_position(in_p);
  ComplexNumber ret = inner_field.value(in_p, component);
  return ret * factor;
}

void ExactSolutionRamped::vector_value(const Position &in_p, dealii::Vector<ComplexNumber> &values) const {
  const double factor = get_ramping_factor_for_position(in_p);
  inner_field.vector_value(in_p, values);
  for(unsigned int i = 0; i < 3; i++) {
    values[i] *= factor;
  }
}

dealii::Tensor<1, 3, ComplexNumber> ExactSolutionRamped::curl(const Position &in_p) const {
  const double factor = get_ramping_factor_for_position(in_p);
  dealii::Tensor<1,3, ComplexNumber> ret = inner_field.curl(in_p);
  ret *= factor;
  return ret;
}

dealii::Tensor<1, 3, ComplexNumber> ExactSolutionRamped::val(const Position &in_p) const {
  const double factor = get_ramping_factor_for_position(in_p);
  dealii::Tensor<1,3, ComplexNumber> ret = inner_field.val(in_p);
  ret *= factor;
  return ret;
}

ExactSolutionRamped::ExactSolutionRamped(double z_for_max, double z_for_min, bool in_rectangular, bool in_dual): dealii::Function<3, ComplexNumber>(3), inner_field(in_rectangular, in_dual), min_z(z_for_min), max_z(z_for_max) { }

double ExactSolutionRamped::get_ramping_factor_for_position(const Position &in_p) const {
  if(in_p[2] <= min_z) {
    return 1.0;
  }
  if(in_p[2] >= max_z) {
    return 0.0;
  }
  const double length = max_z - min_z;
  const double current = in_p[2] - min_z;
  return length / current;
};