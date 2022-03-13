#include "ExactSolutionRamped.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "../Core/Types.h"
#include "../Helpers/PointVal.h"

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
  const double h = 0.0001;
  dealii::Tensor<1, 3, ComplexNumber> ret;
  dealii::Vector<ComplexNumber> dxF;
  dealii::Vector<ComplexNumber> dyF;
  dealii::Vector<ComplexNumber> dzF;
  dealii::Vector<ComplexNumber> val;
  dxF.reinit(6, false);
  dyF.reinit(6, false);
  dzF.reinit(6, false);
  val.reinit(6, false);
  this->vector_value(in_p, val);
  Position deltap = in_p;
  deltap[0] = deltap[0] + h;
  this->vector_value(deltap, dxF);
  deltap = in_p;
  deltap[1] = deltap[1] + h;
  this->vector_value(deltap, dyF);
  deltap = in_p;
  deltap[2] = deltap[2] + h;
  this->vector_value(deltap, dzF);
  for (int i = 0; i < 3; i++) {
    dxF[i] = (dxF[i] - val[i]) / h;
    dyF[i] = (dyF[i] - val[i]) / h;
    dzF[i] = (dzF[i] - val[i]) / h;
  }
  ret[0] = dyF[2] - dzF[1];
  ret[1] = dzF[0] - dxF[2];
  ret[2] = dxF[1] - dyF[0];

  return ret;
}

dealii::Tensor<1, 3, ComplexNumber> ExactSolutionRamped::val(const Position &in_p) const {
  const double factor = get_ramping_factor_for_position(in_p);
  dealii::Tensor<1,3, ComplexNumber> ret = inner_field.val(in_p);
  ret *= factor;
  return ret;
}

ExactSolutionRamped::ExactSolutionRamped():
  dealii::Function<3, ComplexNumber>(3),
  inner_field(),
  min_z(GlobalParams.tapering_min_z),
  max_z(GlobalParams.tapering_max_z),
  do_c0(GlobalParams.Signal_tapering_type == SignalTaperingType::C0) { }

double ExactSolutionRamped::get_ramping_factor_for_position(const Position &in_p) const {
  if(do_c0) {
    return compute_ramp_for_c0(in_p);
  } else {
    return compute_ramp_for_c1(in_p);
  }
}

double ExactSolutionRamped::get_ramping_factor_derivative_for_position(const Position &in_p) const {
  if(do_c0) {
    return -1.0 / std::abs(max_z-min_z);
  } else {
    const double x = ramping_delta(in_p);
    return 6*x*x -6*x;
  }
}

double ExactSolutionRamped::ramping_delta(const Position &in_p) const {
  if(in_p[2] <= min_z) {
    return 0.0;
  }
  if(in_p[2] >= max_z) {
    return 1.0;
  }
  const double length = max_z - min_z;
  return (in_p[2] - min_z) / length;
}

double ExactSolutionRamped::compute_ramp_for_c0(const Position &in_p) const {
  if(in_p[2] <= min_z) {
    return 1.0;
  }
  if(in_p[2] >= max_z) {
    return 0.0;
  }
  const double delta = ramping_delta(in_p);
  return - delta + 1;
}

double ExactSolutionRamped::compute_ramp_for_c1(const Position &in_p) const {
  if(in_p[2] <= min_z) {
    return 1.0;
  }
  if(in_p[2] >= max_z) {
    return 0.0;
  }
  const double x = ramping_delta(in_p);
  return 2*std::pow(x,3) - 3*x*x + 1;
}