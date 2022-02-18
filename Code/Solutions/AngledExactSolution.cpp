#include "./AngledExactSolution.h"
#include "../Core/Types.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>

AngledExactSolution::AngledExactSolution(): Function<3, ComplexNumber>(3) {
    base_solution = new ExactSolution(true, false);
    u1.resize(3);
    u2.resize(3);
    u3.resize(3);
    u1[0] = 1;
    u1[1] = 0;
    u1[2] = 0;
    u2[0] = 0;
    u2[1] = 2;
    u2[2] = -1;
    u3[0] = 0;
    u3[1] = -1;
    u3[2] = 1;
    
}

Position AngledExactSolution::transform_position(const Position &in_p) const {
  Position ret = in_p;
  ret[2] = in_p[2] - in_p[1];
  return ret;
}

ComplexNumber AngledExactSolution::value(const Position &p, const unsigned int component) const {
  Position transformed = transform_position(p);
  dealii::Vector<ComplexNumber> field(3);
  base_solution->vector_value(transformed, field);
  ComplexNumber ret(0,0);
  if(component == 0) {
    for(unsigned int i = 0; i < 3; i++) {
      ret += u1[i] * field[i];
    }
  }
  if(component == 1) {
    for(unsigned int i = 0; i < 3; i++) {
      ret += u2[i] * field[i];
    }
    ret *= std::sqrt(2);
  }
  if(component == 2) {
    for(unsigned int i = 0; i < 3; i++) {
      ret += u3[i] * field[i];
    }
  }
  return ret;
}

void AngledExactSolution::vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const {
  Position transformed = transform_position(p);
  dealii::Vector<ComplexNumber> base_val, ret;
  base_val.reinit(3);
  ret.reinit(3);
  base_solution->vector_value(transformed, base_val);
  for(unsigned int i = 0; i < 3; i++) {
    ret[0] += u1[i] * base_val[i];
    ret[1] += u2[i] * base_val[i] * std::sqrt(2);
    ret[2] += u3[i] * base_val[i];
  }
  value = ret;
}

dealii::Tensor<1, 3, ComplexNumber> AngledExactSolution::curl(const Position & p) const {
  Position transformed = transform_position(p);
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
  vector_value(transformed, val);
  Position deltap = transformed;
  deltap[0] = deltap[0] + h;
  vector_value(deltap, dxF);
  deltap = transformed;
  deltap[1] = deltap[1] + h;
  vector_value(deltap, dyF);
  deltap = transformed;
  deltap[2] = deltap[2] + h;
  vector_value(deltap, dzF);
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

dealii::Tensor<1, 3, ComplexNumber> AngledExactSolution::val(const Position &in_p) const {
  Position transformed = transform_position(in_p);
  dealii::Tensor<1, 3, ComplexNumber> ret;
  NumericVectorLocal vals;
  base_solution->vector_value(transformed, vals);
  for(unsigned int i = 0; i < 3; i++) {
    ret[i] = vals[i];
  }
  ret[1] *= std::sqrt(2);
  return ret;
}
