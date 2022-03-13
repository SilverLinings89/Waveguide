#include "ExactSolutionConjugate.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include <complex>
#include "../Core/Types.h"
#include "../Helpers/PointVal.h"

void conjugate_vector(dealii::Tensor<1,3,ComplexNumber> * in_v) {
  for(unsigned int i = 0; i < 3; i++) {
    (*in_v)[i].imag( -(*in_v)[i].imag());
  }
}

void conjugate_vector(dealii::Vector<ComplexNumber> * in_v) {
  for(unsigned int i = 0; i < 3; i++) {
    (*in_v)[i].imag( -(*in_v)[i].imag());
  }
}

ComplexNumber ExactSolutionConjugate::value(const Position &in_p, const unsigned int component) const {
  ComplexNumber ret = inner_field.value(in_p, component);
  ret.imag(-ret.imag());
  return ret;
}

void ExactSolutionConjugate::vector_value(const Position &in_p, dealii::Vector<ComplexNumber> &values) const {
  inner_field.vector_value(in_p, values);
  conjugate_vector(&values);
}

dealii::Tensor<1, 3, ComplexNumber> ExactSolutionConjugate::curl(const Position &in_p) const {
  dealii::Tensor<1,3,ComplexNumber> ret = inner_field.curl(in_p);
  conjugate_vector(&ret);
  return ret;
}

dealii::Tensor<1, 3, ComplexNumber> ExactSolutionConjugate::val(const Position &in_p) const {
  dealii::Tensor<1,3, ComplexNumber> ret = inner_field.val(in_p);
  conjugate_vector(&ret);
  return ret;
}

ExactSolutionConjugate::ExactSolutionConjugate():
  dealii::Function<3, ComplexNumber>(3),
  inner_field()
{ }

