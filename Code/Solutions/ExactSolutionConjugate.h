#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>
#include "../Helpers/PointVal.h"
#include "../Core/GlobalObjects.h"
#include "./ExactSolution.h"

class ExactSolutionConjugate: public dealii::Function<3, ComplexNumber> {
  private:
    ExactSolution inner_field;

  public:
    ExactSolutionConjugate( bool in_rectangular = false, bool in_dual = false);
    ComplexNumber value(const Position &p, const unsigned int component) const;
    void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const;
    dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;
    dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;
};
