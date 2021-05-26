#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>
#include "../Helpers/PointVal.h"
#include "../Core/GlobalObjects.h"
#include "./ExactSolution.h"

class ExactSolutionRamped: public dealii::Function<3, ComplexNumber> {
  private:
    ExactSolution inner_field;
    const double min_z;
    const double max_z;
    const bool do_c0;

  public:
    ExactSolutionRamped( bool in_rectangular = false, bool in_dual = false);
    double get_ramping_factor_for_position(const Position &) const;
    ComplexNumber value(const Position &p, const unsigned int component) const;
    void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const;
    dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;
    dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;
    double compute_ramp_for_c0(const Position &in_p) const;
    double compute_ramp_for_c1(const Position &in_p) const;
    double ramping_delta(const Position &in_p) const;
    double get_ramping_factor_derivative_for_position(const Position &in_p) const;
};
