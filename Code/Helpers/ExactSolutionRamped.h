#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>
#include "PointVal.h"
#include "../Core/GlobalObjects.h"
#include "./ExactSolution.h"

/**
 * \class ExactSolution
 * \brief This class is derived from the Function class and can be used to
 * estimate the L2-error for a straight waveguide. In the case of a completely
 * cylindrical waveguide, an analytic solution is known (the modes of the
 * input-signal themselves) and this class offers a representation of this
 * analytical solution. If the waveguide has any other shape, this solution does
 * not lose its value completely - it can still be used as a starting-vector for
 * iterative solvers.
 *
 * The structure of this class is defined by the properties of the
 * Function-class meaning that we have two functions:
 *  -#  virtual double value (const Point<dim> &p, const unsigned int component
 * ) calculates the value for a single component of the vector-valued
 * return-value.
 *  -#  virtual void vector_value (const Point<dim> &p,	Vector<double> &value)
 * puts these individual components into the parameter value, which is a
 * reference to a vector, handed over to store the result.
 *
 * \author Pascal Kraft
 * \date 23.11.2015
 */

class ExactSolutionRamped: public dealii::Function<3, ComplexNumber> {
  private:
    ExactSolution inner_field;
    const double min_z;
    const double max_z;

  public:
    ExactSolutionRamped(double z_for_max, double z_for_min, bool in_rectangular = false, bool in_dual = false);
    double get_ramping_factor_for_position(const Position &) const;
    ComplexNumber value(const Position &p, const unsigned int component) const;
    void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const;
    dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;
    dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;
};
