#pragma once 

/**
 * @file PointSourceField.h
 * @author your name (you@domain.com)
 * @brief Some implementations of fields that can be used in the code for forcing or error computation.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <deal.II/base/function.h>
#include "../Core/Types.h"

class PointSourceFieldHertz: public dealii::Function<3, ComplexNumber> {
public:
  double k = 1;
  const ComplexNumber ik;
  double cell_diameter = 0.01;
  PointSourceFieldHertz(double in_k = 1.0);
  void set_cell_diameter(double diameter);
  virtual ~PointSourceFieldHertz();
  ComplexNumber value(const Position &p, const unsigned int component = 0) const override;
  void vector_value(const Position &p, NumericVectorLocal &vec) const override;
  void vector_curl(const Position &p, NumericVectorLocal &vec);
};

class PointSourceFieldCosCos: public dealii::Function<3, ComplexNumber> {
public:
  PointSourceFieldCosCos();
  virtual ~PointSourceFieldCosCos();
  ComplexNumber value(const Position &p, const unsigned int component = 0) const override;
  void vector_value(const Position &p, NumericVectorLocal &vec) const override;
  void vector_curl(const Position &p, NumericVectorLocal &vec);
};


/**
 * @brief The TE10 mode of a rectangular waveguide with perfectly conducting walls.
 * For a waveguide of width a in the x-direction (x in [-a/2, a/2]) filled with a homogeneous medium with wavenumber
 * k, the field E = (0, cos(pi x / a), 0) exp(i beta z) with beta = sqrt(k^2 - (pi/a)^2) is an exact solution of the
 * time-harmonic Maxwell equations whose tangential components vanish on all four side walls. Together with PEC
 * side walls ("Lateral boundary = PEC") it is a test with an exact solution in which only the output side has to
 * be truncated by an absorbing boundary condition.
 */
class MetallicWaveguideMode: public dealii::Function<3, ComplexNumber> {
public:
  const double width;
  const double beta;
  MetallicWaveguideMode(double in_k, double in_width);
  ComplexNumber value(const Position &p, const unsigned int component = 0) const override;
  void vector_value(const Position &p, NumericVectorLocal &vec) const override;
};
