#pragma once 

/**
 * @file PointSourceField.h
 * @author Pascal Kraft
 * @brief Some implementations of fields that can be used in the code for forcing or error computation.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <deal.II/base/function.h>
#include "../Core/Types.h"

/**
 * @brief The electric field of a Hertz dipole in a homogeneous medium.
 * The dipole is located at the origin and points in the z-direction. With r = |x|, n = x / r, p = e_z and the
 * wavenumber k the field is
 * E = i / (4 pi) * exp(i k r) / r * ( k^2 (n x p) x n + (3 n (n . p) - p) (1 / r^2 - i k / r) ).
 * It is singular at the origin; use the parameter "global z shift" to keep the origin out of the domain. Selected by
 * source type 2.
 */
class PointSourceFieldHertz: public dealii::Function<3, ComplexNumber> {
public:
  /** @brief Wavenumber of the medium in 1/micrometer. */
  double k = 1;
  /** @brief The imaginary unit times k. */
  const ComplexNumber ik;
  /** @brief Edge length of the cube around the origin that vector_value() tests for (see the remark there). */
  double cell_diameter = 0.01;
  /**
   * @brief Creates the field for a wavenumber.
   * @param in_k wavenumber of the medium, 2 pi sqrt(epsilon_r) / lambda.
   */
  PointSourceFieldHertz(double in_k = 1.0);
  /**
   * @brief Sets cell_diameter.
   * @param diameter the new value, usually the diameter of a mesh cell.
   */
  void set_cell_diameter(double diameter);
  /** @brief Nothing to release. */
  virtual ~PointSourceFieldHertz();
  /**
   * @brief One component of the field.
   * @param p position in micrometers.
   * @param component 0, 1 or 2 for the x-, y- or z-component.
   * @return ComplexNumber the value of the component (computed with vector_value()).
   */
  ComplexNumber value(const Position &p, const unsigned int component = 0) const override;
  /**
   * @brief All three components of the field.
   * Positions inside the cube of edge length cell_diameter around the origin are detected but the zero written for
   * them is overwritten by the formula, which divides by |p|.
   * @param p position in micrometers.
   * @param vec vector of size 3 that receives the field.
   */
  void vector_value(const Position &p, NumericVectorLocal &vec) const override;
  /**
   * @brief Legacy function that is not called anywhere. It is not the curl of vector_value().
   * Writes (0, p_z E_x f, p_y E_x f) with f = (1 - i k r) / r for r >= 0.01 and f = 100 - i k otherwise.
   * @param p position in micrometers.
   * @param vec vector of size 3 that receives the result.
   */
  void vector_curl(const Position &p, NumericVectorLocal &vec);
};

/**
 * @brief A smooth real test field without singularity.
 * E = (cos(c y) cos(c z), cos(c x) cos(c z), cos(c x) cos(c y)) with c = 1 / sqrt(2). Selected by source type 1.
 */
class PointSourceFieldCosCos: public dealii::Function<3, ComplexNumber> {
public:
  /** @brief Creates the field (three components). */
  PointSourceFieldCosCos();
  /** @brief Nothing to release. */
  virtual ~PointSourceFieldCosCos();
  /**
   * @brief One component of the field.
   * @param p position in micrometers.
   * @param component 0, 1 or 2 for the x-, y- or z-component.
   * @return ComplexNumber the value of the component; 0 for other component indices.
   */
  ComplexNumber value(const Position &p, const unsigned int component = 0) const override;
  /**
   * @brief All three components of the field.
   * @param p position in micrometers.
   * @param vec vector of size 3 that receives the field.
   */
  void vector_value(const Position &p, NumericVectorLocal &vec) const override;
  /**
   * @brief Legacy function that is not called anywhere.
   * Writes c (cos(c x) sin(c z), cos(c y) sin(c x), cos(c z) sin(c y)) with c = 1 / sqrt(2). This is only one of the
   * two terms of every component of the curl of the field.
   * @param p position in micrometers.
   * @param vec vector of size 3 that receives the result.
   */
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
  /** @brief Width a of the waveguide in the x-direction in micrometers. */
  const double width;
  /** @brief Propagation constant sqrt(k^2 - (pi / a)^2) of the mode in 1/micrometer. */
  const double beta;
  /**
   * @brief Creates the mode.
   * @param in_k wavenumber of the medium inside the waveguide.
   * @param in_width width of the waveguide in the x-direction. The mode only propagates if in_k > pi / in_width.
   */
  MetallicWaveguideMode(double in_k, double in_width);
  /**
   * @brief One component of the field.
   * @param p position in micrometers.
   * @param component 0, 1 or 2. Only component 1 (y) is nonzero.
   * @return ComplexNumber cos(pi x / a) exp(i beta z) for the y-component and 0 otherwise.
   */
  ComplexNumber value(const Position &p, const unsigned int component = 0) const override;
  /**
   * @brief All three components of the field.
   * @param p position in micrometers.
   * @param vec vector of size 3 that receives the field.
   */
  void vector_value(const Position &p, NumericVectorLocal &vec) const override;
};
