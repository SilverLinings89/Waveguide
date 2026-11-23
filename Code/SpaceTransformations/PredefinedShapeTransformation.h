#pragma once 

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <math.h>
#include <vector>

#include "../Core/InnerDomain.h"
#include "../Core/Sector.h"
#include "SpaceTransformation.h"

/**
 * \class PredefinedShapeTransformation
 * \brief This class is used to describe the hump examples.
 *
 * The waveguide axis is displaced vertically by m(z), where m interpolates the nodes of a ShapeDescription
 * (one line of Modes/test.csv, a photonic wire bond of the project partners). The physical coordinates are
 * (x, y + m(z), z).
 */

class PredefinedShapeTransformation : public SpaceTransformation {
  dealii::Tensor<2,3,double> I;
 public:
  PredefinedShapeTransformation();

  virtual ~PredefinedShapeTransformation();

  Position math_to_phys(Position coord) const;

  Position phys_to_math(Position coord) const;

  dealii::Tensor<2, 3, ComplexNumber> get_Tensor(Position &coordinate) ;

  dealii::Tensor<2, 3, double> get_Space_Transformation_Tensor(Position &coordinate);

  Tensor<2,3,double> get_J(Position &) override;

  Tensor<2,3,double> get_J_inverse(Position &) override;

  /**
   * The shape is given completely by the nodes of Parameters::sd, so there is nothing to estimate. The function
   * only writes the nodes to the log.
   */
  void estimate_and_initialize();

  /**
   * Returns the vertical displacement of the waveguide axis for a system-coordinate: the piecewise cubic
   * interpolation of the nodes of Parameters::sd (ShapeDescription::displacement()). The first node is located at
   * the lower end of the global z-range. The nodes do not have to be equidistant.
   */
  double get_m(double in_z) const;

  /**
   * Returns the slope of the waveguide axis for a system-coordinate: the derivative of get_m() with respect to z
   * (ShapeDescription::slope()).
   */
  double get_v(double in_z) const;

  /**
   * Console output of the current Waveguide Structure.
   */
  void Print() const;

};