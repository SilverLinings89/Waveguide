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
 * \class InhomogenousTransformationRectangle
 * \brief In this case we regard a rectangular waveguide and the effects on the
 * material tensor by the space transformation and the boundary condition PML
 * may overlap (hence inhomogenous space transformation)
 *
 * If this kind of boundary condition works stably we will also be able to  deal
 * with more general settings (which might for example incorporate angles in
 * between the output and input connector. \author Pascal Kraft \date 28.11.2016
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
   * This member contains all the Sectors who, as a sum, form the complete
   * Waveguide. These Sectors are a partition of the simulated domain.
   */
  std::vector<Sector<2>> case_sectors;

  /**
   * At the beginning (before the first solution of a system) only the boundary
   * conditions for the shape of the waveguide are known. Therefore the values
   * for the degrees of freedom need to be estimated. This function sets all
   * variables to appropiate values and estimates an appropriate shape based on
   * averages and a polynomial interpolation of the boundary conditions on the
   * shape.
   */
  void estimate_and_initialize();

  /**
   * Returns the shift for a system-coordinate;
   */
  double get_m(double in_z) const;

  /**
   * Returns the tilt for a system-coordinate;
   */
  double get_v(double in_z) const;

  /**
   * Console output of the current Waveguide Structure.
   */
  void Print() const;

};