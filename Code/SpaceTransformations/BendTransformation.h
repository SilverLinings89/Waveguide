#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <math.h>
#include <vector>

#include "../Core/InnerDomain.h"
#include "../Core/Sector.h"
#include "./SpaceTransformation.h"

/**
 * \class BendTransformation
 * \brief This transformation maps a 90-degree bend of a waveguide to a straight waveguide.
 *
 * This transformation determines the full arch-length of the 90-degree bend as the length given as the global-z-length of the system. 
 * It can then determine all properties of the transformation. The computation of the material tensors is performed via symbolic differentiation
 * instead of the version chosen in other transformations. This ansatz is therefore the one most easy to use for a new transformation.
 * 
 * The bend transformation also has internal sectors for the option of shape transformation. The y-shifts represent an inward or outward shift in radial direction, the width remains the same.
 * 
 * \author Pascal Kraft 
 * \date 14.12.2021
 */

class BendTransformation : public SpaceTransformation {
 public:
  BendTransformation();

  virtual ~BendTransformation();

  Position math_to_phys(Position coord) const override;

  Position phys_to_math(Position coord) const override;

  dealii::Tensor<2, 3, ComplexNumber> get_Tensor(Position &coordinate) override;

  dealii::Tensor<2, 3, double> get_Space_Transformation_Tensor(Position &coordinate) override;

  void estimate_and_initialize() override;

  void Print() const override;
};
