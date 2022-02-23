#include "BendTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <complex>
#include "../Core/Sector.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"

using namespace dealii;

BendTransformation::BendTransformation()
    : SpaceTransformation() {
}

BendTransformation::~BendTransformation() {}

Position BendTransformation::math_to_phys(Position ) const {
  Position ret;
  
  return ret;
}

Position BendTransformation::phys_to_math(Position ) const {
  Position ret;

  return ret;
}

Tensor<2, 3, ComplexNumber>
BendTransformation::get_Tensor(Position &position) {
  return get_Space_Transformation_Tensor(position);
}

Tensor<2, 3, double>
BendTransformation::get_Space_Transformation_Tensor(Position &)  {
  Tensor<2, 3, double> transformation;
  return transformation;
}

void BendTransformation::estimate_and_initialize() {
  return;
}

void BendTransformation::Print() const {
  //TODO
  std::cout << "Printing is not yet implemented." << std::endl;
}
