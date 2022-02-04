#include "AngleWaveguideTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "../Core/Sector.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "../Helpers/staticfunctions.h"
#include "SpaceTransformation.h"

using namespace dealii;

AngleWaveguideTransformation::AngleWaveguideTransformation()
    : SpaceTransformation(3),
      sectors(GlobalParams.Number_of_sectors) {
  homogenized = false;
  transformation_tensor[0][0] = 1;
  transformation_tensor[0][1] = 0;
  transformation_tensor[0][2] = 0;

  transformation_tensor[1][0] = 0;
  transformation_tensor[1][1] = 2;
  transformation_tensor[1][2] = -1;

  transformation_tensor[2][0] = 0;
  transformation_tensor[2][1] = -1;
  transformation_tensor[2][2] = 1;
  
  std::vector<double> qs;
  qs.push_back(1);
  qs.push_back(std::sqrt(2));
  qs.push_back(1);

  for(unsigned int i = 0; i < 3; i++) {
    for(unsigned int j = 0; j < 3; j++) {
        transformation_tensor[i][j] *= std::sqrt(2) / (qs[i]*qs[j]);
    }
  }
}

AngleWaveguideTransformation::~AngleWaveguideTransformation() {}

Position AngleWaveguideTransformation::math_to_phys(Position coord) const {
  Position ret;
  ret[0] = coord[0];
  ret[1] = coord[1];
  ret[2] = coord[2] + coord[1];
  return ret;
}

Position AngleWaveguideTransformation::phys_to_math(Position coord) const {
  Position ret;
  ret[0] = coord[0];
  ret[1] = coord[1];
  ret[2] = coord[2] - coord[1];
  return ret;
}

Tensor<2, 3, ComplexNumber>
AngleWaveguideTransformation::get_Tensor(Position &position) const {
  return get_Space_Transformation_Tensor(position);
}

double AngleWaveguideTransformation::get_dof(int) const {
  std::cout << "Getting dof of transformation that has no dofs." << std::endl;
  return 0.0;
}

double AngleWaveguideTransformation::get_free_dof(int) const {
  std::cout << "Getting dof of transformation that has no dofs." << std::endl;
  return 0.0;
}

void AngleWaveguideTransformation::set_dof(int , double) {
  std::cout << "Setting dof of transformation that has no dofs." << std::endl;
}

void AngleWaveguideTransformation::set_free_dof(int ,double) {
  std::cout << "Setting dof of transformation that has no dofs." << std::endl;
}

double AngleWaveguideTransformation::Sector_Length() const {
  return GlobalParams.Sector_thickness;
}

void AngleWaveguideTransformation::estimate_and_initialize() {
  
}

double AngleWaveguideTransformation::get_r(double) const {
  std::cout << "Asking for Radius of rectangular Waveguide." << std::endl;
  return 0;
}

double AngleWaveguideTransformation::get_m(double) const {
  return 0;
}

double AngleWaveguideTransformation::get_v(double) const {
  return 0;
}

double AngleWaveguideTransformation::get_Q1(double) const {
  return 1.0;
}

double AngleWaveguideTransformation::get_Q2(double) const {
  return std::sqrt(2);
}

double AngleWaveguideTransformation::get_Q3(double) const {
  return 1;
}

Vector<double> AngleWaveguideTransformation::Dofs() const {
  Vector<double> ret;
  return ret;
}

unsigned int AngleWaveguideTransformation::NFreeDofs() const {
  return 0;
}

bool AngleWaveguideTransformation::IsDofFree(int) const {
  return false;
}

void AngleWaveguideTransformation::Print() const {
  std::cout << "Printing is not yet implemented." << std::endl;
}

unsigned int AngleWaveguideTransformation::NDofs() const {
  return 0;
}

Tensor<2, 3, double>
AngleWaveguideTransformation::get_Space_Transformation_Tensor(Position &) const {
  return transformation_tensor;
}
