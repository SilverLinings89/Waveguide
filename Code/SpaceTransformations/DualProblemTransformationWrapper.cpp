#include "DualProblemTransformationWrapper.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "../Core/Sector.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "../Helpers/staticfunctions.h"
#include "SpaceTransformation.h"

using namespace dealii;

DualProblemTransformationWrapper::DualProblemTransformationWrapper(
    SpaceTransformation *in_st)
    : SpaceTransformation(3),
      epsilon_K(GlobalParams.Epsilon_R_in_waveguide),
      epsilon_M(GlobalParams.Epsilon_R_outside_waveguide),
      sectors(GlobalParams.Number_of_sectors),
      deltaY(GlobalParams.Vertical_displacement_of_waveguide) {
  st = in_st;
  homogenized = st->homogenized;
}

Position DualProblemTransformationWrapper::math_to_phys(Position coord) const {
  return st->math_to_phys(coord);
}

Position DualProblemTransformationWrapper::phys_to_math(Position coord) const {
  return st->phys_to_math(coord);
}

Position transform_position(Position in_position) {
  Position ret = in_position;
  ret[2] = -ret[2];
  // ret[2] += GlobalParams.M_BC_Zplus*GlobalParams.Sector_thickness;
  return ret;
}

Tensor<2, 3, double>
DualProblemTransformationWrapper::get_Space_Transformation_Tensor_Homogenized(
    Position &position) const {
  std::cout << "This should never be called: "
               "DualProblemTransformationWrapper::get_Space_Transformation_"
               "Tensor_Homogenized"
            << std::endl;
  return st->get_Space_Transformation_Tensor_Homogenized(position);
}

Tensor<2, 3, double>
DualProblemTransformationWrapper::get_Space_Transformation_Tensor(
    Position &position) const {
  std::cout
      << "This should never be called: "
         "DualProblemTransformationWrapper::get_Space_Transformation_Tensor"
      << std::endl;
  return st->get_Space_Transformation_Tensor(position);
}

Tensor<2, 3, ComplexNumber> DualProblemTransformationWrapper::get_Tensor(
    Position &position) const {
  Position p = transform_position(position);

  Tensor<2, 3, double> transformation;

  if (homogenized) {
    transformation = st->get_Space_Transformation_Tensor_Homogenized(p);
  } else {
    transformation = st->get_Space_Transformation_Tensor(p);
  }

  return transformation;
}

void DualProblemTransformationWrapper::estimate_and_initialize() {
  st->estimate_and_initialize();
  return;
}

double DualProblemTransformationWrapper::get_Q1(double z) const {
  return st->get_Q1(z);
}

double DualProblemTransformationWrapper::get_Q2(double z) const {
  return st->get_Q2(z);
}

double DualProblemTransformationWrapper::get_Q3(double z) const {
  return st->get_Q3(z);
}

double DualProblemTransformationWrapper::get_dof(int dof) const {
  return st->get_dof(dof);
}

void DualProblemTransformationWrapper::set_dof(int dof, double value) {
  return st->set_dof(dof, value);
}

double DualProblemTransformationWrapper::get_free_dof(int dof) const {
  return st->get_free_dof(dof);
}

void DualProblemTransformationWrapper::set_free_dof(int dof, double value) {
  return st->set_free_dof(dof, value);
}

std::pair<int, double>
DualProblemTransformationWrapper::Z_to_Sector_and_local_z(double in_z) const {
  return st->Z_to_Sector_and_local_z(in_z);
}

double DualProblemTransformationWrapper::Sector_Length() const {
  return st->Sector_Length();
}

double DualProblemTransformationWrapper::get_r(double in_z) const {
  return st->get_r(in_z);
}

double DualProblemTransformationWrapper::get_m(double in_z) const {
  return st->get_m(in_z);
}

double DualProblemTransformationWrapper::get_v(double in_z) const {
  return st->get_v(in_z);
}

Vector<double> DualProblemTransformationWrapper::Dofs() const {
  return st->Dofs();
}

unsigned int DualProblemTransformationWrapper::NFreeDofs() const {
  return st->NFreeDofs();
}

unsigned int DualProblemTransformationWrapper::NDofs() const {
  return st->NDofs();
}

bool DualProblemTransformationWrapper::IsDofFree(int input) const {
  return st->IsDofFree(input);
}

void DualProblemTransformationWrapper::Print() const { return st->Print(); }
