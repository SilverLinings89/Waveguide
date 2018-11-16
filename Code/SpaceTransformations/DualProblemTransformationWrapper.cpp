/*
 * DualProblemTransformationWrapper.cpp
 *
 *  Created on: Jan 10, 2017
 *      Author: pascal
 */

#ifndef DualTransformationWrapper_CPP
#define DualTransformationWrapper_CPP

#include "DualProblemTransformationWrapper.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "../Core/Sector.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "../Helpers/staticfunctions.h"
#include "SpaceTransformation.h"

using namespace dealii;

DualProblemTransformationWrapper::DualProblemTransformationWrapper(
    SpaceTransformation *in_st, int inner_rank)
    : SpaceTransformation(3, inner_rank),
      XMinus(-(GlobalParams.M_R_XLength * 0.5 - GlobalParams.M_BC_XMinus)),
      XPlus(GlobalParams.M_R_XLength * 0.5 - GlobalParams.M_BC_XPlus),
      YMinus(-(GlobalParams.M_R_YLength * 0.5 - GlobalParams.M_BC_YMinus)),
      YPlus(GlobalParams.M_R_YLength * 0.5 - GlobalParams.M_BC_YPlus),
      ZMinus(-GlobalParams.M_R_ZLength * 0.5),
      ZPlus(GlobalParams.M_R_ZLength * 0.5),
      epsilon_K(GlobalParams.M_W_epsilonin),
      epsilon_M(GlobalParams.M_W_epsilonout),
      sectors(GlobalParams.M_W_Sectors),
      deltaY(GlobalParams.M_W_Delta) {
  st = in_st;
  homogenized = st->homogenized;
}

Point<3> DualProblemTransformationWrapper::math_to_phys(Point<3> coord) const {
  return st->math_to_phys(coord);
}

Point<3> DualProblemTransformationWrapper::phys_to_math(Point<3> coord) const {
  return st->phys_to_math(coord);
}

bool DualProblemTransformationWrapper::PML_in_X(Point<3> &p) const {
  return st->PML_in_X(p);
}

bool DualProblemTransformationWrapper::PML_in_Y(Point<3> &p) const {
  return st->PML_in_Y(p);
}

bool DualProblemTransformationWrapper::PML_in_Z(Point<3> &p) const {
  return st->PML_in_Z(p);
}

double DualProblemTransformationWrapper::Preconditioner_PML_Z_Distance(
    Point<3> &p, unsigned int block) const {
  return st->Preconditioner_PML_Z_Distance(p, block);
}

double DualProblemTransformationWrapper::PML_X_Distance(Point<3> &p) const {
  return st->PML_X_Distance(p);
}

double DualProblemTransformationWrapper::PML_Y_Distance(Point<3> &p) const {
  return st->PML_Y_Distance(p);
}

double DualProblemTransformationWrapper::PML_Z_Distance(Point<3> &p) const {
  return st->PML_Z_Distance(p);
}

dealii::Point<3, double> transform_position(Point<3> in_position) {
  Point<3> ret = in_position;
  ret[2] = -ret[2];
  // ret[2] += GlobalParams.M_BC_Zplus*GlobalParams.SectorThickness;
  return ret;
}

Tensor<2, 3, double>
DualProblemTransformationWrapper::get_Space_Transformation_Tensor_Homogenized(
    Point<3> &position) const {
  std::cout << "This should never be called: "
               "DualProblemTransformationWrapper::get_Space_Transformation_"
               "Tensor_Homogenized"
            << std::endl;
  return st->get_Space_Transformation_Tensor_Homogenized(position);
}

Tensor<2, 3, double>
DualProblemTransformationWrapper::get_Space_Transformation_Tensor(
    Point<3> &position) const {
  std::cout
      << "This should never be called: "
         "DualProblemTransformationWrapper::get_Space_Transformation_Tensor"
      << std::endl;
  return st->get_Space_Transformation_Tensor(position);
}

Tensor<2, 3, std::complex<double>>
DualProblemTransformationWrapper::Apply_PML_To_Tensor(
    Point<3> &, Tensor<2, 3, double> transformation) const {
  std::cout << "This function should never be called: "
               "DualProblemTransformationWrapper::Apply_PML_To_Tensor"
            << std::endl;

  Tensor<2, 3, std::complex<double>> ret2;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      ret2[i][j] = transformation[i][j] * std::complex<double>(1.0, 0.0);
    }
  }

  return ret2;
}

Tensor<2, 3, std::complex<double>>
DualProblemTransformationWrapper::Apply_PML_To_Tensor_For_Preconditioner(
    Point<3> &, Tensor<2, 3, double> transformation, int) const {
  std::cout << "This function should never be called: "
               "DualProblemTransformationWrapper::Apply_PML_To_Tensor_For_"
               "Preconditioner"
            << std::endl;

  Tensor<2, 3, std::complex<double>> ret2;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      ret2[i][j] = transformation[i][j] * std::complex<double>(1.0, 0.0);
    }
  }

  return ret2;
}

Tensor<2, 3, std::complex<double>> DualProblemTransformationWrapper::get_Tensor(
    Point<3> &position) const {
  Point<3> p = transform_position(position);

  Tensor<2, 3, double> transformation;

  if (homogenized) {
    transformation = st->get_Space_Transformation_Tensor_Homogenized(p);
  } else {
    transformation = st->get_Space_Transformation_Tensor(p);
  }

  Tensor<2, 3, std::complex<double>> ret =
      st->Apply_PML_To_Tensor(position, transformation);

  return ret;
}

Tensor<2, 3, std::complex<double>>
DualProblemTransformationWrapper::get_Preconditioner_Tensor(Point<3> &position,
                                                            int block) const {
  Point<3> p = transform_position(position);

  Tensor<2, 3, double> transformation;

  if (homogenized) {
    transformation = st->get_Space_Transformation_Tensor_Homogenized(p);
  } else {
    transformation = st->get_Space_Transformation_Tensor(p);
  }

  Tensor<2, 3, std::complex<double>> ret =
      st->Apply_PML_To_Tensor_For_Preconditioner(position, transformation,
                                                 block);

  return ret;
}

std::complex<double> DualProblemTransformationWrapper::gauss_product_2D_sphere(
    double z, int n, double R, double Xc, double Yc, Waveguide *in_w,
    Evaluation_Metric in_m) {
  return st->gauss_product_2D_sphere(z, n, R, Xc, Yc, in_w, in_m);
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

int DualProblemTransformationWrapper::Z_to_Layer(double z) const {
  return st->Z_to_Layer(z);
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

#endif
