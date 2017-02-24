/*
 * DualProblemTransformationWrapper.cpp
 *
 *  Created on: Jan 10, 2017
 *      Author: pascal
 */

#ifndef DualTransformationWrapper_CPP
#define DualTransformationWrapper_CPP

#include "../Helpers/staticfunctions.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "DualProblemTransformationWrapper.h"

using namespace dealii;

DualProblemTransformationWrapper::DualProblemTransformationWrapper (SpaceTransformation * in_st, int in_rank, int inner_rank):
    SpaceTransformation(3, inner_rank),
  XMinus( -(GlobalParams.M_R_XLength *0.5 - GlobalParams.M_BC_XMinus)),
  XPlus( GlobalParams.M_R_XLength *0.5 - GlobalParams.M_BC_XPlus),
  YMinus( -(GlobalParams.M_R_YLength *0.5 - GlobalParams.M_BC_YMinus)),
  YPlus( GlobalParams.M_R_YLength *0.5 - GlobalParams.M_BC_YPlus),
  ZMinus( - GlobalParams.M_R_ZLength *0.5 ),
  ZPlus( GlobalParams.M_R_ZLength *0.5 ),
  epsilon_K(GlobalParams.M_W_epsilonin),
  epsilon_M(GlobalParams.M_W_epsilonout),
  sectors(GlobalParams.M_W_Sectors),
  deltaY(GlobalParams.M_W_Delta)
{
  st = in_st;
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


double DualProblemTransformationWrapper::Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block ) const{
  return st->Preconditioner_PML_Z_Distance(p, block);
}

double DualProblemTransformationWrapper::PML_X_Distance(Point<3> &p) const{
  return st->PML_X_Distance(p);
}

double DualProblemTransformationWrapper::PML_Y_Distance(Point<3> &p) const{
  return st->PML_Y_Distance(p);
}

double DualProblemTransformationWrapper::PML_Z_Distance(Point<3> &p) const{
  return st->PML_Z_Distance(p);
}

Tensor<2,3, std::complex<double>> DualProblemTransformationWrapper::get_Tensor(Point<3> & position) const {
  // Point<3> p = position;
  // p[2] = (GlobalParams.M_R_ZLength/2.0) - p[2];
  return st->get_Tensor(position);
}

Tensor<2,3, std::complex<double>> DualProblemTransformationWrapper::get_Preconditioner_Tensor(Point<3> & position, int block) const {
  // Point<3> p = position;
  // p[2] = (GlobalParams.M_R_ZLength/2.0) - p[2];
  return st->get_Preconditioner_Tensor(position,block);
}

std::complex<double> DualProblemTransformationWrapper::gauss_product_2D_sphere(double z, int n, double R, double Xc, double Yc, Waveguide * in_w)
{
  return gauss_product_2D_sphere(z,n,R, Xc, Yc, in_w);
}

std::complex<double> DualProblemTransformationWrapper::evaluate_for_z(double in_z, Waveguide * in_w) {
  return st->evaluate_for_z(in_z, in_w);
}

void DualProblemTransformationWrapper::estimate_and_initialize() {
  st->estimate_and_initialize();
  return;
}

double DualProblemTransformationWrapper::get_Q1 (double z) const {
  return st->get_Q1(z);
}

double DualProblemTransformationWrapper::get_Q2 (double z) const {
  return st->get_Q2(z);
}

double DualProblemTransformationWrapper::get_Q3 (double z) const {
  return st->get_Q3(z);
}

double DualProblemTransformationWrapper::get_dof( int dof) const {
  return st->get_dof(dof);
}

void DualProblemTransformationWrapper::set_dof(int dof, double value) {
  return st->set_dof(dof, value);
}

double DualProblemTransformationWrapper::get_free_dof( int dof) const {
  return st->get_free_dof(dof);
}

void DualProblemTransformationWrapper::set_free_dof(int dof, double value) {
  return st->set_free_dof(dof, value);
}

std::pair<int, double> DualProblemTransformationWrapper::Z_to_Sector_and_local_z(double in_z) const {
  return st->Z_to_Sector_and_local_z(-in_z);
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

int DualProblemTransformationWrapper::Z_to_Layer( double z) const {
  return st->Z_to_Layer(z);
}

Vector<double> DualProblemTransformationWrapper::Dofs() const {
  return st->Dofs();
}

unsigned int DualProblemTransformationWrapper::NFreeDofs()  const{
  return st->NFreeDofs();
}

unsigned int DualProblemTransformationWrapper::NDofs() const {
  return st->NDofs();
}

bool DualProblemTransformationWrapper::IsDofFree(int input) const {
  return st->IsDofFree(input);
}

void DualProblemTransformationWrapper::Print() const {
  return st->Print();
}

#endif



