/*
 * HSIEDofType.cpp
 *
 *  Created on: Oct 8, 2018
 *      Author: kraft
 */

#include "HSIEDofType.h"
#include <../Helpers/Parameters.h>
#include <../Helpers/staticfunctions.h>
#include <deal.II/base/tensor.h>

double k0 = 0.5;

template <int hsie_order>
HSIE_Dof_Type<hsie_order>::HSIE_Dof_Type(unsigned int in_type,
                                         unsigned int in_order) {
  type = in_type;
  order = in_order;
  base_point = -1;
  base_edge = -1;
  this->hardy_monomial_base = new std::vector<std::complex<double>>();
  this->IPsiK = new std::vector<std::complex<double>>();
  this->dxiPsiK = new std::vector<std::complex<double>>();
  this->hardy_monomial_base.resize(hsie_order + 1);
  this->IPsiK.resize(hsie_order + 1);
  this->dxiPsiK.resize(hsie_order + 1);
  for (unsigned int i = 0; i < hsie_order + 1; i++) {
    this->hardy_monomial_base[i] = std::complex<double>(0.0, 0.0);
    this->IPsiK[i] = std::complex<double>(0.0, 0.0);
    this->dxiPsiK[i] = std::complex<double>(0.0, 0.0);
  }
  this->hardy_monomial_base[in_order] = std::complex<double>(1.0, 0.0);
  if (this->D_and_I_initialized) {
    this->D = new dealii::Tensor<2, hsie_order + 1, std::complex<double>>();
    for (int i = 0; i < hsie_order + 1; i++) {
      for (int j = i; j < hsie_order + 1; j++) {
        this->D[i, j] = 0;
        this->D[i, j] = matrixD(i, j, k0);
        this->D[j, i] = matrixD(i, j, k0);
      }
    }
    this->I = invert(this->D);
    this->D_and_I_initialized = true;
  }
}

template <int hsie_order>
inline void HSIE_Dof_Type<hsie_order>::set_base_point(unsigned int in_bp) {
  if (in_bp < 4 && in_bp > -1) this->base_point = in_bp;
}

template <int hsie_order>
inline void HSIE_Dof_Type<hsie_order>::set_base_edge(unsigned int in_bp) {
  if (in_bp < 4 && in_bp > -1) this->base_edge = in_bp;
}

template <int hsie_order>
inline void HSIE_Dof_Type<hsie_order>::compute_IPsik() {
  std::vector<std::complex<double>> temp =
      new std::vector<std::complex<double>>();
  temp.resize(hsie_order + 1);
  for (int i = 0; i < hsie_order + 1; i++) {
    temp[i] = 0;
  }
  for (int i = 0; i < hsie_order; i++) {
    if (this->hardy_monomial_base[i] != 0) {
      temp[i] += 0.5 * this->hardy_monomial_base[i];
      temp[i + 1] += 0.5 * this->hardy_monomial_base[i];
    }
  }
  if (this->hardy_monomial_base[hsie_order] != 0) {
    std::cout << "Hardy Base Monomial overflow." << std::endl;
  }
  for (unsigned int i = 0; i < hsie_order + 1; i++) {
    std::complex<double> r(0, 0);
    for (unsigned int j = 0; j < hsie_order + 1; j++) {
      r += I[i, j] * temp[j];
    }
    this->IPsiK[i] = r;
  }
}

template <int hsie_order>
inline void HSIE_Dof_Type<hsie_order>::compute_dxiPsik() {
  std::vector<std::complex<double>> temp =
      new std::vector<std::complex<double>>();
  temp.resize(hsie_order + 1);
  for (int i = 0; i < hsie_order + 1; i++) {
    temp[i] = 0;
  }
  for (int i = 0; i < hsie_order; i++) {
    if (this->hardy_monomial_base[i] != 0) {
      temp[i] += 0.5 * this->hardy_monomial_base[i];
      temp[i + 1] += 0.5 * this->hardy_monomial_base[i];
    }
  }
  for (unsigned int i = 0; i < hsie_order; i++) {
    this->dxiPsiK[i] = (i + 1) * temp[i + 1];
  }
  this->dxiPsiK[hsie_order] = 0;
}

template <int hsie_order>
inline std::complex<double> HSIE_Dof_Type<hsie_order>::eval_base(
    std::complex<double>* in_base, std::complex<double> in_x) {
  std::complex<double> ret(0, 0);
  std::complex<double> x_temp(1.0, 0.0);
  for (unsigned int i = 0; i < hsie_order; i++) {
    ret += x_temp * in_base[i];
    x_temp *= in_x;
  }
  ret += x_temp * in_base[hsie_order];
  return ret;
}

template <int hsie_order>
inline unsigned int HSIE_Dof_Type<hsie_order>::get_order() {
  return hsie_order;
}

template <int hsie_order>
inline unsigned int HSIE_Dof_Type<hsie_order>::get_type() {
  return this->type;
}

template <int hsie_order>
HSIE_Dof_Type<hsie_order>::~HSIE_Dof_Type() {
  // TODO Auto-generated destructor stub
}

template <int hsie_order>
inline void HSIE_Dof_Type<hsie_order>::set_base_point(unsigned int point) {
  if (this->type == 3) {
    this->base_point = point;
  }
}

template <int hsie_order>
std::vector<std::complex<double>> HSIE_Dof_Type<hsie_order>::evaluate_U(
    dealii::Point<2, double> in_x, double in_xi) {
  std::vector<std::complex<double>> ret =
      new std::vector<std::complex<double>>();
  ret.resize(3);
  if (type == 0) {
    // U_1 = 0;
  }
  if (type == 2) {
    // U_2 = U_3 = 0;
  }
  if (type == 3) {
    // U_1 = 0;
  }
  return ret;
}

template <int hsie_order>
std::vector<std::complex<double>> HSIE_Dof_Type<hsie_order>::evaluate_U_for_ACT(
    dealii::Point<2, double> in_x, double in_xi) {
  std::vector<std::complex<double>> ret =
      new std::vector<std::complex<double>>();
  ret.resize(3);
  if (type == 0) {
    // entries in every component
  }
  if (type == 2) {
    // Comp 1 = 0;
  }
  if (type == 3) {
    // entries in every component
  }
  return ret;
}
