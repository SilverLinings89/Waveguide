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
bool HSIE_Dof_Type<hsie_order>::D_and_I_initialized = false;

template <int hsie_order>
HSIE_Dof_Type<hsie_order>::HSIE_Dof_Type(unsigned int in_type,
                                         unsigned int in_order,
                                         unsigned int in_q_count) {
  this->v_k = new std::vector<Tensor<1, 2>>();
  this->w_k = new std::vector<double>();
  v_k.resize(in_q_count);
  w_k.resize(in_q_count);
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
  this->y_length = 1.0;
  this->x_length = 1.0;
}

template <int hsie_order>
inline void HSIE_Dof_Type<hsie_order>::set_base_point(unsigned int in_bp) {
  if (in_bp < 4 && in_bp > -1) this->base_point = in_bp;
}

template <int hsie_order>
void HSIE_Dof_Type<hsie_order>::prepare_for_quadrature_points(
    std::vector<dealii::Point<2, double>> q_points) {
  dealii::MappingQ1 temp_mapping(1);
  PolynomialsNedelec<2> temp = new PolynomialsNedelec<2>(0);
  std::vector<Tensor<1, 2>> v_k_storage = new std::vector<Tensor<1, 2>>();
  std::vector<Tensor<2, 2>> v_k_n_2 = new std::vector<Tensor<2, 2>>();
  std::vector<Tensor<3, 2>> v_k_n_3 = new std::vector<Tensor<3, 2>>();
  std::vector<Tensor<4, 2>> v_k_n_4 = new std::vector<Tensor<4, 2>>();
  std::vector<Tensor<5, 2>> v_k_n_5 = new std::vector<Tensor<5, 2>>();
  v_k_storage.resize(1);
  for (unsigned int i = 0; i < q_points.size(); i++) {
    temp.compute(q_points[i], v_k_storage, v_k_n_2, v_k_n_3, v_k_n_4, v_k_n_5);
    this->v_k[i] = v_k_storage[0];
    this->w_k[i] = (1.0 - q_points[i][0]) * (1.0 - q_points[i][1]);
  }
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
    temp[i] += 0.5 * this->hardy_monomial_base[i];
    temp[i + 1] += 0.5 * this->hardy_monomial_base[i];
  }
  if (this->hardy_monomial_base[hsie_order] != std::complex<double>(0.0, 0.0)) {
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
    temp[i] += 0.5 * this->hardy_monomial_base[i];
    temp[i + 1] += 0.5 * this->hardy_monomial_base[i];
  }
  for (unsigned int i = 0; i < hsie_order; i++) {
    this->dxiPsiK[i] = std::complex<double>(i + 1, 0) * temp[i + 1];
  }
  this->dxiPsiK[hsie_order] = 0;
}

template <int hsie_order>
std::complex<double> HSIE_Dof_Type<hsie_order>::eval_base(
    std::vector<std::complex<double>>* in_base, std::complex<double> in_x) {
  std::complex<double> ret(0, 0);
  std::complex<double> x_temp(1.0, 0.0);
  for (unsigned int i = 0; i < hsie_order; i++) {
    ret += x_temp * in_base->operator[](i);
    x_temp *= in_x;
  }
  ret += x_temp * in_base->operator[](hsie_order);
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
std::vector<std::complex<double>> HSIE_Dof_Type<hsie_order>::evaluate_U(
    dealii::Point<2, double> in_x, double in_xi) {
  std::vector<std::complex<double>> ret =
      new std::vector<std::complex<double>>();
  ret.resize(3);
  // Edge Functions
  if (type == 0) {
    // U_1 = 0;
  }
  // Ray Functions
  if (type == 2) {
    // U_2 = U_3 = 0;
  }
  // Infinite Face Functions
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

template <int hsie_order>
std::complex<double> HSIE_Dof_Type<hsie_order>::component_1a(
    double x, double y, std::complex<double> xhi) {
  std::complex<double> ret(0, 0);
  if (type == 0) {
    return ret;
  }
  if (type == 2) {
    double dy = 1 / this->y_length;
    if (this->base_point == 0 || this->base_point == 3) {
      dy *= -1.0;
    }
    if (this->base_point == 0 || this->base_point == 1) {
      dy *= x / x_length;
    } else {
      dy *= (x_length - x) / x_length;
    }
    std::vector<std::complex<double>> temp_vec =
        this->apply_T_plus(this->hardy_monomial_base, 0.0);
    ret = eval_base(&temp_vec, xhi);
    return ret;
  }
  if (type == 3) {
    return ret;
  }
  return ret;
}

template <int hsie_order>
std::complex<double> HSIE_Dof_Type<hsie_order>::component_1b(
    double x, double y, std::complex<double> xhi) {
  std::complex<double> ret(0, 0);
  if (type == 0) {
    return ret;
  }
  if (type == 2) {
    return ret;
  }
  if (type == 3) {
    // entries in every component
  }
  return ret;
}

template <int hsie_order>
std::complex<double> HSIE_Dof_Type<hsie_order>::component_2a(
    double x, double y, std::complex<double> xhi) {
  std::complex<double> ret(0, 0);
  if (type == 0) {
    return ret;
  }
  if (type == 2) {
    return ret;
  }
  if (type == 3) {
    // entries in every component
  }
  return ret;
}

template <int hsie_order>
std::complex<double> HSIE_Dof_Type<hsie_order>::component_2b(
    double x, double y, std::complex<double> xhi) {
  std::complex<double> ret(0, 0);
  if (type == 0) {
    return ret;
  }
  if (type == 2) {
    double dx = 1 / this->x_length;
    if (this->base_point == 0 || this->base_point == 1) {
      dx *= -1.0;
    }
    if (this->base_point == 0 || this->base_point == 3) {
      dx *= y / y_length;
    } else {
      dx *= (y_length - y) / y_length;
    }
    std::vector<std::complex<double>> temp_vec =
        this->apply_T_plus(this->hardy_monomial_base, 0.0);
    ret = eval_base(&temp_vec, xhi);
    return ret;
  }
  if (type == 3) {
    return ret;
  }
  return ret;
}

template <int hsie_order>
std::complex<double> HSIE_Dof_Type<hsie_order>::component_3a(
    double x, double y, std::complex<double> xhi) {
  std::complex<double> ret(0, 0);
  if (type == 0) {
    // entries in every component
  }
  if (type == 2) {
    return ret;
  }
  if (type == 3) {
    // entries in every component
  }
  return ret;
}

template <int hsie_order>
std::complex<double> HSIE_Dof_Type<hsie_order>::component_3b(
    double x, double y, std::complex<double> xhi) {
  std::complex<double> ret(0, 0);
  if (type == 0) {
    // entries in every component
  }
  if (type == 2) {
    return ret;
  }
  if (type == 3) {
    // entries in every component
  }
  return ret;
}

template <int hsie_order>
std::vector<std::complex<double>> HSIE_Dof_Type<hsie_order>::apply_T_plus(
    std::vector<std::complex<double>> in_base_vector, double u0) {
  std::vector<std::complex<double>> ret =
      new std::vector<std::complex<double>>();
  ret.resize(hsie_order + 1);
  ret[0] = 0.5 * u0 + in_base_vector[0];
  for (unsigned int i = 1; i <= hsie_order; i++) {
    ret[i] = in_base_vector[i - 1] + in_base_vector[i];
  }
}

template <int hsie_order>
std::vector<std::complex<double>> HSIE_Dof_Type<hsie_order>::apply_T_minus(
    std::vector<std::complex<double>> in_base_vector, double u0) {
  std::vector<std::complex<double>> ret =
      new std::vector<std::complex<double>>();
  ret.resize(hsie_order + 1);
  ret[0] = 0.5 * u0 + in_base_vector[0];
  for (unsigned int i = 1; i <= hsie_order; i++) {
    ret[i] = in_base_vector[i - 1] - in_base_vector[i];
  }
}