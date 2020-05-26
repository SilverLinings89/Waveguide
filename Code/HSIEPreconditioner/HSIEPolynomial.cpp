//
// Created by kraft on 08.11.19.
//

#include "HSIEPolynomial.h"
#include "../Helpers/staticfunctions.h"
#include "DofData.h"

bool HSIEPolynomial::matricesLoaded = false;
dealii::FullMatrix<std::complex<double>> HSIEPolynomial::D;
dealii::FullMatrix<std::complex<double>> HSIEPolynomial::I;

void HSIEPolynomial::computeDandI(unsigned int dimension,
                                  std::complex<double> k0) {
  HSIEPolynomial::D.reinit(dimension, dimension);
  for (unsigned int i = 0; i < dimension; i++) {
    for (unsigned int j = 0; j < dimension; j++) {
      HSIEPolynomial::D.set(i, j, matrixD(i, j, k0));
    }
  }

  HSIEPolynomial::I.copy_from(HSIEPolynomial::D);
  HSIEPolynomial::I.invert(HSIEPolynomial::D);
  HSIEPolynomial::matricesLoaded = true;
}

std::complex<double> HSIEPolynomial::evaluate(std::complex<double> x_in) {
  std::complex<double> ret(a[0]);
  std::complex<double> x = x_in;
  for (unsigned long i = 1; i < this->a.size(); i++) {
    ret += a[i] * x;
    x = x * x_in;
  }
  return ret;
}

std::complex<double> HSIEPolynomial::evaluate_dx(std::complex<double> x_in) {
  std::complex<double> ret(da[0]);
  std::complex<double> x = x_in;
  for (unsigned long i = 1; i < this->da.size(); i++) {
    ret += da[i] * x;
    x = x * x_in;
  }
  return ret;
}

HSIEPolynomial::HSIEPolynomial(unsigned int order, std::complex<double> in_k0) {
  this->a = std::vector<std::complex<double>>();
  this->da = std::vector<std::complex<double>>();
  for (unsigned int i = 0; i < order; i++) {
    this->a.emplace_back(0.0, 0.0);
  }
  this->a.emplace_back(1.0, 0.0);
  this->update_derivative();
  this->k0 = in_k0;
}


HSIEPolynomial::HSIEPolynomial(DofData &in_dof, std::complex<double> in_k0) {
  this->a = std::vector<std::complex<double>>();
  this->da = std::vector<std::complex<double>>();
  for (int i = 0; i < in_dof.hsie_order - 1; i++) {
    this->a.emplace_back(0.0, 0.0);
  }

  this->a.emplace_back(1.0, 0.0);

  this->update_derivative();
  this->k0 = in_k0;
}

HSIEPolynomial::HSIEPolynomial(std::vector<std::complex<double>> in_a,
                               std::complex<double> in_k0) {
  this->a = in_a;
  this->update_derivative();
  this->k0 = in_k0;
}

HSIEPolynomial HSIEPolynomial::applyD() {
  if (!this->matricesLoaded || this->a.size() > D.size()[0]) {
    this->computeDandI(this->a.size() + 2, this->k0);
  }
  std::vector<std::complex<double>> n_a;
  for (unsigned int i = 0; i < this->a.size(); i++) {
    std::complex<double> component(0, 0);
    for (unsigned int j = 0; j < this->a.size(); j++) {
      component += this->a[j] * this->D(i, j);
    }
    n_a.push_back(component);
  }
  return HSIEPolynomial(n_a, k0);
}

HSIEPolynomial HSIEPolynomial::applyI() {
  if (!this->matricesLoaded || this->a.size() > D.size()[0]) {
    this->computeDandI(this->a.size() + 2, this->k0);
  }
  std::vector<std::complex<double>> n_a;
  for (unsigned int i = 0; i < this->a.size(); i++) {
    std::complex<double> component(0, 0);
    for (unsigned int j = 0; j < this->a.size(); j++) {
      component += this->a[j] * this->I(i, j);
    }
    n_a.push_back(component);
  }
  return HSIEPolynomial(n_a, k0);
}

void HSIEPolynomial::update_derivative() {
  this->da = std::vector<std::complex<double>>();
  for (unsigned int i = 1; i < this->a.size(); i++) {
    this->da.emplace_back(i * this->a[i].real(), i * this->a[i].imag());
  }
}

void HSIEPolynomial::applyTplus(std::complex<double> u_0) {
  std::complex<double> temp_pre;
  std::complex<double> temp_post;
  temp_post = this->a[0];
  this->a[0] = u_0 + this->a[0];
  for (unsigned int i = 1; i < this->a.size(); i++) {
    temp_pre = this->a[i];
    this->a[i] += temp_post;
    temp_post = temp_pre;
  }
  this->a.push_back(temp_post);
  this->multiplyBy(0.5);
}

void HSIEPolynomial::applyTminus(std::complex<double> u_0) {
  std::complex<double> temp_pre;
  std::complex<double> temp_post;
  temp_post = this->a[0];
  this->a[0] = u_0 - this->a[0];
  for (unsigned int i = 1; i < this->a.size(); i++) {
    temp_pre = this->a[i];
    this->a[i] = temp_post - this->a[i];
    temp_post = temp_pre;
  }
  this->a.push_back(temp_post);
  this->multiplyBy(0.5);
}

void HSIEPolynomial::multiplyBy(double factor) {
  for (auto &i : this->a) {
    i *= factor;
  }
  this->update_derivative();
}

void HSIEPolynomial::multiplyBy(std::complex<double> factor) {
  for (auto &i : this->a) {
    i *= factor;
  }
  this->update_derivative();
}

HSIEPolynomial HSIEPolynomial::PsiMinusOne(std::complex<double> k0) {
  std::complex<double> one(1, 0);
  std::complex<double> i(0, 1);
  std::vector<std::complex<double>> a;
  a.emplace_back(0, 0);
  HSIEPolynomial ret(a, k0);
  ret.applyTminus(one);
  ret.multiplyBy(one / (i * k0));
  return ret;
}

HSIEPolynomial HSIEPolynomial::PsiJ(int j, std::complex<double> k0) {
  if (j == -1) {
    return HSIEPolynomial::PsiMinusOne(k0);
  }
  std::complex<double> one(1, 0);
  std::complex<double> i(0, 1);
  HSIEPolynomial ret(j, k0);
  ret.applyTminus(std::complex<double>(0, 0));
  ret.multiplyBy(one / (i * k0));
  return ret;
}

HSIEPolynomial HSIEPolynomial::PhiMinusOne(std::complex<double> k0) {
  std::complex<double> one(1, 0);
  std::vector<std::complex<double>> a;
  a.emplace_back(0, 0);
  HSIEPolynomial ret(a, k0);
  ret.applyTplus(one);
  return ret;
}

HSIEPolynomial HSIEPolynomial::PhiJ(int j, std::complex<double> k0) {
  if (j == -1) {
    return HSIEPolynomial::PhiMinusOne(k0);
  }
  std::complex<double> zero(0, 0);
  HSIEPolynomial ret(j, k0);
  ret.applyTplus(zero);
  return ret;
}

HSIEPolynomial HSIEPolynomial::ZeroPolynomial() {
  std::vector<std::complex<double>> inp;
  inp.emplace_back(0, 0);
  inp.emplace_back(0, 0);
  inp.emplace_back(0, 0);
  inp.emplace_back(0, 0);
  return HSIEPolynomial(inp, std::complex<double>(0, 0));
}

void HSIEPolynomial::add(HSIEPolynomial b) {
  if (b.a.size() > a.size()) {
    while (a.size() < b.a.size()) {
      a.emplace_back(0, 0);
    }
  }
  for (unsigned int i = 0; i < b.a.size(); i++) {
    a[i] += b.a[i];
  }
  update_derivative();
}

void HSIEPolynomial::applyDerivative() {
  update_derivative();
  a = da;
  update_derivative();
}
