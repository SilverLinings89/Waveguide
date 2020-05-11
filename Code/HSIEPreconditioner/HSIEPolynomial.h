//
// Created by kraft on 08.11.19.
//

#ifndef WAVEGUIDEPROBLEM_HSIEPOLYNOMIAL_H
#define WAVEGUIDEPROBLEM_HSIEPOLYNOMIAL_H

#include <deal.II/base/point.h>
#include <deal.II/lac/full_matrix.h>
#include "DofData.h"
#include <complex>

class HSIEPolynomial {
 public:
  std::vector<std::complex<double>> a;
  std::vector<std::complex<double>> da;
  std::complex<double> k0;
  std::complex<double> evaluate(std::complex<double> x);
  std::complex<double> evaluate_dx(std::complex<double> x);
  void update_derivative();
  static void computeDandI(unsigned int, std::complex<double>);
  static HSIEPolynomial PsiMinusOne(std::complex<double> k0);
  static HSIEPolynomial PsiJ(int j, std::complex<double> k0);
  static HSIEPolynomial ZeroPolynomial();

  static HSIEPolynomial PhiMinusOne(std::complex<double> k0);
  static HSIEPolynomial PhiJ(int j, std::complex<double> k0);

  HSIEPolynomial(unsigned int, std::complex<double>);
  HSIEPolynomial(DofData&, std::complex<double>);
  HSIEPolynomial(std::vector<std::complex<double>> in_a,
                 std::complex<double> k0);

  static bool matricesLoaded;
  static dealii::FullMatrix<std::complex<double>> D;
  static dealii::FullMatrix<std::complex<double>> I;

  HSIEPolynomial applyD();
  HSIEPolynomial applyI();

  void multiplyBy(std::complex<double> factor);
  void multiplyBy(double factor);
  void applyTplus(std::complex<double> u_0);
  void applyTminus(std::complex<double> u_0);
  void applyDerivative();
  void add(HSIEPolynomial b);
};

#endif  // WAVEGUIDEPROBLEM_HSIEPOLYNOMIAL_H
