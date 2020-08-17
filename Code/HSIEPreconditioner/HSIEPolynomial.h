#pragma once

#include <deal.II/lac/full_matrix.h>
#include "DofData.h"
#include "../Core/Types.h"

class HSIEPolynomial {
 public:
  std::vector<ComplexNumber> a;
  std::vector<ComplexNumber> da;
  ComplexNumber k0;
  ComplexNumber evaluate(ComplexNumber x);
  ComplexNumber evaluate_dx(ComplexNumber x);
  void update_derivative();
  static void computeDandI(unsigned int,ComplexNumber );
  static HSIEPolynomial PsiMinusOne(ComplexNumber k0);
  static HSIEPolynomial PsiJ(int j, ComplexNumber k0);
  static HSIEPolynomial ZeroPolynomial();

  static HSIEPolynomial PhiMinusOne(ComplexNumber k0);
  static HSIEPolynomial PhiJ(int j, ComplexNumber k0);

  HSIEPolynomial(unsigned int,ComplexNumber );
  HSIEPolynomial(DofData&, ComplexNumber);
  HSIEPolynomial(std::vector<ComplexNumber> in_a,
                 ComplexNumber k0);

  static bool matricesLoaded;
  static dealii::FullMatrix<ComplexNumber> D;
  static dealii::FullMatrix<ComplexNumber> I;

  HSIEPolynomial applyD();
  HSIEPolynomial applyI();

  void multiplyBy(ComplexNumber factor);
  void multiplyBy(double factor);
  void applyTplus(ComplexNumber u_0);
  void applyTminus(ComplexNumber u_0);
  void applyDerivative();
  void add(HSIEPolynomial b);
};
