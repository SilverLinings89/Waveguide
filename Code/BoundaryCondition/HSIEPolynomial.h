#pragma once

#include <deal.II/lac/full_matrix.h>
#include "DofData.h"
#include "../Core/Types.h"

/**
 * \class HSIEPolynomial
 * 
 * \brief This class basically represents a polynomial and its derivative. It is required for the HSIE implementation.
 * 
 * The core data in this class is a vector a, which stores the coefficients of the polynomials and a vector da, which stores the coefficients of the derivative. Both can be evaluated for a given x with the respective functions.
 * Additionally, there are functions to initialize a polynomial that are required by the hardy space infinite elements and some operators can be applied (like T_plus and T_minus).
 * As an important remark: The value kappa_0 used in HSIE is also kept in these values because we want to be able to apply the operators D and I to one a polynomial. Since they aren't cheap to compute, I precomute them once as static members of this class.
 * If you only intend to use evaluation, evaluation of the derivative, summation and multiplication with constants, then that value is not relevant.
 */

class HSIEPolynomial {
 public:
  std::vector<ComplexNumber> a; // This array stores the coefficients.
  std::vector<ComplexNumber> da; // This array stores the coefficients of the the derivative.
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
