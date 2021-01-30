#include "../Code/Core/Types.h"
#include "../Code/BoundaryCondition/HSIEPolynomial.h"

#include <deal.II/lac/full_matrix.h>

TEST(HSIEPolynomialTests, TestOperatorTplus) {
  std::vector<ComplexNumber> in_a;
  in_a.emplace_back(0.0, 0.0);
  in_a.emplace_back(0.0, 0.0);
  in_a.emplace_back(1.0, 0.0);
  in_a.emplace_back(0.0, 1.0);
  in_a.emplace_back(0.0, 0.0);
  ComplexNumber k0(0.0, 1.0);
  HSIEPolynomial poly(in_a, k0);
  poly.applyTplus(ComplexNumber(1, -1));
  ASSERT_EQ(poly.a[0], ComplexNumber(0.5, -0.5));
  ASSERT_EQ(poly.a[1], ComplexNumber(0, 0));
  ASSERT_EQ(poly.a[2], ComplexNumber(0.5, 0));
  ASSERT_EQ(poly.a[3], ComplexNumber(0.5, 0.5));
  ASSERT_EQ(poly.a[4], ComplexNumber(0, 0.5));
}

TEST(HSIEPolynomialTests, ProductOfDandIShouldBeIdentity) {
  std::vector<ComplexNumber> in_a;
  in_a.emplace_back(0.0, 0.0);
  in_a.emplace_back(0.0, 0.0);
  in_a.emplace_back(1.0, 0.0);
  in_a.emplace_back(0.0, 1.0);
  in_a.emplace_back(0.0, 0.0);
  ComplexNumber k0(0.0, 1.0);
  HSIEPolynomial poly(in_a, k0);
  poly.applyD();
  dealii::FullMatrix<ComplexNumber> product(HSIEPolynomial::D.size(0),
      HSIEPolynomial::D.size(1));
  HSIEPolynomial::D.mmult(product, HSIEPolynomial::I, false);
  ASSERT_NEAR(product.frobenius_norm(), std::sqrt(HSIEPolynomial::I.size(0)),
      0.0001);
}