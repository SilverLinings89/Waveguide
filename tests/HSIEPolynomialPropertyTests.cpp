// Algebraic properties of the polynomials that represent functions in the Hardy space (HSIEPolynomial).
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <deal.II/lac/full_matrix.h>
#include "../Code/BoundaryCondition/HSIEPolynomial.h"
#include "../Code/Core/Types.h"

namespace {

  const std::vector<ComplexNumber> evaluation_points = {ComplexNumber(0, 0), ComplexNumber(1, 0), ComplexNumber(-1, 0), ComplexNumber(0.3, -0.8), ComplexNumber(-1.7, 0.4), ComplexNumber(0, 2)};

  const std::vector<ComplexNumber> values_of_k0 = {ComplexNumber(1, 0), ComplexNumber(0, 1), ComplexNumber(4.0537, 0), ComplexNumber(2.5, -1.5), ComplexNumber(0.2, 0.9)};

  std::vector<ComplexNumber> coefficients(unsigned int n) {
    std::vector<ComplexNumber> ret;
    for (unsigned int i = 0; i < n; i++) {
      ret.emplace_back(std::cos(1.0 + 2.0 * i), 0.5 * std::sin(3.0 * i + 0.4));
    }
    return ret;
  }

  // x^n by repeated multiplication (std::pow is not defined for 0^0).
  ComplexNumber power(ComplexNumber x, unsigned int n) {
    ComplexNumber ret(1, 0);
    for (unsigned int i = 0; i < n; i++) {
      ret *= x;
    }
    return ret;
  }

  ComplexNumber power_sum(const std::vector<ComplexNumber> & a, ComplexNumber x) {
    ComplexNumber ret(0, 0);
    for (unsigned int i = 0; i < a.size(); i++) {
      ret += a[i] * power(x, i);
    }
    return ret;
  }

  void expect_close(ComplexNumber value, ComplexNumber expected, double tolerance = 1e-12) {
    EXPECT_LT(std::abs(value - expected), tolerance * (1.0 + std::abs(expected))) << "value " << value << ", expected " << expected;
  }

}

TEST(HSIEPolynomialPropertyTests, EvaluationIsSumOfMonomials) {
  for (unsigned int n = 1; n < 8; n++) {
    HSIEPolynomial polynomial(coefficients(n), ComplexNumber(1, 0));
    for (ComplexNumber x : evaluation_points) {
      expect_close(polynomial.evaluate(x), power_sum(polynomial.a, x));
    }
  }
}

TEST(HSIEPolynomialPropertyTests, MonomialConstructor) {
  for (unsigned int degree = 0; degree < 6; degree++) {
    HSIEPolynomial monomial(degree, ComplexNumber(2, 1));
    ASSERT_EQ(monomial.a.size(), degree + 1);
    EXPECT_EQ(monomial.k0, ComplexNumber(2, 1));
    for (ComplexNumber x : evaluation_points) {
      expect_close(monomial.evaluate(x), power(x, degree));
    }
  }
}

TEST(HSIEPolynomialPropertyTests, DerivativeMatchesDifferenceQuotient) {
  const double h = 1e-5;
  for (unsigned int n = 1; n < 7; n++) {
    HSIEPolynomial polynomial(coefficients(n), ComplexNumber(1, 0));
    ASSERT_EQ(polynomial.da.size(), n - 1);
    for (ComplexNumber x : evaluation_points) {
      const ComplexNumber difference_quotient = (polynomial.evaluate(x + h) - polynomial.evaluate(x - h)) / (2 * h);
      if (n == 1) continue; // A constant has no derivative coefficients; evaluate_dx() must not be called.
      expect_close(polynomial.evaluate_dx(x), difference_quotient, 1e-7);
    }
  }
}

TEST(HSIEPolynomialPropertyTests, ApplyDerivativeReplacesPolynomialByItsDerivative) {
  HSIEPolynomial polynomial(coefficients(6), ComplexNumber(1, 0));
  HSIEPolynomial derivative = polynomial;
  derivative.applyDerivative();
  ASSERT_EQ(derivative.a.size(), 5u);
  for (unsigned int i = 0; i < 5; i++) {
    expect_close(derivative.a[i], (i + 1.0) * polynomial.a[i + 1]);
  }
  for (ComplexNumber x : evaluation_points) {
    expect_close(derivative.evaluate(x), polynomial.evaluate_dx(x));
  }
  // The second derivative is available as the derivative of the result.
  ASSERT_EQ(derivative.da.size(), 4u);
  expect_close(derivative.da[3], 20.0 * polynomial.a[5]);
}

TEST(HSIEPolynomialPropertyTests, AdditionAndScalingAreLinear) {
  for (unsigned int n = 1; n < 6; n++) {
    for (unsigned int m = 1; m < 6; m++) {
      HSIEPolynomial p(coefficients(n), ComplexNumber(1, 0));
      HSIEPolynomial q(coefficients(m + 2), ComplexNumber(1, 0));
      q.a.resize(m);
      q.update_derivative();
      const ComplexNumber factor(0.4, -1.3);
      HSIEPolynomial combination = p;
      combination.multiplyBy(factor);
      combination.add(q);
      combination.multiplyBy(2.0);
      ASSERT_EQ(combination.a.size(), std::max(n, m));
      for (ComplexNumber x : evaluation_points) {
        expect_close(combination.evaluate(x), 2.0 * (factor * p.evaluate(x) + q.evaluate(x)));
        if (std::max(n, m) > 1) {
          const ComplexNumber dp = (n > 1) ? p.evaluate_dx(x) : ComplexNumber(0, 0);
          const ComplexNumber dq = (m > 1) ? q.evaluate_dx(x) : ComplexNumber(0, 0);
          expect_close(combination.evaluate_dx(x), 2.0 * (factor * dp + dq));
        }
      }
    }
  }
}

// T_+(u_0, U)(x) = (u_0 + (x + 1) U(x)) / 2 and T_-(u_0, U)(x) = (u_0 + (x - 1) U(x)) / 2.
TEST(HSIEPolynomialPropertyTests, OperatorsTplusAndTminus) {
  const ComplexNumber u_0(0.7, -0.3);
  for (unsigned int n = 1; n < 6; n++) {
    const HSIEPolynomial U(coefficients(n), ComplexNumber(1, 0));
    HSIEPolynomial t_plus = U, t_minus = U;
    t_plus.applyTplus(u_0);
    t_minus.applyTminus(u_0);
    ASSERT_EQ(t_plus.a.size(), n + 1);
    ASSERT_EQ(t_minus.a.size(), n + 1);
    HSIEPolynomial copy = U;
    for (ComplexNumber x : evaluation_points) {
      expect_close(t_plus.evaluate(x), 0.5 * (u_0 + (x + 1.0) * copy.evaluate(x)));
      expect_close(t_minus.evaluate(x), 0.5 * (u_0 + (x - 1.0) * copy.evaluate(x)));
      // T_+ - T_- = U and T_+ + T_- = u_0 + x U.
      expect_close(t_plus.evaluate(x) - t_minus.evaluate(x), copy.evaluate(x));
      expect_close(t_plus.evaluate(x) + t_minus.evaluate(x), u_0 + x * copy.evaluate(x));
    }
  }
}

// Product rule: d/dx T_+-(u_0, U)(x) = (U(x) + (x +- 1) U'(x)) / 2.
TEST(HSIEPolynomialPropertyTests, DerivativeOfTplusAndTminusFollowsProductRule) {
  const ComplexNumber u_0(0.7, -0.3);
  for (unsigned int n = 2; n < 6; n++) {
    HSIEPolynomial U(coefficients(n), ComplexNumber(1, 0));
    HSIEPolynomial t_plus = U, t_minus = U;
    t_plus.applyTplus(u_0);
    t_minus.applyTminus(u_0);
    for (ComplexNumber x : evaluation_points) {
      expect_close(t_plus.evaluate_dx(x), 0.5 * (U.evaluate(x) + (x + 1.0) * U.evaluate_dx(x)));
      expect_close(t_minus.evaluate_dx(x), 0.5 * (U.evaluate(x) + (x - 1.0) * U.evaluate_dx(x)));
    }
  }
}

// Phi_j = T_+(0, x^j) and Psi_j = T_-(0, x^j) / (i k_0); for j = -1 the boundary value 1 and the zero polynomial.
TEST(HSIEPolynomialPropertyTests, BasisPolynomialsPhiAndPsi) {
  const ComplexNumber i(0, 1);
  for (ComplexNumber k0 : values_of_k0) {
    for (ComplexNumber x : evaluation_points) {
      expect_close(HSIEPolynomial::PhiMinusOne(k0).evaluate(x), 0.5);
      expect_close(HSIEPolynomial::PhiJ(-1, k0).evaluate(x), 0.5);
      expect_close(HSIEPolynomial::PsiMinusOne(k0).evaluate(x), 0.5 / (i * k0));
      expect_close(HSIEPolynomial::PsiJ(-1, k0).evaluate(x), 0.5 / (i * k0));
      for (int j = 0; j < 6; j++) {
        expect_close(HSIEPolynomial::PhiJ(j, k0).evaluate(x), 0.5 * (x + 1.0) * power(x, j));
        expect_close(HSIEPolynomial::PsiJ(j, k0).evaluate(x), 0.5 * (x - 1.0) * power(x, j) / (i * k0));
      }
    }
    EXPECT_EQ(HSIEPolynomial::PhiJ(3, k0).k0, k0);
    EXPECT_EQ(HSIEPolynomial::PsiJ(3, k0).a.size(), 5u);
  }
}

TEST(HSIEPolynomialPropertyTests, ZeroPolynomialVanishes) {
  HSIEPolynomial zero = HSIEPolynomial::ZeroPolynomial();
  for (ComplexNumber x : evaluation_points) {
    EXPECT_EQ(zero.evaluate(x), ComplexNumber(0, 0));
    EXPECT_EQ(zero.evaluate_dx(x), ComplexNumber(0, 0));
  }
  HSIEPolynomial sum(coefficients(3), ComplexNumber(1, 0));
  const ComplexNumber before = sum.evaluate(ComplexNumber(0.3, -0.8));
  sum.add(zero);
  expect_close(sum.evaluate(ComplexNumber(0.3, -0.8)), before);
}

// D = id + T / (2 i k_0) with the tridiagonal matrix T of the Laguerre recurrence: T(j, j-1) = j, T(j, j) = -(2j+1),
// T(j, j+1) = j + 1.
TEST(HSIEPolynomialPropertyTests, MatrixDIsIdentityPlusScaledRecurrenceMatrix) {
  for (ComplexNumber k0 : values_of_k0) {
    for (unsigned int dimension : {1u, 2u, 5u, 9u}) {
      HSIEPolynomial::computeDandI(dimension, k0);
      ASSERT_TRUE(HSIEPolynomial::matricesLoaded);
      ASSERT_EQ(HSIEPolynomial::D.m(), dimension);
      ASSERT_EQ(HSIEPolynomial::D.n(), dimension);
      const ComplexNumber scale = 1.0 / (ComplexNumber(0, 2) * k0);
      for (unsigned int row = 0; row < dimension; row++) {
        for (unsigned int column = 0; column < dimension; column++) {
          ComplexNumber expected(0, 0);
          if (column == row) expected = 1.0 - (2.0 * row + 1.0) * scale;
          if (column + 1 == row) expected = static_cast<double>(row) * scale;
          if (column == row + 1) expected = (row + 1.0) * scale;
          expect_close(HSIEPolynomial::D(row, column), expected);
        }
      }
    }
  }
}

TEST(HSIEPolynomialPropertyTests, MatricesDAndIAreInverse) {
  for (ComplexNumber k0 : values_of_k0) {
    for (unsigned int dimension : {1u, 2u, 3u, 5u, 8u, 12u}) {
      HSIEPolynomial::computeDandI(dimension, k0);
      ASSERT_EQ(HSIEPolynomial::I.m(), dimension);
      dealii::FullMatrix<ComplexNumber> product(dimension, dimension), reverse_product(dimension, dimension);
      HSIEPolynomial::D.mmult(product, HSIEPolynomial::I);
      HSIEPolynomial::I.mmult(reverse_product, HSIEPolynomial::D);
      for (unsigned int row = 0; row < dimension; row++) {
        for (unsigned int column = 0; column < dimension; column++) {
          const ComplexNumber expected = (row == column) ? 1.0 : 0.0;
          EXPECT_LT(std::abs(product(row, column) - expected), 1e-9) << "k0 = " << k0 << ", dimension " << dimension;
          EXPECT_LT(std::abs(reverse_product(row, column) - expected), 1e-9) << "k0 = " << k0 << ", dimension " << dimension;
        }
      }
    }
  }
}

// For a polynomial with as many coefficients as the matrices have rows, the operators D and I undo each other.
TEST(HSIEPolynomialPropertyTests, OperatorsDAndIAreInverse) {
  for (ComplexNumber k0 : values_of_k0) {
    for (unsigned int n : {1u, 2u, 4u, 7u}) {
      HSIEPolynomial::computeDandI(n, k0);
      HSIEPolynomial polynomial(coefficients(n), k0);
      HSIEPolynomial there_and_back = polynomial.applyD().applyI();
      HSIEPolynomial back_and_there = polynomial.applyI().applyD();
      ASSERT_EQ(there_and_back.a.size(), n);
      for (unsigned int i = 0; i < n; i++) {
        expect_close(there_and_back.a[i], polynomial.a[i], 1e-9);
        expect_close(back_and_there.a[i], polynomial.a[i], 1e-9);
      }
    }
  }
}

// The matrices are static and shared by all polynomials. applyD() uses the row index for the coefficient of the
// result, i.e. the result is D a.
TEST(HSIEPolynomialPropertyTests, ApplyDMultipliesCoefficientsWithD) {
  const ComplexNumber k0(2.5, -1.5);
  const unsigned int n = 4;
  HSIEPolynomial::computeDandI(n, k0);
  HSIEPolynomial polynomial(coefficients(n), k0);
  const HSIEPolynomial result = polynomial.applyD();
  for (unsigned int row = 0; row < n; row++) {
    ComplexNumber expected(0, 0);
    for (unsigned int column = 0; column < n; column++) {
      expected += HSIEPolynomial::D(row, column) * polynomial.a[column];
    }
    expect_close(result.a[row], expected);
  }
  EXPECT_EQ(result.k0, k0);
}

// A polynomial that is too long for the loaded matrices triggers their computation for its own k_0.
TEST(HSIEPolynomialPropertyTests, ApplyDComputesMatricesIfTheyAreTooSmall) {
  HSIEPolynomial::computeDandI(2, ComplexNumber(1, 0));
  const ComplexNumber k0(0.2, 0.9);
  HSIEPolynomial polynomial(coefficients(5), k0);
  polynomial.applyD();
  EXPECT_GE(HSIEPolynomial::D.m(), 5u);
  expect_close(HSIEPolynomial::D(0, 1), 1.0 / (ComplexNumber(0, 2) * k0));
}
