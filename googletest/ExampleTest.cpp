//
// Created by kraft on 25.07.19.
//

#include "../Code/Core/DOFManager.h"
#include "../Code/HSIEPreconditioner/LaguerreFunction.h"

#include "gtest/gtest.h"
#include "../Code/HSIEPreconditioner/HSIESurface.h"
#include "../Code/HSIEPreconditioner/HSIEPolynomial.h"
#include "../Code/HSIEPreconditioner/HSIESurface.cpp"
#include <deal.II/grid/grid_generator.h>

TEST(HSIESurfaceTests, AssemblationTest) {
    dealii::Triangulation<3> tria;
    dealii::GridGenerator::subdivided_hyper_cube(tria, 9, -1.0, 1.0);
    std::complex<double> k0(0.0, -1.0);
    HSIESurface<5> surf = {&tria, 0, 0, 1, k0};
    surf.initialize();
}

TEST(HSIEPolynomialTests, TestOperatorTplus) {
    std::vector<std::complex<double>> in_a;
    in_a.emplace_back(0.0, 0.0);
    in_a.emplace_back(0.0, 0.0);
    in_a.emplace_back(1.0, 0.0);
    in_a.emplace_back(0.0, 1.0);
    in_a.emplace_back(0.0, 0.0);
    std::complex<double> k0(0.0, 1.0);
    HSIEPolynomial poly(in_a, k0);
    poly.applyTplus(std::complex<double>(1,-1));
    ASSERT_EQ(poly.a[0], std::complex<double>(0.5, -0.5));
    ASSERT_EQ(poly.a[1], std::complex<double>(0, 0));
    ASSERT_EQ(poly.a[2], std::complex<double>(0.5, 0));
    ASSERT_EQ(poly.a[3], std::complex<double>(0.5, 0.5));
    ASSERT_EQ(poly.a[4], std::complex<double>(0, 0.5));
}

TEST(HSIEPolynomialTests, ProductOfDandIShouldBeIdentity) {
    std::vector<std::complex<double>> in_a;
    in_a.emplace_back(0.0, 0.0);
    in_a.emplace_back(0.0, 0.0);
    in_a.emplace_back(1.0, 0.0);
    in_a.emplace_back(0.0, 1.0);
    in_a.emplace_back(0.0, 0.0);
    std::complex<double> k0(0.0, 1.0);
    HSIEPolynomial poly(in_a, k0);
    poly.applyD();
    dealii::FullMatrix<std::complex<double>> product(HSIEPolynomial::D.size(0), HSIEPolynomial::D.size(1));
    HSIEPolynomial::D.mmult(product, HSIEPolynomial::I, false);
    ASSERT_EQ(product.frobenius_norm(), std::sqrt(HSIEPolynomial::I.size(0)));
}

TEST(DOFManagerTests, StaticFunctionTest) {
    ASSERT_EQ(DOFManager::testValue(),4);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_1) {
    ASSERT_EQ(LaguerreFunction::evaluate(0,13,2), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_2) {
    ASSERT_EQ(LaguerreFunction::evaluate(0,1,4), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_3) {
    for(int i = 0; i < 10; i++) {
        ASSERT_EQ(LaguerreFunction::evaluate(1,i,2), -2 + i + 1);
    }
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_4) {
    for(int i = 0; i < 10; i++) {
        ASSERT_EQ(LaguerreFunction::evaluate(2,i,2), 2 - 2*i - 4 + (i+2)*(i+1)/2);
    }
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_5) {
    for(int i = 0; i < 10; i++) {
        for(unsigned int j = 0; j < 10; j++) {
            ASSERT_EQ(LaguerreFunction::evaluate(i,j,0), LaguerreFunction::binomial_coefficient(i+j,i));
        }
    }
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_1) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(5,2), 10);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_2) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(0,0), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_3) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(1,0), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_4) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(1,1), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_5) {
    ASSERT_EQ(LaguerreFunction::binomial_coefficient(15,5), 3003);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_1) {
    ASSERT_EQ(LaguerreFunction::factorial(1), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_2) {
    ASSERT_EQ(LaguerreFunction::factorial(2), 2);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_3) {
    ASSERT_EQ(LaguerreFunction::factorial(4), 24);
}

TEST(LAGUERRE_FUNCTION_TESTS, FACTORIAL_5) {
    ASSERT_EQ(LaguerreFunction::factorial(5), 120);
}