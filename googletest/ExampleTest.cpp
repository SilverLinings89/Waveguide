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
    dealii::Triangulation<3> tria ;
    dealii::Triangulation<2,3> temp_triangulation;
    dealii::Triangulation<2> surf_tria;
    std::cout << "Building initial 3D mesh" << std::endl;
    std::vector<unsigned int> repetitions;
    repetitions.push_back(9);
    repetitions.push_back(9);
    repetitions.push_back(9);
    dealii::Point<3,double> left(-1, -1, -1);
    dealii::Point<3,double> right(1, 1, 1);
    dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, left, right, true);
    const unsigned int dest_cells = 9*9*9;
    ASSERT_EQ(tria.n_active_cells(), dest_cells);
    std::complex<double> k0(0.0, -1.0);
    std::set<unsigned int> b_ids;
    b_ids.insert(5);
    std::cout << "Extract boundardy" << std::endl;
    std::map<dealii::Triangulation<2,3>::cell_iterator, dealii::Triangulation<3,3>::face_iterator > association = dealii::GridGenerator::extract_boundary_mesh( tria, temp_triangulation, b_ids);
    const unsigned int dest_surf_cells = 9*9;
    ASSERT_EQ(temp_triangulation.n_active_cells(), dest_surf_cells);
    std::cout << "Flatten boundardy" << std::endl;
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
    std::cout << tria.n_active_cells() << " - " << temp_triangulation.n_active_cells() << " - " << surf_tria.n_active_cells() << std::endl;
    ASSERT_EQ(surf_tria.n_active_cells(), dest_surf_cells);
    std::cout << "Call Constructor" << std::endl;
    HSIESurface<5> surf(surf_tria, 0, 0, 2, k0, association);

    std::cout << "Run initialize:" << std::endl;
    surf.initialize();
    ASSERT_EQ(surf_tria.n_active_lines(), 180);
    ASSERT_EQ(surf_tria.n_vertices(), 100);
    DofCount cnt = surf.compute_n_vertex_dofs();
    print_dof_count(cnt);

    ASSERT_EQ(surf.compute_n_face_dofs().hsie, 0);
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