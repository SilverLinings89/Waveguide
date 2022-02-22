#include "../Code/BoundaryCondition/LaguerreFunction.h"
#include "../third_party/googletest/googletest/include/gtest/gtest.h"
#include "../Code/BoundaryCondition/HSIESurface.h"
#include "../Code/BoundaryCondition/HSIEPolynomial.h"
#include "../Code/BoundaryCondition/BoundaryCondition.h"
#include "../Code/Core/Types.h"
#include "../Code/Helpers/staticfunctions.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/grid_tools.h>
#include "./Fixtures.h"
#include "./TestHelperFunctions.h"
#include "./TestData.h"



TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_1) {
  ASSERT_EQ(LaguerreFunction::evaluate(0, 13, 2), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_2) {
  ASSERT_EQ(LaguerreFunction::evaluate(0, 1, 4), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_3) {
  for (int i = 0; i < 10; i++) {
    ASSERT_EQ(LaguerreFunction::evaluate(1, i, 2), -2 + i + 1);
  }
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_4) {
  for (int i = 0; i < 10; i++) {
    ASSERT_EQ(LaguerreFunction::evaluate(2, i, 2),
        2 - 2 * i - 4 + (i + 2) * (i + 1) / 2);
  }
}

TEST(LAGUERRE_FUNCTION_TESTS, LAGUERRE_EVALUATION_5) {
  for (int i = 0; i < 10; i++) {
    for (unsigned int j = 0; j < 10; j++) {
      ASSERT_EQ(LaguerreFunction::evaluate(i, j, 0),
          LaguerreFunction::binomial_coefficient(i + j, i));
    }
  }
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_1) {
  ASSERT_EQ((int)LaguerreFunction::binomial_coefficient(5, 2), 10);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_2) {
  ASSERT_EQ((int)LaguerreFunction::binomial_coefficient(0, 0), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_3) {
  ASSERT_EQ((int)LaguerreFunction::binomial_coefficient(1, 0), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_4) {
  ASSERT_EQ((int)LaguerreFunction::binomial_coefficient(1, 1), 1);
}

TEST(LAGUERRE_FUNCTION_TESTS, BINOMIAL_COEFFICIENTS_5) {
  ASSERT_EQ((int)LaguerreFunction::binomial_coefficient(15, 5), 3003);
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

TEST(STATIC_FUNCTION_TEST, COMPARISON_TESTS_1) {
  Position p1 = {0,0,1};
  Position p2 = {0,0,0};
  Position p3 = {0,1,1};
  Position p4 = {0,1,0};
  Position p5 = {1,0,1};
  Position p6 = {1,0,0};
  Position p7 = {1,1,1};
  Position p8 = {1,1,0};
  std::pair<DofNumber, Position> pair_1 = {1,p1};
  std::pair<DofNumber, Position> pair_2 = {2,p2};
  std::pair<DofNumber, Position> pair_3 = {3,p3};
  std::pair<DofNumber, Position> pair_4 = {4,p4};
  std::pair<DofNumber, Position> pair_5 = {5,p5};
  std::pair<DofNumber, Position> pair_6 = {6,p6};
  std::pair<DofNumber, Position> pair_7 = {7,p7};
  std::pair<DofNumber, Position> pair_8 = {8,p8};
  std::vector<std::pair<DofNumber, Position>> dofs;
  dofs.push_back(pair_1);
  dofs.push_back(pair_3);
  dofs.push_back(pair_5);
  dofs.push_back(pair_6);
  dofs.push_back(pair_7);
  dofs.push_back(pair_2);
  dofs.push_back(pair_4);
  dofs.push_back(pair_8);
  std::sort(dofs.begin(), dofs.end(), compareDofBaseData);
  std::string ord = "";
  for(unsigned int i = 0; i < 8; i++) ord += std::to_string(dofs[i].first);
  ASSERT_EQ(ord, "26481537");
}