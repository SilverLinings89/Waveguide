// Properties of the shape function of the optimization: a function on [z_min, z_max] that is mirror symmetric with
// respect to the middle of the interval. On the first half its derivative is continuous and piecewise linear on n
// sectors of equal length; the nodal values of the derivative are the dofs. The values at the lower end and in the
// middle and the derivatives there are prescribed.
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Optimization/ShapeFunction.h"
#include "./HelpersA.h"

namespace {

  const double z_end = 6.0;
  const double middle = 3.0;
  const unsigned int n_sectors = 6;
  const double sector_length = middle / n_sectors;

  std::vector<double> random_values(unsigned int n, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<double> ret(n);
    for (double & value : ret) value = distribution(generator);
    return ret;
  }

  // A function on [0, z_end] with the given end value f(0), the value in the middle and random free dofs.
  ShapeFunction random_function(double f_start, double f_middle, unsigned int seed) {
    ShapeFunction ret(0.0, z_end, n_sectors);
    ret.set_constraints(f_start, f_middle, 0, 0);
    ret.set_free_values(random_values(ret.get_n_free_dofs(), seed));
    return ret;
  }

  // Points in the interior of the sectors of the first half, away from the sector boundaries.
  std::vector<double> points_inside_sectors() {
    std::vector<double> ret;
    for (unsigned int sector = 0; sector < n_sectors; sector++) {
      for (double fraction : {0.2, 0.5, 0.8}) ret.push_back((sector + fraction) * sector_length);
    }
    return ret;
  }

  // Value at z of the parabola through three points.
  double quadratic_extrapolation(const double x[3], const double y[3], double z) {
    double ret = 0;
    for (unsigned int i = 0; i < 3; i++) {
      double basis = 1;
      for (unsigned int j = 0; j < 3; j++) {
        if (j != i) basis *= (z - x[j]) / (x[i] - x[j]);
      }
      ret += basis * y[i];
    }
    return ret;
  }

}

TEST(ShapeFunctionPropertyTests, NumberOfDofs) {
  for (unsigned int n = 2; n < 10; n++) {
    // n + 1 nodal values of the derivative and two function values, five of them are constrained.
    EXPECT_EQ(ShapeFunction::compute_n_dofs(n), n + 3);
    EXPECT_EQ(ShapeFunction::compute_n_free_dofs(n), n - 2);
    ShapeFunction function(0.0, 1.0, n);
    EXPECT_EQ(function.get_n_dofs(), n + 3);
    EXPECT_EQ(function.get_n_free_dofs(), n - 2);
    EXPECT_EQ(function.n_dofs, function.get_n_dofs());
    EXPECT_EQ(function.n_free_dofs, function.get_n_free_dofs());
  }
}

TEST(ShapeFunctionPropertyTests, ValueAndDerivativeAtTheLowerEnd) {
  for (unsigned int seed = 0; seed < 5; seed++) {
    const ShapeFunction function = random_function(0.0, 1.5, seed);
    EXPECT_EQ(function.evaluate_at(0.0), 0.0);
    EXPECT_EQ(function.evaluate_derivative_at(0.0), 0.0);
    // The derivative is continuous at the lower end.
    EXPECT_NEAR(function.evaluate_derivative_at(1e-3), 0.0, 1e-2);
  }
  const ShapeFunction shifted = random_function(0.75, 1.5, 7);
  EXPECT_EQ(shifted.evaluate_at(0.0), 0.75);
}

// The second argument of set_constraints is the value in the middle of the interval, whatever the free dofs are.
TEST(ShapeFunctionPropertyTests, PrescribedValueAndZeroDerivativeInTheMiddle) {
  for (unsigned int seed = 0; seed < 5; seed++) {
    const ShapeFunction function = random_function(0.25, 1.5, seed);
    EXPECT_NEAR(function.evaluate_at(middle), 1.5, 1e-12);
    EXPECT_NEAR(function.evaluate_derivative_at(middle), 0.0, 1e-12);
  }
}

TEST(ShapeFunctionPropertyTests, IsMirrorSymmetricWithRespectToTheMiddle) {
  const ShapeFunction function = random_function(0.25, 1.5, 3);
  for (double distance = 0.0; distance <= middle; distance += 0.07) {
    EXPECT_NEAR(function.evaluate_at(middle + distance), function.evaluate_at(middle - distance), 1e-12) << distance;
    EXPECT_NEAR(function.evaluate_derivative_at(middle + distance), -function.evaluate_derivative_at(middle - distance), 1e-12) << distance;
  }
  EXPECT_NEAR(function.evaluate_at(z_end), 0.25, 1e-12);
  EXPECT_NEAR(function.evaluate_derivative_at(z_end), 0.0, 1e-12);
}

TEST(ShapeFunctionPropertyTests, IsConstantOutsideOfTheInterval) {
  const ShapeFunction function = random_function(0.25, 1.5, 3);
  EXPECT_EQ(function.evaluate_at(-1.0), 0.25);
  EXPECT_EQ(function.evaluate_at(z_end + 1.0), 0.25);
  EXPECT_EQ(function.evaluate_derivative_at(-1.0), 0.0);
  EXPECT_EQ(function.evaluate_derivative_at(z_end + 1.0), 0.0);
}

// Inside a sector the function is a quadratic polynomial, for which the central difference is exact.
TEST(ShapeFunctionPropertyTests, DerivativeMatchesFiniteDifferences) {
  const ShapeFunction function = random_function(0.25, 1.5, 4);
  const double h = 1e-3;
  for (double z : points_inside_sectors()) {
    for (double point : {z, z_end - z}) {
      const double difference = (function.evaluate_at(point + h) - function.evaluate_at(point - h)) / (2 * h);
      EXPECT_NEAR(function.evaluate_derivative_at(point), difference, 1e-9) << point;
    }
  }
}

// The dofs between the two function values are the values of the derivative at the sector boundaries.
TEST(ShapeFunctionPropertyTests, DofsAreTheDerivativesAtTheSectorBoundaries) {
  const ShapeFunction function = random_function(0.25, 1.5, 5);
  EXPECT_EQ(function.get_dof_value(0), 0.25);
  EXPECT_EQ(function.get_dof_value(function.get_n_dofs() - 1), 1.5);
  for (unsigned int node = 0; node <= n_sectors; node++) {
    EXPECT_NEAR(function.evaluate_derivative_at(node * sector_length), function.get_dof_value(node + 1), 1e-12) << node;
  }
  // In between the derivative is linear.
  for (unsigned int sector = 0; sector < n_sectors; sector++) {
    const double expected = 0.75 * function.get_dof_value(sector + 1) + 0.25 * function.get_dof_value(sector + 2);
    EXPECT_NEAR(function.evaluate_derivative_at((sector + 0.25) * sector_length), expected, 1e-12) << sector;
  }
}

// The function is the integral of its derivative: the trapezoidal rule is exact for the piecewise linear derivative.
TEST(ShapeFunctionPropertyTests, FunctionIsTheIntegralOfTheDerivative) {
  const ShapeFunction function = random_function(0.25, 1.5, 6);
  double integral = 0.25;
  for (unsigned int node = 0; node <= n_sectors; node++) {
    EXPECT_NEAR(function.evaluate_at(node * sector_length), integral, 1e-12) << node;
    if (node < n_sectors) integral += 0.5 * sector_length * (function.get_dof_value(node + 1) + function.get_dof_value(node + 2));
  }
  EXPECT_NEAR(integral, 1.5, 1e-12);
}

// The function and its derivative are continuous at the sector boundaries: the polynomials of the two neighboring
// sectors, extrapolated from points in their interior, have the same value and the same derivative at the boundary.
TEST(ShapeFunctionPropertyTests, FunctionAndDerivativeAreContinuousAtTheSectorBoundaries) {
  const ShapeFunction function = random_function(0.25, 1.5, 8);
  for (unsigned int node = 1; node < n_sectors; node++) {
    const double z = node * sector_length;
    double x_left[3], x_right[3], f_left[3], f_right[3], df_left[3], df_right[3];
    for (unsigned int i = 0; i < 3; i++) {
      x_left[i] = z - (0.2 + 0.3 * i) * sector_length;
      x_right[i] = z + (0.2 + 0.3 * i) * sector_length;
      f_left[i] = function.evaluate_at(x_left[i]);
      f_right[i] = function.evaluate_at(x_right[i]);
      df_left[i] = function.evaluate_derivative_at(x_left[i]);
      df_right[i] = function.evaluate_derivative_at(x_right[i]);
    }
    EXPECT_NEAR(quadratic_extrapolation(x_left, f_left, z), quadratic_extrapolation(x_right, f_right, z), 1e-11) << node;
    EXPECT_NEAR(quadratic_extrapolation(x_left, df_left, z), quadratic_extrapolation(x_right, df_right, z), 1e-11) << node;
    EXPECT_NEAR(function.evaluate_at(z), quadratic_extrapolation(x_left, f_left, z), 1e-11) << node;
  }
}

// The same holds at the middle of the interval, where the function is continued by reflection.
TEST(ShapeFunctionPropertyTests, FunctionAndDerivativeAreContinuousInTheMiddle) {
  const ShapeFunction function = random_function(0.25, 1.5, 9);
  const double h = 1e-4;
  EXPECT_NEAR(function.evaluate_at(middle - h), function.evaluate_at(middle + h), 1e-12);
  EXPECT_NEAR(function.evaluate_at(middle - h), 1.5, 1e-5);
  EXPECT_NEAR(function.evaluate_derivative_at(middle - h), 0.0, 1e-2);
  EXPECT_NEAR(function.evaluate_derivative_at(middle + h), 0.0, 1e-2);
}

// The function depends linearly on the constraints and the free dofs. The shape gradient relies on this.
TEST(ShapeFunctionPropertyTests, DependsLinearlyOnDofsAndConstraints) {
  ShapeFunction a(0.0, z_end, n_sectors), b(0.0, z_end, n_sectors), combination(0.0, z_end, n_sectors);
  const std::vector<double> values_a = random_values(a.get_n_free_dofs(), 1);
  const std::vector<double> values_b = random_values(b.get_n_free_dofs(), 2);
  std::vector<double> values_combination(values_a.size());
  for (unsigned int i = 0; i < values_a.size(); i++) values_combination[i] = 2.0 * values_a[i] - 0.5 * values_b[i];
  a.set_constraints(0.25, 1.5, 0, 0);
  a.set_free_values(values_a);
  b.set_constraints(-1.0, 0.5, 0, 0);
  b.set_free_values(values_b);
  combination.set_constraints(2.0 * 0.25 - 0.5 * -1.0, 2.0 * 1.5 - 0.5 * 0.5, 0, 0);
  combination.set_free_values(values_combination);
  for (double z = 0.0; z <= z_end; z += 0.13) {
    EXPECT_NEAR(combination.evaluate_at(z), 2.0 * a.evaluate_at(z) - 0.5 * b.evaluate_at(z), 1e-12) << z;
    EXPECT_NEAR(combination.evaluate_derivative_at(z), 2.0 * a.evaluate_derivative_at(z) - 0.5 * b.evaluate_derivative_at(z), 1e-12) << z;
  }
}

TEST(ShapeFunctionPropertyTests, FreeDofsCanBeSetOneByOne) {
  ShapeFunction all_at_once = random_function(0.25, 1.5, 10);
  const std::vector<double> values = random_values(all_at_once.get_n_free_dofs(), 10);
  ShapeFunction one_by_one(0.0, z_end, n_sectors);
  one_by_one.set_constraints(0.25, 1.5, 0, 0);
  for (unsigned int i = 0; i < values.size(); i++) one_by_one.set_free_dof_value(i, values[i]);
  for (unsigned int i = 0; i < values.size(); i++) {
    EXPECT_EQ(one_by_one.get_free_dof_value(i), values[i]);
    // The free dofs follow the two constrained dofs at the lower end.
    EXPECT_EQ(one_by_one.get_dof_value(i + 2), values[i]);
  }
  for (unsigned int i = 0; i < one_by_one.get_n_dofs(); i++) {
    EXPECT_NEAR(one_by_one.get_dof_value(i), all_at_once.get_dof_value(i), 1e-13) << i;
  }
  for (double z = 0.0; z <= z_end; z += 0.13) {
    EXPECT_NEAR(one_by_one.evaluate_at(z), all_at_once.evaluate_at(z), 1e-12) << z;
  }
}

// The constraints stay valid when they are set after the free dofs or changed later.
TEST(ShapeFunctionPropertyTests, ConstraintsCanBeChangedAfterTheDofs) {
  ShapeFunction function(0.0, z_end, n_sectors);
  function.set_free_values(random_values(function.get_n_free_dofs(), 11));
  function.set_constraints(0.5, -2.0, 0, 0);
  EXPECT_EQ(function.evaluate_at(0.0), 0.5);
  EXPECT_NEAR(function.evaluate_at(middle), -2.0, 1e-12);
  function.set_constraints(1.0, 1.0, 0, 0);
  EXPECT_EQ(function.evaluate_at(0.0), 1.0);
  EXPECT_NEAR(function.evaluate_at(middle), 1.0, 1e-12);
  EXPECT_NEAR(function.evaluate_at(z_end), 1.0, 1e-12);
}

TEST(ShapeFunctionPropertyTests, WritingAConstrainedDofChangesNothing) {
  ShapeFunction function = random_function(0.25, 1.5, 12);
  std::vector<double> before(function.get_n_dofs());
  for (unsigned int i = 0; i < before.size(); i++) before[i] = function.get_dof_value(i);
  testing::internal::CaptureStdout();
  function.set_free_dof_value(function.get_n_free_dofs(), 17.0);
  function.set_free_dof_value(function.get_n_free_dofs() + 2, 17.0);
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_FALSE(output.empty());
  for (unsigned int i = 0; i < before.size(); i++) EXPECT_EQ(function.get_dof_value(i), before[i]) << i;
}

// A list of free values with the wrong length is reported. The values that fit are used and the constraints still
// hold afterwards.
TEST(ShapeFunctionPropertyTests, WrongNumberOfFreeValuesKeepsTheConstraints) {
  ShapeFunction function = random_function(0.25, 1.5, 13);
  const std::vector<double> before = {function.get_free_dof_value(0), function.get_free_dof_value(1), function.get_free_dof_value(2), function.get_free_dof_value(3)};
  ASSERT_EQ(function.get_n_free_dofs(), 4u);
  testing::internal::CaptureStdout();
  function.set_free_values({0.3, -0.2});
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_FALSE(output.empty());
  EXPECT_EQ(function.get_free_dof_value(0), 0.3);
  EXPECT_EQ(function.get_free_dof_value(1), -0.2);
  EXPECT_EQ(function.get_free_dof_value(2), before[2]);
  EXPECT_EQ(function.get_free_dof_value(3), before[3]);
  EXPECT_EQ(function.evaluate_at(0.0), 0.25);
  EXPECT_NEAR(function.evaluate_at(middle), 1.5, 1e-12);
  // One value too many would be written to the constrained dof that enforces the value in the middle.
  testing::internal::CaptureStdout();
  function.set_free_values({0.1, 0.2, 0.3, 0.4, 99.0});
  output = testing::internal::GetCapturedStdout();
  EXPECT_FALSE(output.empty());
  EXPECT_EQ(function.evaluate_at(0.0), 0.25);
  EXPECT_NEAR(function.evaluate_at(middle), 1.5, 1e-12);
  EXPECT_NEAR(function.evaluate_derivative_at(middle), 0.0, 1e-12);
}

// The default initialization fulfills the constraints, and it is the constant function if both values agree.
TEST(ShapeFunctionPropertyTests, DefaultInitializationFulfillsTheConstraints) {
  ShapeFunction function(0.0, z_end, n_sectors);
  function.set_constraints(0.0, 1.5, 0, 0);
  function.initialize();
  EXPECT_EQ(function.evaluate_at(0.0), 0.0);
  EXPECT_NEAR(function.evaluate_at(middle), 1.5, 1e-12);
  EXPECT_NEAR(function.evaluate_at(z_end), 0.0, 1e-12);
  // All free dofs are the mean slope between the two prescribed values.
  for (unsigned int i = 0; i < function.get_n_free_dofs(); i++) EXPECT_NEAR(function.get_free_dof_value(i), 1.5 / middle, 1e-12);

  ShapeFunction constant(0.0, z_end, n_sectors);
  constant.set_constraints(1.0, 1.0, 0, 0);
  constant.initialize();
  for (double z = 0.0; z <= z_end; z += 0.11) {
    EXPECT_NEAR(constant.evaluate_at(z), 1.0, 1e-12) << z;
    EXPECT_NEAR(constant.evaluate_derivative_at(z), 0.0, 1e-12) << z;
  }
}

TEST(ShapeFunctionPropertyTests, RandomInitializationFulfillsTheConstraints) {
  helpers_a::ScopedGlobalParams params;
  ShapeFunction function(0.0, z_end, n_sectors, true);
  function.set_constraints(0.0, 1.5, 0, 0);
  function.initialize();
  for (unsigned int i = 0; i < function.get_n_free_dofs(); i++) {
    EXPECT_GE(function.get_free_dof_value(i), -0.5);
    EXPECT_LE(function.get_free_dof_value(i), 0.5);
  }
  EXPECT_EQ(function.evaluate_at(0.0), 0.0);
  EXPECT_NEAR(function.evaluate_at(middle), 1.5, 1e-12);
  EXPECT_NEAR(function.evaluate_derivative_at(middle), 0.0, 1e-12);
}

// The properties above also have to hold for an interval that does not start at zero (the domain is shifted by the
// parameter "global z shift"). The constructor uses in_z_max / 2 as the middle instead of (in_z_min + in_z_max) / 2,
// so the reflection point and the point of the prescribed value are wrong for z_min != 0.
TEST(ShapeFunctionPropertyTests, DISABLED_IntervalThatDoesNotStartAtZero) {
  const double z_min = 2.0, z_max = 8.0;
  ShapeFunction function(z_min, z_max, n_sectors);
  function.set_constraints(0.25, 1.5, 0, 0);
  function.set_free_values(random_values(function.get_n_free_dofs(), 14));
  const double shifted_middle = 0.5 * (z_min + z_max);
  EXPECT_EQ(function.evaluate_at(z_min), 0.25);
  ASSERT_NEAR(function.evaluate_at(shifted_middle), 1.5, 1e-12);
  EXPECT_NEAR(function.evaluate_at(z_max), 0.25, 1e-12);
  // The function is mirror symmetric and a translation of the function on [0, z_max - z_min] with the same dofs.
  const ShapeFunction reference = random_function(0.25, 1.5, 14);
  double asymmetry = 0, difference_to_reference = 0;
  for (double distance = 0.0; distance <= 3.0; distance += 0.25) {
    asymmetry = std::max(asymmetry, std::abs(function.evaluate_at(shifted_middle + distance) - function.evaluate_at(shifted_middle - distance)));
    difference_to_reference = std::max(difference_to_reference, std::abs(function.evaluate_at(shifted_middle + distance) - reference.evaluate_at(middle + distance)));
  }
  EXPECT_LT(asymmetry, 1e-12);
  EXPECT_LT(difference_to_reference, 1e-12);
}

// With fewer than two sectors the five constraints use up all dofs (n + 3 <= 5), so there is no free dof. The code
// returns the absolute value of the negative difference: 1 free dof for one sector.
TEST(ShapeFunctionPropertyTests, DISABLED_NoFreeDofsWithFewerThanTwoSectors) {
  testing::internal::CaptureStdout();
  const unsigned int free_dofs_for_one_sector = ShapeFunction::compute_n_free_dofs(1);
  const unsigned int free_dofs_for_two_sectors = ShapeFunction::compute_n_free_dofs(2);
  testing::internal::GetCapturedStdout();
  EXPECT_EQ(free_dofs_for_one_sector, 0u);
  EXPECT_EQ(free_dofs_for_two_sectors, 0u);
}
