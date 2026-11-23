// Tests for the free functions in Code/Helpers/staticfunctions.{h,cpp}.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <map>
#include <random>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Helpers/staticfunctions.h"
#include "./HelpersA.h"

namespace {

  using helpers_a::ScopedGeometry;
  using helpers_a::ScopedGlobalParams;

  typedef Position (*CoordinateChange)(const Position &);
  const CoordinateChange to_5[5] = {Transform_0_to_5, Transform_1_to_5, Transform_2_to_5, Transform_3_to_5, Transform_4_to_5};
  const CoordinateChange from_5[5] = {Transform_5_to_0, Transform_5_to_1, Transform_5_to_2, Transform_5_to_3, Transform_5_to_4};

  InterfaceDofData interface_dof(DofNumber index, double x, double y, double z, unsigned int order = 0) {
    InterfaceDofData ret(index, Position(x, y, z));
    ret.order = order;
    return ret;
  }

  // Lexicographic key of a point that is exact for the coordinates used in these tests.
  std::array<long, 3> key(const Position & p) {
    return {std::lround(p[0] * 1e6), std::lround(p[1] * 1e6), std::lround(p[2] * 1e6)};
  }

}

TEST(RepositoryPathTests, DefaultRootContainsTheModeFile) {
  helpers_a::ScopedEnvironmentVariable root("WAVEGUIDE_ROOT", nullptr);
  EXPECT_TRUE(std::filesystem::exists(repository_path("Modes/output_mode.dat")));
}

TEST(RepositoryPathTests, EnvironmentVariableReplacesRoot) {
  {
    helpers_a::ScopedEnvironmentVariable root("WAVEGUIDE_ROOT", "/some/folder");
    EXPECT_EQ(repository_path("Modes/a.dat"), "/some/folder/Modes/a.dat");
  }
  {
    helpers_a::ScopedEnvironmentVariable root("WAVEGUIDE_ROOT", "/some/folder/");
    EXPECT_EQ(repository_path("Modes/a.dat"), "/some/folder/Modes/a.dat");
  }
}

TEST(ExecTests, ReturnsStandardOutputOfCommand) {
  EXPECT_EQ(exec("printf 'first\\nsecond'"), "first\nsecond");
  EXPECT_EQ(exec("true"), "");
}

// The function reads the pipe in blocks of 128 characters; longer outputs have to be returned completely.
TEST(ExecTests, ReturnsOutputLongerThanReadBuffer) {
  const std::string output = exec("printf '%01000d' 7");
  ASSERT_EQ(output.size(), 1000u);
  EXPECT_EQ(output.back(), '7');
  EXPECT_EQ(std::count(output.begin(), output.end(), '0'), 999);
}

TEST(MatrixDTests, IsTridiagonal) {
  const ComplexNumber k0(0.7, 1.3);
  for (int row = 0; row < 8; row++) {
    for (int column = 0; column < 8; column++) {
      if (std::abs(row - column) > 1) {
        EXPECT_EQ(matrixD(row, column, k0), ComplexNumber(0, 0)) << row << "," << column;
      } else {
        EXPECT_GT(std::abs(matrixD(row, column, k0)), 0.0) << row << "," << column;
      }
    }
  }
}

// D = Id + 1 / (2 i kappa_0) * L, where L is the symmetric tridiagonal matrix with the rows (n, -(2n+1), n+1). This
// is the representation of the operator 1 + ((z - 1)^2 d/dz + (z - 1)) / (2 i kappa_0) on the monomials z^n. Applied to
// the coefficients of the constant function 1 / (1 - z) = sum z^n, which is in the kernel of the second term, D is
// the identity, so every row sums to 1.
TEST(MatrixDTests, IsSymmetricAndRowsSumToOne) {
  const ComplexNumber k0(0.7, 1.3);
  const int n = 9;
  for (int row = 0; row < n; row++) {
    ComplexNumber row_sum(0, 0);
    for (int column = 0; column <= row + 1; column++) {
      row_sum += matrixD(row, column, k0);
      EXPECT_LT(std::abs(matrixD(row, column, k0) - matrixD(column, row, k0)), 1e-14);
    }
    EXPECT_LT(std::abs(row_sum - ComplexNumber(1, 0)), 1e-13) << row;
  }
}

TEST(MatrixDTests, ApproachesIdentityForLargeKappa) {
  const ComplexNumber k0(1e12, 0);
  for (int row = 0; row < 5; row++) {
    for (int column = 0; column < 5; column++) {
      const ComplexNumber expected = (row == column) ? 1.0 : 0.0;
      EXPECT_LT(std::abs(matrixD(row, column, k0) - expected), 1e-10);
    }
  }
}

TEST(InterfaceDofTests, MinimalAndMaximalIndex) {
  std::vector<InterfaceDofData> dofs;
  for (DofNumber index : {17u, 4u, 99u, 23u}) dofs.push_back(interface_dof(index, 0, 0, 0));
  const std::pair<DofNumber, DofNumber> range = get_max_and_min_dof_for_interface_data(dofs);
  EXPECT_EQ(range.first, 4u);
  EXPECT_EQ(range.second, 99u);
}

TEST(InterfaceDofTests, ShiftAddsOffsetToEveryIndex) {
  std::vector<InterfaceDofData> dofs;
  for (DofNumber index : {0u, 5u, 12u}) dofs.push_back(interface_dof(index, index, 0, 0));
  shift_interface_dof_data(&dofs, 100);
  EXPECT_EQ(dofs[0].index, 100u);
  EXPECT_EQ(dofs[1].index, 105u);
  EXPECT_EQ(dofs[2].index, 112u);
  EXPECT_EQ(dofs[1].base_point, Position(5, 0, 0));
}

TEST(InterfaceDofTests, DofsAreCloseOnlyAtTheSameBasePoint) {
  EXPECT_TRUE(areDofsClose(interface_dof(1, 0.3, 0.4, 0.5), interface_dof(2, 0.3, 0.4, 0.5)));
  EXPECT_TRUE(areDofsClose(interface_dof(1, 0.3, 0.4, 0.5), interface_dof(2, 0.3 + 1e-7, 0.4, 0.5)));
  EXPECT_FALSE(areDofsClose(interface_dof(1, 0.3, 0.4, 0.5), interface_dof(2, 0.3, 0.4, 0.6)));
  EXPECT_FALSE(areDofsClose(interface_dof(1, 0.3, 0.4, 0.5), interface_dof(2, 0.31, 0.4, 0.5)));
}

TEST(PositionOrderTests, OrdersByZThenYThenX) {
  EXPECT_TRUE(comparePositions(Position(9, 9, 0), Position(0, 0, 1)));
  EXPECT_FALSE(comparePositions(Position(0, 0, 1), Position(9, 9, 0)));
  EXPECT_TRUE(comparePositions(Position(9, 0, 1), Position(0, 1, 1)));
  EXPECT_FALSE(comparePositions(Position(0, 1, 1), Position(9, 0, 1)));
  EXPECT_TRUE(comparePositions(Position(0, 1, 1), Position(1, 1, 1)));
  EXPECT_FALSE(comparePositions(Position(1, 1, 1), Position(0, 1, 1)));
}

TEST(PositionOrderTests, IsIrreflexiveAndAsymmetric) {
  std::mt19937 generator(3);
  std::uniform_int_distribution<int> coordinate(0, 3);
  std::vector<Position> points;
  for (unsigned int i = 0; i < 40; i++) points.emplace_back(0.5 * coordinate(generator), 0.5 * coordinate(generator), 0.5 * coordinate(generator));
  for (const Position & a : points) {
    EXPECT_FALSE(comparePositions(a, a));
    for (const Position & b : points) {
      EXPECT_FALSE(comparePositions(a, b) && comparePositions(b, a));
      if (key(a) != key(b)) {
        EXPECT_TRUE(comparePositions(a, b) || comparePositions(b, a));
      }
    }
  }
}

// Differences of round-off size in z and y must not change the order: two matching surfaces of neighboring processes
// have to be sorted identically.
TEST(PositionOrderTests, IgnoresRoundOffInZAndY) {
  EXPECT_TRUE(comparePositions(Position(0, 0, 1 + 1e-9), Position(1, 0, 1)));
  EXPECT_TRUE(comparePositions(Position(0, 2 + 1e-9, 1), Position(1, 2, 1 + 1e-9)));
  EXPECT_FALSE(comparePositions(Position(1, 2, 1 - 1e-9), Position(0, 2 + 1e-9, 1)));
}

TEST(PositionOrderTests, SortingAGridGivesZMajorOrder) {
  std::vector<Position> points;
  for (int x = 0; x < 3; x++) {
    for (int y = 0; y < 3; y++) {
      for (int z = 0; z < 3; z++) points.emplace_back(0.5 * x, 0.25 * y, 2.0 * z);
    }
  }
  std::shuffle(points.begin(), points.end(), std::mt19937(11));
  std::sort(points.begin(), points.end(), comparePositions);
  unsigned int index = 0;
  for (int z = 0; z < 3; z++) {
    for (int y = 0; y < 3; y++) {
      for (int x = 0; x < 3; x++) {
        EXPECT_LT(points[index].distance(Position(0.5 * x, 0.25 * y, 2.0 * z)), 1e-12) << index;
        index++;
      }
    }
  }
}

TEST(PositionOrderTests, WrappersCompareByPosition) {
  const Position lower(1, 0, 0);
  const Position upper(0, 0, 1);
  EXPECT_TRUE(compareDofBaseData(std::make_pair(DofNumber(7), lower), std::make_pair(DofNumber(3), upper)));
  EXPECT_FALSE(compareDofBaseData(std::make_pair(DofNumber(3), upper), std::make_pair(DofNumber(7), lower)));
  SurfaceCellData cell_lower, cell_upper;
  cell_lower.surface_face_center = lower;
  cell_upper.surface_face_center = upper;
  EXPECT_TRUE(compareSurfaceCellData(cell_lower, cell_upper));
  EXPECT_FALSE(compareSurfaceCellData(cell_upper, cell_lower));
  FEAdjointEvaluation evaluation_lower, evaluation_upper;
  evaluation_lower.x = lower;
  evaluation_upper.x = upper;
  EXPECT_TRUE(compareFEAdjointEvals(evaluation_lower, evaluation_upper));
  EXPECT_FALSE(compareFEAdjointEvals(evaluation_upper, evaluation_lower));
}

TEST(InterfaceDofTests, OrderByGlobalIndex) {
  EXPECT_TRUE(compareDofDataByGlobalIndex(interface_dof(3, 5, 5, 5), interface_dof(4, 0, 0, 0)));
  EXPECT_FALSE(compareDofDataByGlobalIndex(interface_dof(4, 0, 0, 0), interface_dof(3, 5, 5, 5)));
  EXPECT_FALSE(compareDofDataByGlobalIndex(interface_dof(4, 0, 0, 0), interface_dof(4, 5, 5, 5)));
}

TEST(InterfaceDofTests, DofsWithTheSameBasePointAreOrderedByOrderThenIndex) {
  const InterfaceDofData first = interface_dof(20, 1, 1, 1, 0);
  const InterfaceDofData second = interface_dof(10, 1, 1, 1, 1);
  const InterfaceDofData third = interface_dof(11, 1, 1, 1, 1);
  const InterfaceDofData elsewhere = interface_dof(0, 0, 0, 2, 5);
  EXPECT_TRUE(compareDofBaseDataAndOrientation(first, second));
  EXPECT_FALSE(compareDofBaseDataAndOrientation(second, first));
  EXPECT_TRUE(compareDofBaseDataAndOrientation(second, third));
  EXPECT_FALSE(compareDofBaseDataAndOrientation(third, second));
  EXPECT_FALSE(compareDofBaseDataAndOrientation(first, first));
  // A different base point decides before order and index.
  EXPECT_TRUE(compareDofBaseDataAndOrientation(third, elsewhere));
  EXPECT_FALSE(compareDofBaseDataAndOrientation(elsewhere, first));
}

// Two processes see the same interface dofs with different indices and in a different order. The coupling has to
// pair the dofs that have the same base point and the same order.
TEST(InterfaceCouplingTests, PairsDofsWithTheSameBasePointAndOrder) {
  std::vector<InterfaceDofData> side_1, side_2;
  DofNumber index = 0;
  for (int x = 0; x < 4; x++) {
    for (int y = 0; y < 3; y++) {
      for (unsigned int order = 0; order < 2; order++) {
        side_1.push_back(interface_dof(index, 0.5 * x, 0.25 * y, 3.0, order));
        side_2.push_back(interface_dof(1000 + 7 * index, 0.5 * x, 0.25 * y, 3.0, order));
        index++;
      }
    }
  }
  std::map<DofNumber, InterfaceDofData> by_index_1, by_index_2;
  for (const auto & dof : side_1) by_index_1[dof.index] = dof;
  for (const auto & dof : side_2) by_index_2[dof.index] = dof;
  std::shuffle(side_1.begin(), side_1.end(), std::mt19937(1));
  std::shuffle(side_2.begin(), side_2.end(), std::mt19937(2));
  const std::vector<DofCouplingInformation> coupling = get_coupling_information(side_1, side_2);
  ASSERT_EQ(coupling.size(), by_index_1.size());
  std::set<DofNumber> seen;
  for (const DofCouplingInformation & pair : coupling) {
    ASSERT_EQ(by_index_1.count(pair.first_dof), 1u);
    ASSERT_EQ(by_index_2.count(pair.second_dof), 1u);
    EXPECT_LT(by_index_1[pair.first_dof].base_point.distance(by_index_2[pair.second_dof].base_point), 1e-12);
    EXPECT_EQ(by_index_1[pair.first_dof].order, by_index_2[pair.second_dof].order);
    EXPECT_EQ(pair.coupling_value, 1.0);
    seen.insert(pair.first_dof);
  }
  EXPECT_EQ(seen.size(), by_index_1.size());
}

TEST(InterfaceCouplingTests, ConstraintsIdentifyLargerIndexWithSmallerIndex) {
  std::vector<InterfaceDofData> side_1, side_2;
  // Indices 0..3 on side 1 and 4..7 on side 2, except for one pair where side 1 has the larger index.
  const DofNumber indices_1[4] = {0, 1, 9, 3};
  const DofNumber indices_2[4] = {4, 5, 2, 7};
  for (unsigned int i = 0; i < 4; i++) {
    side_1.push_back(interface_dof(indices_1[i], 0.5 * i, 0, 1));
    side_2.push_back(interface_dof(indices_2[i], 0.5 * i, 0, 1));
  }
  std::reverse(side_2.begin(), side_2.end());
  Constraints constraints = get_affine_constraints_for_InterfaceData(side_1, side_2, 10);
  constraints.close();
  EXPECT_EQ(constraints.n_constraints(), 4u);
  for (unsigned int i = 0; i < 4; i++) {
    const DofNumber larger = std::max(indices_1[i], indices_2[i]);
    const DofNumber smaller = std::min(indices_1[i], indices_2[i]);
    ASSERT_TRUE(constraints.is_constrained(larger)) << larger;
    EXPECT_FALSE(constraints.is_constrained(smaller)) << smaller;
    const auto * entries = constraints.get_constraint_entries(larger);
    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(entries->size(), 1u);
    EXPECT_EQ((*entries)[0].first, smaller);
    EXPECT_EQ((*entries)[0].second, ComplexNumber(1, 0));
    EXPECT_EQ(constraints.get_inhomogeneity(larger), ComplexNumber(0, 0));
  }
}

// Maps the corners of [-1,1]^3 to the corners of the subdomain (SquareMeshGenerator::prepare_triangulation, which
// is not called). The code writes the z-coordinate to component 3 of the point, which is out of bounds; the Release
// build stops with a trap instruction in every call. The call is made in a child process so that the test fails
// instead of ending the test program.
TEST(GeometryMappingTests, DISABLED_ReferenceCubeCornersAreMappedToLocalGeometry) {
  ScopedGeometry geometry;
  Geometry.local_x_range = std::make_pair(-2.0, 3.0);
  Geometry.local_y_range = std::make_pair(-5.0, 7.0);
  Geometry.local_z_range = std::make_pair(11.0, 13.0);
  EXPECT_EXIT({
    const Position lower = Triangulation_Shit_To_Local_Geometry(Position(-1, -1, -1));
    const Position upper = Triangulation_Shit_To_Local_Geometry(Position(1, 1, 1));
    const bool correct = lower.distance(Position(-2, -5, 11)) < 1e-14 && upper.distance(Position(3, 7, 13)) < 1e-14;
    std::_Exit(correct ? 0 : 1);
  }, testing::ExitedWithCode(0), "");
}

TEST(SurfaceTransformTests, BackTransformationIsTheInverse) {
  const Position p(0.3, -1.7, 2.9);
  for (unsigned int b = 0; b < 5; b++) {
    EXPECT_LT(from_5[b](to_5[b](p)).distance(p), 1e-14) << "boundary " << b;
    EXPECT_LT(to_5[b](from_5[b](p)).distance(p), 1e-14) << "boundary " << b;
  }
}

// Boundary b is a plane with a constant coordinate b / 2. After the change of coordinates this has to be the third
// coordinate, because the surface mesh is taken from the xy-plane (GeometryManager::initialize_inner_domain), and
// lengths have to be preserved.
TEST(SurfaceTransformTests, NormalCoordinateBecomesThirdCoordinateAndLengthsArePreserved) {
  const Position p(0.3, -1.7, 2.9);
  const Position q(-0.4, 0.6, 1.1);
  for (unsigned int b = 0; b < 5; b++) {
    const unsigned int normal_direction = b / 2;
    EXPECT_DOUBLE_EQ(std::abs(to_5[b](p)[2]), std::abs(p[normal_direction])) << "boundary " << b;
    EXPECT_NEAR(to_5[b](p).distance(to_5[b](q)), p.distance(q), 1e-14) << "boundary " << b;
  }
}

// The outward normal of the upper boundaries (+x, +y) is mapped to the outward normal +z of boundary 5.
TEST(SurfaceTransformTests, OutwardNormalOfUpperBoundariesIsMappedToPlusZ) {
  EXPECT_LT(Transform_1_to_5(Position(1, 0, 0)).distance(Position(0, 0, 1)), 1e-14);
  EXPECT_LT(Transform_3_to_5(Position(0, 1, 0)).distance(Position(0, 0, 1)), 1e-14);
}

TEST(MeshSizeTests, FineCellsInTheCenterOfTheCoreAndCoarseCellsFarOutside) {
  ScopedGlobalParams params;
  GlobalParams.Width_of_waveguide = 2.0;
  GlobalParams.Height_of_waveguide = 1.8;
  EXPECT_EQ(hmax_for_cell_center(Position(0, 0, 0)), 0.05);
  EXPECT_EQ(hmax_for_cell_center(Position(0.5, 0.5, 3)), 0.05);
  EXPECT_EQ(hmax_for_cell_center(Position(3, 3, 0)), 0.1);
  EXPECT_EQ(hmax_for_cell_center(Position(1.5, 0, 0)), 0.1);
}

// The core is |x| < width / 2 and |y| < height / 2. The code compares the x-coordinate with both the width and the
// height and ignores y.
TEST(MeshSizeTests, DISABLED_CoreIsLimitedInYByItsHeight) {
  ScopedGlobalParams params;
  GlobalParams.Width_of_waveguide = 2.0;
  GlobalParams.Height_of_waveguide = 1.8;
  // Above the core.
  EXPECT_EQ(hmax_for_cell_center(Position(0, 2.0, 0)), 0.1);
  // Inside the core, between height / 2 and width / 2 in x.
  EXPECT_EQ(hmax_for_cell_center(Position(0.95, 0, 0)), 0.05);
}

TEST(InterpolationPolynomialTests, FulfillsTheFourHermiteConditions) {
  const double v0 = 0.7, v1 = -1.3, d0 = 2.5, d1 = -0.4;
  EXPECT_DOUBLE_EQ(InterpolationPolynomial(0, v0, v1, d0, d1), v0);
  EXPECT_NEAR(InterpolationPolynomial(1, v0, v1, d0, d1), v1, 1e-14);
  EXPECT_DOUBLE_EQ(InterpolationPolynomialDerivative(0, v0, v1, d0, d1), d0);
  EXPECT_NEAR(InterpolationPolynomialDerivative(1, v0, v1, d0, d1), d1, 1e-14);
}

// The interpolant of a cubic polynomial is the polynomial itself.
TEST(InterpolationPolynomialTests, ReproducesCubicPolynomials) {
  auto p = [](double z) { return 2 * z * z * z - 3.5 * z * z + 0.25 * z + 1.5; };
  auto dp = [](double z) { return 6 * z * z - 7 * z + 0.25; };
  for (double z = 0; z <= 1.0; z += 0.0625) {
    EXPECT_NEAR(InterpolationPolynomial(z, p(0), p(1), dp(0), dp(1)), p(z), 1e-14) << z;
    EXPECT_NEAR(InterpolationPolynomialDerivative(z, p(0), p(1), dp(0), dp(1)), dp(z), 1e-13) << z;
  }
}

TEST(InterpolationPolynomialTests, DerivativeMatchesFiniteDifferences) {
  const double v0 = 0.7, v1 = -1.3, d0 = 2.5, d1 = -0.4;
  const double h = 1e-5;
  for (double z = 0.05; z < 1.0; z += 0.1) {
    const double difference = (InterpolationPolynomial(z + h, v0, v1, d0, d1) - InterpolationPolynomial(z - h, v0, v1, d0, d1)) / (2 * h);
    EXPECT_NEAR(InterpolationPolynomialDerivative(z, v0, v1, d0, d1), difference, 1e-8) << z;
  }
}

TEST(InterpolationPolynomialTests, IsConstantOutsideOfUnitInterval) {
  const double v0 = 0.7, v1 = -1.3, d0 = 2.5, d1 = -0.4;
  EXPECT_EQ(InterpolationPolynomial(-0.5, v0, v1, d0, d1), v0);
  EXPECT_EQ(InterpolationPolynomial(1.5, v0, v1, d0, d1), v1);
}

TEST(InterpolationPolynomialTests, ZeroDerivativeVersionIsASmoothStep) {
  const double v0 = 0.7, v1 = -1.3;
  EXPECT_DOUBLE_EQ(InterpolationPolynomialZeroDerivative(0, v0, v1), v0);
  EXPECT_NEAR(InterpolationPolynomialZeroDerivative(1, v0, v1), v1, 1e-14);
  // Point symmetry with respect to the center of the step.
  EXPECT_NEAR(InterpolationPolynomialZeroDerivative(0.5, v0, v1), 0.5 * (v0 + v1), 1e-14);
  for (double z = 0.1; z < 0.5; z += 0.1) {
    EXPECT_NEAR(InterpolationPolynomialZeroDerivative(z, v0, v1) + InterpolationPolynomialZeroDerivative(1 - z, v0, v1), v0 + v1, 1e-14);
  }
  // Monotone between the two values.
  for (double z = 0.0; z < 1.0; z += 0.05) {
    EXPECT_GE(InterpolationPolynomialZeroDerivative(z, v0, v1), InterpolationPolynomialZeroDerivative(z + 0.05, v0, v1));
  }
  const double h = 1e-4;
  EXPECT_NEAR((InterpolationPolynomialZeroDerivative(h, v0, v1) - v0) / h, 0, 1e-3);
  EXPECT_NEAR((v1 - InterpolationPolynomialZeroDerivative(1 - h, v0, v1)) / h, 0, 1e-3);
}

TEST(SigmaTests, LinearRampBetweenMinAndMax) {
  EXPECT_EQ(sigma(1.0, 2.0, 4.0), 0.0);
  EXPECT_EQ(sigma(2.0, 2.0, 4.0), 0.0);
  EXPECT_DOUBLE_EQ(sigma(2.5, 2.0, 4.0), 0.25);
  EXPECT_DOUBLE_EQ(sigma(3.0, 2.0, 4.0), 0.5);
  EXPECT_EQ(sigma(4.0, 2.0, 4.0), 1.0);
  EXPECT_EQ(sigma(7.0, 2.0, 4.0), 1.0);
}

TEST(SigmaTests, StepForEmptyInterval) {
  EXPECT_EQ(sigma(1.9, 2.0, 2.0), 0.0);
  EXPECT_EQ(sigma(2.1, 2.0, 2.0), 1.0);
}

TEST(EdgeOrientationTests, AxisParallelEdgesThroughTheOrigin) {
  EXPECT_TRUE(get_orientation(Position(0, 0, 0), Position(1, 0, 0)));
  EXPECT_FALSE(get_orientation(Position(0, 0, 0), Position(-1, 0, 0)));
  EXPECT_TRUE(get_orientation(Position(0, 0, 0), Position(0, 0, 2)));
  EXPECT_FALSE(get_orientation(Position(0, 0, 0), Position(0, 0, -2)));
}

// An edge and the reversed edge have opposite orientations, and the orientation does not depend on where the edge
// is located. The code determines the dominant direction from vertex_2[i] - vertex_1[1] instead of
// vertex_2[i] - vertex_1[i], so both properties fail for edges away from the origin.
TEST(EdgeOrientationTests, DISABLED_OrientationOfEdgesAwayFromTheOrigin) {
  // An edge in the +x direction at y = 5.
  EXPECT_TRUE(get_orientation(Position(0, 5, 0), Position(1, 5, 0)));
  EXPECT_FALSE(get_orientation(Position(1, 5, 0), Position(0, 5, 0)));
  // An edge in the -z direction at y = 5.
  EXPECT_FALSE(get_orientation(Position(2, 5, 1), Position(2, 5, 0)));
  EXPECT_TRUE(get_orientation(Position(2, 5, 0), Position(2, 5, 1)));
}

TEST(CrossProductTests, UnitVectorsAndAnticommutativity) {
  NumericVectorLocal e_x(3), e_y(3), e_z(3);
  e_x[0] = 1;
  e_y[1] = 1;
  e_z[2] = 1;
  NumericVectorLocal result = crossproduct(e_x, e_y);
  result -= e_z;
  EXPECT_LT(result.l2_norm(), 1e-15);
  result = crossproduct(e_y, e_z);
  result -= e_x;
  EXPECT_LT(result.l2_norm(), 1e-15);
  NumericVectorLocal u(3), v(3);
  u[0] = ComplexNumber(1, 2);
  u[1] = ComplexNumber(-0.5, 0.3);
  u[2] = ComplexNumber(0, 1);
  v[0] = ComplexNumber(0.2, -1);
  v[1] = ComplexNumber(3, 0);
  v[2] = ComplexNumber(-1, -1);
  const NumericVectorLocal uv = crossproduct(u, v);
  const NumericVectorLocal vu = crossproduct(v, u);
  ComplexNumber u_dot_uv(0, 0), v_dot_uv(0, 0);
  for (unsigned int i = 0; i < 3; i++) {
    EXPECT_LT(std::abs(uv[i] + vu[i]), 1e-14);
    // The bilinear product without conjugation vanishes.
    u_dot_uv += u[i] * uv[i];
    v_dot_uv += v[i] * uv[i];
  }
  EXPECT_LT(std::abs(u_dot_uv), 1e-13);
  EXPECT_LT(std::abs(v_dot_uv), 1e-13);
}

// (i e_x) x (i e_y) = - e_z. A product that conjugates one argument would give + e_z.
TEST(CrossProductTests, DoesNotConjugate) {
  NumericVectorLocal u(3), v(3);
  u[0] = ComplexNumber(0, 1);
  v[1] = ComplexNumber(0, 1);
  const NumericVectorLocal result = crossproduct(u, v);
  EXPECT_EQ(result[0], ComplexNumber(0, 0));
  EXPECT_EQ(result[1], ComplexNumber(0, 0));
  EXPECT_EQ(result[2], ComplexNumber(-1, 0));
}

// On a single process a message is printed if its level is at least the configured level in the order
// DEBUG_ALL < DEBUG_ONE < PRODUCTION_ALL < PRODUCTION_ONE.
TEST(LoggingTests, VisibilityFollowsTheOrderOfTheLevels) {
  ScopedGlobalParams params;
  const LoggingLevel levels[4] = {LoggingLevel::DEBUG_ALL, LoggingLevel::DEBUG_ONE, LoggingLevel::PRODUCTION_ALL, LoggingLevel::PRODUCTION_ONE};
  for (unsigned int configured = 0; configured < 4; configured++) {
    GlobalParams.Logging_Level = levels[configured];
    for (unsigned int message = 0; message < 4; message++) {
      EXPECT_EQ(is_visible_message_in_current_logging_level(levels[message]), message >= configured) << configured << " " << message;
    }
  }
}

TEST(LoggingTests, PrintInfoWritesRankBlockIndicesLabelAndMessage) {
  ScopedGlobalParams params;
  GlobalParams.MPI_Rank = 5;
  GlobalParams.Index_in_x_direction = 1;
  GlobalParams.Index_in_y_direction = 2;
  GlobalParams.Index_in_z_direction = 3;
  GlobalParams.Logging_Level = LoggingLevel::DEBUG_ALL;
  testing::internal::CaptureStdout();
  print_info("Some::function", "the message", LoggingLevel::PRODUCTION_ONE);
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output.rfind("[5:1x2x3]", 0), 0u) << output;
  EXPECT_NE(output.find("Some::function"), std::string::npos);
  EXPECT_NE(output.find(": the message\n"), std::string::npos);
  EXPECT_EQ(std::count(output.begin(), output.end(), '\n'), 1);
}

TEST(LoggingTests, PrintInfoSuppressesMessagesBelowTheConfiguredLevel) {
  ScopedGlobalParams params;
  GlobalParams.Logging_Level = LoggingLevel::PRODUCTION_ALL;
  testing::internal::CaptureStdout();
  print_info("label", "hidden", LoggingLevel::DEBUG_ONE);
  print_info("label", 17u, LoggingLevel::DEBUG_ALL);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "");
}

TEST(LoggingTests, PrintInfoOverloadsFormatNumbersListsAndFlags) {
  ScopedGlobalParams params;
  GlobalParams.Logging_Level = LoggingLevel::DEBUG_ALL;
  testing::internal::CaptureStdout();
  print_info("label", 17u, LoggingLevel::PRODUCTION_ONE);
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find(": 17\n"), std::string::npos) << output;
  testing::internal::CaptureStdout();
  print_info("label", std::vector<unsigned int>{3, 1, 4}, LoggingLevel::PRODUCTION_ONE);
  output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find(": 3 1 4"), std::string::npos) << output;
  testing::internal::CaptureStdout();
  print_info("label", std::array<bool, 6>{true, false, false, true, false, true}, LoggingLevel::PRODUCTION_ONE);
  output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find(": 0 true 1 false 2 false 3 true 4 false 5 true"), std::string::npos) << output;
}

TEST(BoundaryIdTests, OpposingBoundaryIsAnInvolutionInTheSameDirection) {
  for (BoundaryId b = 0; b < 6; b++) {
    const BoundaryId opposite = opposing_Boundary_Id(b);
    EXPECT_NE(opposite, b);
    EXPECT_EQ(opposite / 2, b / 2);
    EXPECT_EQ(opposing_Boundary_Id(opposite), b);
    EXPECT_TRUE(are_opposing_sites(b, opposite));
  }
}

TEST(BoundaryIdTests, OpposingSitesAreExactlyThePairsOfOneDirection) {
  for (BoundaryId a = 0; a < 6; a++) {
    for (BoundaryId b = 0; b < 6; b++) {
      EXPECT_EQ(are_opposing_sites(a, b), a != b && a / 2 == b / 2) << a << " " << b;
    }
  }
}

TEST(BoundaryIdTests, AdjacentBoundariesAreTheFourOthers) {
  for (BoundaryId b = 0; b < 6; b++) {
    const std::vector<BoundaryId> adjacent = get_adjacent_boundary_ids(b);
    ASSERT_EQ(adjacent.size(), 4u);
    EXPECT_TRUE(std::is_sorted(adjacent.begin(), adjacent.end()));
    for (BoundaryId other : adjacent) {
      EXPECT_NE(other, b);
      EXPECT_NE(other, opposing_Boundary_Id(b));
      EXPECT_LT(other, 6u);
      // Adjacency is symmetric.
      const std::vector<BoundaryId> back = get_adjacent_boundary_ids(other);
      EXPECT_NE(std::find(back.begin(), back.end(), b), back.end());
    }
    EXPECT_EQ(std::set<BoundaryId>(adjacent.begin(), adjacent.end()).size(), 4u);
  }
}

TEST(BoundaryIdTests, OnlyAbsorbingSurfaceTypeIsAbsorbing) {
  EXPECT_TRUE(is_absorbing_boundary(SurfaceType::ABC_SURFACE));
  EXPECT_FALSE(is_absorbing_boundary(SurfaceType::DIRICHLET_SURFACE));
  EXPECT_FALSE(is_absorbing_boundary(SurfaceType::OPEN_SURFACE));
}

// If all six surfaces carry the same boundary method, the dofs on each of the twelve edges of the domain have to be
// numbered by exactly one of the two surfaces that meet there.
TEST(EdgeOwnershipTests, ExactlyOneOfTwoEqualSurfacesOwnsTheCommonEdge) {
  ScopedGeometry geometry;
  const unsigned int level = 1;
  for (SurfaceType type : {SurfaceType::ABC_SURFACE, SurfaceType::OPEN_SURFACE}) {
    Geometry.levels[level].surface_type.fill(type);
    for (BoundaryId a = 0; a < 6; a++) {
      for (BoundaryId b : get_adjacent_boundary_ids(a)) {
        EXPECT_NE(are_edge_dofs_locally_owned(a, b, level), are_edge_dofs_locally_owned(b, a, level)) << a << " " << b;
      }
    }
  }
}

// The dofs on the edge between an absorbing surface and the interface to a neighboring process belong to the process
// with the lower index: they are owned if the neighbor is on the upper side (odd boundary id).
TEST(EdgeOwnershipTests, EdgeToNeighborIsOwnedIfTheNeighborIsOnTheUpperSide) {
  ScopedGeometry geometry;
  const unsigned int level = 1;
  for (BoundaryId neighbor = 0; neighbor < 6; neighbor++) {
    Geometry.levels[level].surface_type.fill(SurfaceType::ABC_SURFACE);
    Geometry.levels[level].surface_type[neighbor] = SurfaceType::NEIGHBOR_SURFACE;
    for (BoundaryId self : get_adjacent_boundary_ids(neighbor)) {
      EXPECT_EQ(are_edge_dofs_locally_owned(self, neighbor, level), neighbor % 2 == 1) << self << " " << neighbor;
    }
  }
}

TEST(SweepingDirectionTests, HighestLevelSweepsInZ) {
  ScopedGlobalParams params;
  GlobalParams.Sweeping_Level = 1;
  EXPECT_EQ(get_sweeping_direction_for_level(1), SweepingDirection::Z);
  GlobalParams.Sweeping_Level = 2;
  EXPECT_EQ(get_sweeping_direction_for_level(1), SweepingDirection::Y);
  EXPECT_EQ(get_sweeping_direction_for_level(2), SweepingDirection::Z);
  GlobalParams.Sweeping_Level = 3;
  EXPECT_EQ(get_sweeping_direction_for_level(1), SweepingDirection::X);
  EXPECT_EQ(get_sweeping_direction_for_level(2), SweepingDirection::Y);
  EXPECT_EQ(get_sweeping_direction_for_level(3), SweepingDirection::Z);
}

TEST(SplitTests, SplitsAtEveryDelimiterAndKeepsEmptyParts) {
  EXPECT_EQ(split("a,b,c", ","), (std::vector<std::string>{"a", "b", "c"}));
  EXPECT_EQ(split("abc", ","), (std::vector<std::string>{"abc"}));
  EXPECT_EQ(split("", ","), (std::vector<std::string>{""}));
  EXPECT_EQ(split("a,,b,", ","), (std::vector<std::string>{"a", "", "b", ""}));
  EXPECT_EQ(split("key=value", "="), (std::vector<std::string>{"key", "value"}));
  EXPECT_EQ(split("a::b::c", "::"), (std::vector<std::string>{"a", "b", "c"}));
}

TEST(SplitTests, JoiningThePartsGivesTheInput) {
  for (const std::string input : {"x=1;y=2;z", ";;", "no delimiter", ";start", "end;"}) {
    const std::vector<std::string> parts = split(input, ";");
    std::string joined = parts[0];
    for (unsigned int i = 1; i < parts.size(); i++) joined += ";" + parts[i];
    EXPECT_EQ(joined, input);
  }
}

TEST(SolverOptionTests, NamesAreMappedToTheSolvers) {
  EXPECT_EQ(solver_option("GMRES"), SolverOptions::GMRES);
  EXPECT_EQ(solver_option("MINRES"), SolverOptions::MINRES);
  EXPECT_EQ(solver_option("TFQMR"), SolverOptions::TFQMR);
  EXPECT_EQ(solver_option("BICGS"), SolverOptions::BICGS);
  EXPECT_EQ(solver_option("CG"), SolverOptions::S_CG);
  EXPECT_EQ(solver_option("PCONLY"), SolverOptions::PCONLY);
  EXPECT_EQ(solver_option("unknown"), SolverOptions::GMRES);
}

TEST(AdjointTransformationTests, PositionIsMirroredAtTheCenterOfTheZRange) {
  ScopedGeometry geometry;
  Geometry.global_z_range = std::make_pair(2.0, 10.0);
  EXPECT_LT(adjoint_position_transformation(Position(0.3, -0.4, 2.0)).distance(Position(0.3, -0.4, 10.0)), 1e-14);
  EXPECT_LT(adjoint_position_transformation(Position(0.3, -0.4, 10.0)).distance(Position(0.3, -0.4, 2.0)), 1e-14);
  EXPECT_LT(adjoint_position_transformation(Position(0.3, -0.4, 6.0)).distance(Position(0.3, -0.4, 6.0)), 1e-14);
  const Position p(1.5, 0.7, 3.25);
  EXPECT_LT(adjoint_position_transformation(adjoint_position_transformation(p)).distance(p), 1e-14);
}

TEST(AdjointTransformationTests, FieldTransformationChangesTheSignOfXAndZ) {
  dealii::Tensor<1, 3, ComplexNumber> field;
  field[0] = ComplexNumber(1, 2);
  field[1] = ComplexNumber(3, 4);
  field[2] = ComplexNumber(5, 6);
  const dealii::Tensor<1, 3, ComplexNumber> transformed = adjoint_field_transformation(field);
  EXPECT_EQ(transformed[0], ComplexNumber(-1, -2));
  EXPECT_EQ(transformed[1], ComplexNumber(3, 4));
  EXPECT_EQ(transformed[2], ComplexNumber(-5, -6));
  EXPECT_EQ((adjoint_field_transformation(transformed) - field).norm(), 0.0);
}

// Packing the evaluations for MPI and unpacking them on the receiving side gives the primal field as adjoint data on
// the mirrored geometry.
TEST(AdjointTransformationTests, PackingAndUnpackingOfFieldEvaluations) {
  ScopedGeometry geometry;
  Geometry.global_z_range = std::make_pair(0.0, 8.0);
  std::vector<FEAdjointEvaluation> evaluations(3);
  for (unsigned int i = 0; i < 3; i++) {
    evaluations[i].x = Position(0.1 * i, -0.2 * i, 1.0 + i);
    for (unsigned int c = 0; c < 3; c++) evaluations[i].primal_field[c] = ComplexNumber(1.0 + i + 0.1 * c, -2.0 * i + 0.3 * c);
  }
  const std::vector<double> packed = fe_evals_to_double(evaluations);
  ASSERT_EQ(packed.size(), 27u);
  const std::vector<FEAdjointEvaluation> unpacked = fe_evals_from_double(packed);
  ASSERT_EQ(unpacked.size(), 3u);
  for (unsigned int i = 0; i < 3; i++) {
    EXPECT_LT(unpacked[i].x.distance(adjoint_position_transformation(evaluations[i].x)), 1e-14);
    EXPECT_DOUBLE_EQ(unpacked[i].x[2], 8.0 - evaluations[i].x[2]);
    EXPECT_EQ((unpacked[i].adjoint_field - adjoint_field_transformation(evaluations[i].primal_field)).norm(), 0.0);
  }
}

TEST(AdjointTransformationTests, EmptyListOfEvaluations) {
  EXPECT_TRUE(fe_evals_to_double(std::vector<FEAdjointEvaluation>()).empty());
  EXPECT_TRUE(fe_evals_from_double(std::vector<double>()).empty());
}

// The renumbered mesh has to describe the same geometry: same cells, same vertices and the same boundary ids at the
// same places. Its vertices are numbered in the order of comparePositions.
TEST(ReforgeTriangulationTests, KeepsGeometryAndBoundaryIdsAndSortsVertices) {
  dealii::Triangulation<3> original;
  dealii::GridGenerator::subdivided_hyper_rectangle(original, {3, 2, 2}, Position(-1, 0, 2), Position(2, 1, 4), true);
  dealii::Triangulation<3> reforged = reforge_triangulation(&original);
  ASSERT_EQ(reforged.n_active_cells(), original.n_active_cells());
  ASSERT_EQ(reforged.n_vertices(), original.n_vertices());
  const std::vector<Position> & vertices = reforged.get_vertices();
  for (unsigned int i = 0; i + 1 < vertices.size(); i++) {
    EXPECT_TRUE(comparePositions(vertices[i], vertices[i + 1])) << i;
  }
  std::set<std::array<long, 3>> original_vertices, reforged_vertices, original_cells, reforged_cells;
  for (const Position & p : original.get_vertices()) original_vertices.insert(key(p));
  for (const Position & p : vertices) reforged_vertices.insert(key(p));
  EXPECT_EQ(original_vertices, reforged_vertices);
  std::map<std::array<long, 3>, unsigned int> original_boundary_ids, reforged_boundary_ids;
  double original_volume = 0, reforged_volume = 0;
  for (const auto & cell : original.active_cell_iterators()) {
    original_cells.insert(key(cell->center()));
    original_volume += cell->measure();
    for (unsigned int face = 0; face < 6; face++) {
      if (cell->face(face)->at_boundary()) original_boundary_ids[key(cell->face(face)->center())] = cell->face(face)->boundary_id();
    }
  }
  for (const auto & cell : reforged.active_cell_iterators()) {
    reforged_cells.insert(key(cell->center()));
    reforged_volume += cell->measure();
    for (unsigned int face = 0; face < 6; face++) {
      if (cell->face(face)->at_boundary()) reforged_boundary_ids[key(cell->face(face)->center())] = cell->face(face)->boundary_id();
    }
  }
  EXPECT_EQ(original_cells, reforged_cells);
  EXPECT_NEAR(original_volume, 6.0, 1e-12);
  EXPECT_NEAR(reforged_volume, 6.0, 1e-12);
  EXPECT_EQ(original_boundary_ids.size(), 2u * (3 * 2 + 3 * 2 + 2 * 2));
  EXPECT_EQ(original_boundary_ids, reforged_boundary_ids);
}
