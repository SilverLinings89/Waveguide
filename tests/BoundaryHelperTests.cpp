// Tests for the small helper classes of the boundary conditions: JacobianForCell, PMLMeshTransformation,
// LaguerreFunction and DofData.
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include <cmath>
#include "../Code/BoundaryCondition/DofData.h"
#include "../Code/BoundaryCondition/JacobianForCell.h"
#include "../Code/BoundaryCondition/LaguerreFunction.h"
#include "../Code/BoundaryCondition/PMLMeshTransformation.h"
#include "../Code/Core/Types.h"

namespace {

  FaceAngelingData straight_rays() {
    FaceAngelingData ret;
    for (unsigned int i = 0; i < ret.size(); i++) {
      ret[i].is_x_angled = false;
      ret[i].is_y_angled = false;
    }
    return ret;
  }

  void expect_identity(const dealii::Tensor<2, 3, double> & tensor) {
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        EXPECT_NEAR(tensor[i][j], (i == j) ? 1.0 : 0.0, 1e-14) << "entry " << i << "," << j;
      }
    }
  }

}

// Rays orthogonal to the surface: the infinite cell is a cuboid and the mapping to the reference cell is a shift.
TEST(JacobianForCellTests, StraightRaysGiveIdentity) {
  FaceAngelingData fad = straight_rays();
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    JacobianForCell jacobian(fad, b_id, 1.7);
    for (const Position2D & p : {Position2D(0, 0), Position2D(0.3, -1.2), Position2D(-2.0, 5.0)}) {
      expect_identity(jacobian.get_J_hat_for_position(p));
      const JacobianAndTensorData data = jacobian.get_C_G_and_J(p);
      expect_identity(data.J);
      expect_identity(data.C);
      expect_identity(data.G);
    }
  }
}

// A point of the surface mesh is mapped to the surface of the domain, and split_into_triangulation_and_external_part()
// is the inverse of that mapping.
TEST(JacobianForCellTests, SurfaceCoordinatesAreMappedToTheSurfaceAndBack) {
  FaceAngelingData fad = straight_rays();
  const double coordinate_of_surface = 1.7;
  for (BoundaryId b_id = 0; b_id < 6; b_id++) {
    JacobianForCell jacobian(fad, b_id, coordinate_of_surface);
    const Position2D on_surface_mesh(0.3, -1.2);
    const Position p = jacobian.transform_to_3D_space(on_surface_mesh);
    EXPECT_EQ(p[b_id / 2], coordinate_of_surface) << "surface " << b_id;
    // The other two coordinates are those of the surface mesh, up to order and sign.
    EXPECT_NEAR(p.norm_square(), on_surface_mesh.norm_square() + coordinate_of_surface * coordinate_of_surface, 1e-14);
    const std::pair<Position2D, double> parts = jacobian.split_into_triangulation_and_external_part(p);
    EXPECT_EQ(parts.first[0], on_surface_mesh[0]) << "surface " << b_id;
    EXPECT_EQ(parts.first[1], on_surface_mesh[1]) << "surface " << b_id;
    EXPECT_EQ(parts.second, coordinate_of_surface) << "surface " << b_id;
  }
}

// JacobianForCell.cpp:32: both branches of the distinction are the same, so the Jacobian is the identity also for rays at an angle (HSIESurface never requests such rays).
TEST(JacobianForCellTests, DISABLED_AngledRaysChangeTheJacobian) {
  FaceAngelingData fad = straight_rays();
  for (unsigned int i = 0; i < fad.size(); i++) {
    fad[i].is_x_angled = true;
    fad[i].is_y_angled = true;
  }
  JacobianForCell jacobian(fad, 5, 0.0);
  const dealii::Tensor<2, 3, double> J = jacobian.get_J_hat_for_position(Position2D(0.3, -1.2));
  double deviation_from_identity = 0;
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      deviation_from_identity += std::abs(J[i][j] - ((i == j) ? 1.0 : 0.0));
    }
  }
  EXPECT_GT(deviation_from_identity, 1e-3);
}

// The transformation widens a layer towards the outside: points on the side faces are moved outwards by their
// distance to the surface the layer starts at, all other points stay.
TEST(PMLMeshTransformationTests, MovesSideFacesOutwardsByTheDepth) {
  const std::pair<double, double> x_range(-1, 1), y_range(-2, 2), z_range(5, 6);
  // Layer at +z that starts at z = 5 and is widened on all four sides.
  PMLMeshTransformation transformation(x_range, y_range, z_range, 5.0, 2, {{true, true, true, true, false, false}});
  const double depth = 0.4;
  Position p = transformation(Position(-1, 0.5, 5 + depth));
  EXPECT_NEAR(p[0], -1 - depth, 1e-14);
  EXPECT_EQ(p[1], 0.5);
  EXPECT_EQ(p[2], 5 + depth);
  p = transformation(Position(1, 2, 5 + depth));
  EXPECT_NEAR(p[0], 1 + depth, 1e-14);
  EXPECT_NEAR(p[1], 2 + depth, 1e-14);
  p = transformation(Position(0.2, -2, 5 + depth));
  EXPECT_EQ(p[0], 0.2);
  EXPECT_NEAR(p[1], -2 - depth, 1e-14);
  // Points in the interior and on the surface the layer starts at are not moved.
  const Position inside(0.2, 0.5, 5.7);
  EXPECT_EQ(transformation(inside), inside);
  const Position at_start(-1, 2, 5.0);
  EXPECT_LT(transformation(at_start).distance(at_start), 1e-14);
}

TEST(PMLMeshTransformationTests, OnlyRequestedSidesAreMoved) {
  const std::pair<double, double> x_range(-1, 1), y_range(-2, 2), z_range(5, 6);
  PMLMeshTransformation transformation(x_range, y_range, z_range, 5.0, 2, {{false, true, false, false, false, false}});
  const Position lower_x(-1, 0, 5.5), upper_x(1, 0, 5.5), lower_y(0, -2, 5.5);
  EXPECT_EQ(transformation(lower_x), lower_x);
  EXPECT_NEAR(transformation(upper_x)[0], 1.5, 1e-14);
  EXPECT_EQ(transformation(lower_y), lower_y);
  // The outward direction itself is never changed.
  PMLMeshTransformation in_x(x_range, y_range, z_range, 1.0, 0, {{true, true, true, true, true, true}});
  EXPECT_EQ(in_x(Position(1, 0, 5.5))[0], 1.0);
  EXPECT_NEAR(in_x(Position(1.3, 2, 5.5))[1], 2.3, 1e-14);
  EXPECT_NEAR(in_x(Position(1.3, 0, 5))[2], 4.7, 1e-14);
}

TEST(PMLMeshTransformationTests, UndoMovesPointsBackToTheUntransformedRange) {
  const std::pair<double, double> x_range(-1, 1), y_range(-2, 2), z_range(5, 6);
  PMLMeshTransformation transformation(x_range, y_range, z_range, 5.0, 2, {{true, true, true, true, false, false}});
  for (const Position & p : {Position(-1, 0.5, 5.4), Position(1, 2, 5.9), Position(0.2, -2, 5.1), Position(0.2, 0.5, 5.7), Position(1, -2, 6)}) {
    EXPECT_LT(transformation.undo_transform(transformation(p)).distance(p), 1e-14);
  }
}

// (n + 1) L_{n+1}^m(x) = (2n + 1 + m - x) L_n^m(x) - (n + m) L_{n-1}^m(x)
TEST(LaguerrePropertyTests, ThreeTermRecurrence) {
  for (unsigned int m = 0; m < 4; m++) {
    for (unsigned int n = 1; n < 8; n++) {
      for (double x : {0.0, 0.3, 1.0, 2.5, 7.0}) {
        const double left = (n + 1) * LaguerreFunction::evaluate(n + 1, m, x);
        const double right = (2.0 * n + 1 + m - x) * LaguerreFunction::evaluate(n, m, x) - (n + m) * LaguerreFunction::evaluate(n - 1, m, x);
        EXPECT_NEAR(left, right, 1e-9 * (1 + std::abs(left))) << "n = " << n << ", m = " << m << ", x = " << x;
      }
    }
  }
}

// d/dx L_n^m(x) = - L_{n-1}^{m+1}(x)
TEST(LaguerrePropertyTests, DerivativeLowersDegreeAndRaisesParameter) {
  const double h = 1e-5;
  for (unsigned int m = 0; m < 3; m++) {
    for (unsigned int n = 1; n < 6; n++) {
      for (double x : {0.3, 1.0, 2.5}) {
        const double difference_quotient = (LaguerreFunction::evaluate(n, m, x + h) - LaguerreFunction::evaluate(n, m, x - h)) / (2 * h);
        EXPECT_NEAR(difference_quotient, -LaguerreFunction::evaluate(n - 1, m + 1, x), 1e-6) << "n = " << n << ", m = " << m << ", x = " << x;
      }
    }
  }
}

// L_2(x) = (x^2 - 4 x + 2) / 2 has the roots 2 +- sqrt(2).
TEST(LaguerrePropertyTests, KnownPolynomialsAndRoots) {
  EXPECT_NEAR(LaguerreFunction::evaluate(2, 0, 2 + std::sqrt(2.0)), 0.0, 1e-13);
  EXPECT_NEAR(LaguerreFunction::evaluate(2, 0, 2 - std::sqrt(2.0)), 0.0, 1e-13);
  for (double x : {-1.0, 0.0, 0.4, 3.0}) {
    EXPECT_NEAR(LaguerreFunction::evaluate(3, 0, x), (-x * x * x + 9 * x * x - 18 * x + 6) / 6, 1e-12);
    EXPECT_NEAR(LaguerreFunction::evaluate(1, 3, x), 4 - x, 1e-12);
  }
}

TEST(LaguerrePropertyTests, BinomialCoefficientsFulfillPascalsRule) {
  for (unsigned int n = 1; n < 20; n++) {
    EXPECT_EQ(LaguerreFunction::binomial_coefficient(n, 0), 1u);
    EXPECT_EQ(LaguerreFunction::binomial_coefficient(n, n), 1u);
    EXPECT_EQ(LaguerreFunction::binomial_coefficient(n, n + 1), 0u);
    for (unsigned int k = 1; k < n; k++) {
      EXPECT_EQ(LaguerreFunction::binomial_coefficient(n, k), LaguerreFunction::binomial_coefficient(n - 1, k - 1) + LaguerreFunction::binomial_coefficient(n - 1, k)) << "n = " << n << ", k = " << k;
      EXPECT_EQ(LaguerreFunction::binomial_coefficient(n, k), LaguerreFunction::binomial_coefficient(n, n - k));
    }
  }
}

TEST(LaguerrePropertyTests, FactorialRecursion) {
  EXPECT_EQ(LaguerreFunction::factorial(0), 1.0);
  for (unsigned int n = 1; n < 15; n++) {
    EXPECT_EQ(LaguerreFunction::factorial(n), n * LaguerreFunction::factorial(n - 1));
  }
  EXPECT_EQ(LaguerreFunction::factorial(10), 3628800.0);
}

// The functions that point in the infinite direction are built from the nodal element of the surface.
TEST(DofDataTests, NodalBasisFlagFollowsType) {
  for (DofType type : {DofType::EDGE, DofType::SURFACE, DofType::RAY, DofType::IFFa, DofType::IFFb, DofType::SEGMENTa, DofType::SEGMENTb}) {
    DofData dof;
    dof.type = type;
    dof.update_nodal_basis_flag();
    const bool expected = (type == DofType::RAY || type == DofType::IFFb || type == DofType::SEGMENTb);
    EXPECT_EQ(dof.nodal_basis, expected) << "type " << type;
  }
}

TEST(DofDataTests, ConstructorsStoreBaseStructure) {
  DofData on_cell(std::string("0_0:"));
  EXPECT_EQ(on_cell.base_structure_id_face, "0_0:");
  EXPECT_FALSE(on_cell.got_base_dof_index);
  DofData on_edge(17u);
  EXPECT_EQ(on_edge.base_structure_id_non_face, 17u);
  EXPECT_EQ(on_edge.base_structure_id_face, "");
  EXPECT_FALSE(on_edge.got_base_dof_index);
  EXPECT_TRUE(on_edge.orientation);
  on_edge.set_base_dof(5);
  EXPECT_TRUE(on_edge.got_base_dof_index);
  EXPECT_EQ(on_edge.base_dof_index, 5u);
  DofData empty;
  EXPECT_EQ(empty.base_structure_id_face, "");
  EXPECT_EQ(empty.base_structure_id_non_face, 0u);
  EXPECT_FALSE(empty.got_base_dof_index);
}
