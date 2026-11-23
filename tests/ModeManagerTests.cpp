// Tests for the two legacy classes for waveguide modes: ModeManager (analytical modes of a circular waveguide) and
// RectangularMode (mesh and numbering for a modal computation of the rectangular waveguide).
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <vector>
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/GlobalObjects/ModeManager.h"
#include "../Code/ModalComputations/RectangularMode.h"
#include "../Code/Solutions/ExactSolution.h"
#include "./HelpersB.h"

namespace {

  class ModeManagerTests : public helpers_b::GlobalStateTest {
  protected:
    // Normalized frequency 4: the characteristic equation has exactly one root u, between the first zeros of J_0
    // (2.405) and J_1 (3.832).
    void SetUp() override {
      helpers_b::GlobalStateTest::SetUp();
      GlobalParams.Waveguide_value_V = 2.0;
      manager.load();
    }

    double profile(double r) {
      return manager.get_input_component(0, Position(r, 0, 0), 0);
    }

    // The profile is J_0(r) / J_0(u) for r < u, so its value on the axis determines u.
    double radius_of_mode() {
      const double j0_of_u = 1.0 / profile(0.0);
      double lower = 2.405, upper = 3.83;
      for (unsigned int i = 0; i < 60; i++) {
        const double middle = 0.5 * (lower + upper);
        if (j0(middle) > j0_of_u) {
          lower = middle;
        } else {
          upper = middle;
        }
      }
      return 0.5 * (lower + upper);
    }

    ModeManager manager;
  };

  class RectangularModeTests : public helpers_b::GlobalStateTest {
  protected:
    void SetUp() override {
      helpers_b::GlobalStateTest::SetUp();
      GlobalParams.Geometry_Size_X = 6.0;
      GlobalParams.Geometry_Size_Y = 5.4;
      GlobalParams.Nedelec_element_order = 0;
    }
  };

  // Modified Bessel function of the second kind, K_order(x) = int_0^inf exp(-x cosh(t)) cosh(order t) dt, by the
  // trapezoidal rule (which converges exponentially for this integrand).
  double bessel_k(unsigned int order, double x) {
    const double dt = 1e-3;
    double sum = 0.5 * std::exp(-x);
    for (double t = dt; t < 12.0; t += dt) {
      sum += std::exp(-x * std::cosh(t)) * std::cosh(order * t);
    }
    return sum * dt;
  }

  bool is_downstream_or_equal(const Position & a, const Position & b) {
    for (const unsigned int axis : {2u, 1u, 0u}) {
      if (std::abs(a[axis] - b[axis]) > 1e-6) {
        return a[axis] < b[axis];
      }
    }
    return true;
  }

}

TEST_F(ModeManagerTests, ProfileInCoreIsBesselFunction) {
  const double u = radius_of_mode();
  ASSERT_GT(u, 2.405);
  ASSERT_LT(u, 3.83);
  for (const double r : {0.3, 1.0, 1.7, 2.4, u - 0.05}) {
    ASSERT_NEAR(profile(r) / profile(0.0), j0(r), 1e-6) << "at r = " << r;
  }
}

TEST_F(ModeManagerTests, ProfileInCladdingIsModifiedBesselFunction) {
  const double u = radius_of_mode();
  const double reference = profile(u + 0.05);
  double previous = reference;
  for (const double r : {u + 0.3, u + 1.0, u + 2.5, u + 6.0}) {
    ASSERT_NEAR(profile(r) / reference, bessel_k(0, r) / bessel_k(0, u + 0.05), 1e-6) << "at r = " << r;
    // The field decays monotonically without a change of sign.
    ASSERT_GT(profile(r), 0.0);
    ASSERT_LT(profile(r), previous);
    previous = profile(r);
  }
}

TEST_F(ModeManagerTests, ProfileIsContinuousAndNormalizedAtCoreBoundary) {
  const double u = radius_of_mode();
  ASSERT_NEAR(profile(u - 1e-5), 1.0, 1e-3);
  ASSERT_NEAR(profile(u + 1e-5), 1.0, 1e-3);
}

// The root is found by ten bisections of an interval of length 0.1.
TEST_F(ModeManagerTests, RadiusOfModeSolvesCharacteristicEquation) {
  const double u = radius_of_mode();
  auto residual = [&](double argument) {
    const double w = std::sqrt(16.0 - argument * argument);
    return -argument * j0(argument) / j1(argument) - w * bessel_k(0, w) / bessel_k(1, w);
  };
  ASSERT_LT(residual(u - 2e-4) * residual(u + 2e-4), 0.0);
}

TEST_F(ModeManagerTests, ProfileOnlyDependsOnRadiusAndHasOneComponent) {
  for (const double r : {0.7, 2.0, 4.5}) {
    for (const double angle : {0.4, 2.1, 5.0}) {
      const Position p(r * std::cos(angle), r * std::sin(angle), 3.3);
      ASSERT_NEAR(manager.get_input_component(0, p, 0), profile(r), 1e-12);
      ASSERT_EQ(manager.get_input_component(1, p, 0), 0.0);
      ASSERT_EQ(manager.get_input_component(2, p, 0), 0.0);
      // Both connectors have the same normalized frequency.
      ASSERT_EQ(manager.get_output_component(0, p, 0), manager.get_input_component(0, p, 0));
      ASSERT_EQ(manager.get_output_component(1, p, 0), 0.0);
    }
  }
}

// A guided mode in the weakly guiding approximation is continuously differentiable at the core boundary. The profile
// is J_0(r) in the core and K_0(r) in the cladding (ModeManager.cpp:61-63), without the transverse wave numbers
// u / a and w / a, and u solves the characteristic equation of the order 1.
TEST_F(ModeManagerTests, DISABLED_ProfileIsContinuouslyDifferentiableAtCoreBoundary) {
  const double u = radius_of_mode();
  const double h = 1e-4;
  const double slope_in_core = (profile(u - h) - profile(u - 3 * h)) / (2 * h);
  const double slope_in_cladding = (profile(u + 3 * h) - profile(u + h)) / (2 * h);
  ASSERT_NEAR(slope_in_core, slope_in_cladding, 1e-2);
}

// 20 x 20 cells in the cross-section and one layer of the thickness 0.05 in z.
TEST_F(RectangularModeTests, MeshCoversCrossSection) {
  RectangularMode mode;
  mode.make_mesh();
  ASSERT_EQ(mode.triangulation.n_active_cells(), 400u);
  double lowest[3] = {1e10, 1e10, 1e10}, highest[3] = {-1e10, -1e10, -1e10};
  for (const auto & cell : mode.triangulation.active_cell_iterators()) {
    for (unsigned int vertex = 0; vertex < 8; vertex++) {
      for (unsigned int i = 0; i < 3; i++) {
        lowest[i] = std::min(lowest[i], cell->vertex(vertex)[i]);
        highest[i] = std::max(highest[i], cell->vertex(vertex)[i]);
      }
    }
    ASSERT_NEAR(cell->measure(), 0.3 * 0.27 * 0.05, 1e-12);
  }
  const double expected_lowest[3] = {-3.0, -2.7, 0.0}, expected_highest[3] = {3.0, 2.7, 0.05};
  for (unsigned int i = 0; i < 3; i++) {
    ASSERT_NEAR(lowest[i], expected_lowest[i], 1e-12);
    ASSERT_NEAR(highest[i], expected_highest[i], 1e-12);
  }
  // Lowest order Nedelec elements: one degree of freedom per edge.
  ASSERT_EQ(mode.dof_handler.n_dofs(), 2u * 20u * 21u * 2u + 21u * 21u);
}

TEST_F(RectangularModeTests, DofsAreNumberedDownstream) {
  RectangularMode mode;
  mode.make_mesh();
  mode.SortDofsDownstream();
  const unsigned int n_dofs = mode.dof_handler.n_dofs();
  ASSERT_EQ(n_dofs, 2u * 20u * 21u * 2u + 21u * 21u);
  std::vector<Position> position_of_dof(n_dofs);
  std::vector<bool> found(n_dofs, false);
  std::vector<dealii::types::global_dof_index> line_dofs(1);
  for (const auto & cell : mode.dof_handler.active_cell_iterators()) {
    for (unsigned int line = 0; line < dealii::GeometryInfo<3>::lines_per_cell; line++) {
      cell->line(line)->get_dof_indices(line_dofs);
      ASSERT_LT(line_dofs[0], n_dofs);
      position_of_dof[line_dofs[0]] = cell->line(line)->center();
      found[line_dofs[0]] = true;
    }
  }
  // The new numbers are a permutation and ascend with z, then y, then x of the edge they belong to.
  ASSERT_TRUE(std::all_of(found.begin(), found.end(), [](bool value) { return value; }));
  for (unsigned int i = 1; i < n_dofs; i++) {
    ASSERT_TRUE(is_downstream_or_equal(position_of_dof[i - 1], position_of_dof[i])) << "dofs " << i - 1 << " and " << i;
  }
}

TEST_F(RectangularModeTests, SurfaceDofsBelongToRequestedBoundary) {
  RectangularMode mode;
  mode.make_mesh();
  mode.SortDofsDownstream();
  // Number of edges on the sides x = -3 (20 x 1 faces) and z = 0 (20 x 20 faces).
  const unsigned int expected_counts[2] = {20 * 2 + 21, 2 * 20 * 21};
  const BoundaryId boundaries[2] = {0, 4};
  const unsigned int axes[2] = {0, 2};
  const double coordinates[2] = {-3.0, 0.0};
  for (unsigned int i = 0; i < 2; i++) {
    const std::vector<InterfaceDofData> dofs = mode.get_surface_dof_vector_for_boundary_id(boundaries[i]);
    ASSERT_EQ(dofs.size(), expected_counts[i]);
    std::set<DofNumber> indices;
    for (const InterfaceDofData & dof : dofs) {
      ASSERT_NEAR(dof.base_point[axes[i]], coordinates[i], 1e-12);
      indices.insert(dof.index);
    }
    ASSERT_EQ(indices.size(), dofs.size());
  }
}

TEST_F(RectangularModeTests, PermittivityOfCoreAndCladding) {
  GlobalParams.Width_of_waveguide = 2.0;
  GlobalParams.Height_of_waveguide = 1.8;
  GlobalParams.Epsilon_R_in_waveguide = 2.3409;
  GlobalParams.Epsilon_R_outside_waveguide = 1.8496;
  ASSERT_EQ(RectangularMode::compute_epsilon_for_Position(Position(0, 0, 0)), 2.3409);
  ASSERT_EQ(RectangularMode::compute_epsilon_for_Position(Position(-0.99, 0.89, 0.02)), 2.3409);
  ASSERT_EQ(RectangularMode::compute_epsilon_for_Position(Position(1.01, 0, 0)), 1.8496);
  ASSERT_EQ(RectangularMode::compute_epsilon_for_Position(Position(0, -0.91, 0)), 1.8496);
  ASSERT_EQ(RectangularMode::compute_epsilon_for_Position(Position(0.95, 0.95, 0)), 1.8496);
}

// The modal computation is incomplete: n_dofs_total and the boundary surfaces are never set and the eigenvalue
// solver is commented out (RectangularMode.cpp, assemble_system() and solve()). run() would dereference the empty
// surfaces, so this test stops before it.
TEST_F(RectangularModeTests, DISABLED_RunComputesPropagationConstantOfMode) {
  RectangularMode mode;
  mode.make_mesh();
  mode.SortDofsDownstream();
  mode.make_boundary_conditions();
  for (unsigned int surface = 0; surface < 4; surface++) {
    ASSERT_NE(mode.surfaces[surface], nullptr);
  }
  ASSERT_EQ(mode.n_dofs_total, mode.dof_handler.n_dofs());
  mode.assemble_system();
  mode.solve();
  ASSERT_EQ(mode.eigenvalues.size(), 1u);
  ASSERT_NEAR(mode.beta, ExactSolution::beta(), 0.02 * ExactSolution::beta());
}
