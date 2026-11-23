// Tests for the import of the waveguide mode, the signal tapering and the shape description. Each of these tests
// documents a defect that existed in the code at some point (see MODERNIZATION.md).
#include <utility>
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/Optimization/ShapeFunction.h"
#include "../Code/Solutions/ExactSolution.h"
#include "../Code/Solutions/ExactSolutionRamped.h"
#include "../Code/SpaceTransformations/WaveguideTransformation.h"

namespace {

  const ComplexNumber imaginary_unit(0, 1);

  void prepare_mode() {
    GlobalParams = Parameters();
    Geometry.global_z_range = std::pair<double, double>(0.0, 10.0);
    ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
  }

  dealii::Tensor<1, 3, ComplexNumber> field(const ExactSolution & es, double x, double y, double z) {
    return es.val(Position(x, y, z));
  }

}

// A mode that travels in the +z direction as exp(i beta z) has to be divergence free inside the homogeneous core.
// This fails if the longitudinal component is imported with the wrong sign (i.e. for the other time convention).
TEST(ModeProfileTests, ModeIsDivergenceFreeInCore) {
  prepare_mode();
  ExactSolution es;
  const double h = 0.05;
  const double beta = ExactSolution::beta();
  double max_divergence = 0;
  double max_longitudinal_term = 0;
  for (double x = -0.7; x <= 0.71; x += 0.2) {
    for (double y = -0.6; y <= 0.61; y += 0.2) {
      const double z = 0.37;
      const ComplexNumber dEx_dx = (field(es, x + h, y, z)[0] - field(es, x - h, y, z)[0]) / (2 * h);
      const ComplexNumber dEy_dy = (field(es, x, y + h, z)[1] - field(es, x, y - h, z)[1]) / (2 * h);
      const ComplexNumber dEz_dz = imaginary_unit * beta * field(es, x, y, z)[2];
      max_divergence = std::max(max_divergence, std::abs(dEx_dx + dEy_dy + dEz_dz));
      max_longitudinal_term = std::max(max_longitudinal_term, std::abs(dEz_dz));
    }
  }
  ASSERT_GT(max_longitudinal_term, 1e-3);
  ASSERT_LT(max_divergence, 0.05 * max_longitudinal_term);
}

// The propagation constant has to match the stored profile: inside the core every component fulfills the Helmholtz
// equation (laplace_xy + k_0^2 eps_core - beta^2) E = 0.
TEST(ModeProfileTests, ProfileMatchesPropagationConstantAndCorePermittivity) {
  prepare_mode();
  ExactSolution es;
  const double h = 0.05;
  const double beta = ExactSolution::beta();
  const double k0 = 2 * GlobalParams.Pi / GlobalParams.Lambda;
  const double z = 0;
  for (double x = -0.6; x <= 0.61; x += 0.3) {
    for (double y = -0.6; y <= 0.61; y += 0.3) {
      const ComplexNumber center = field(es, x, y, z)[0];
      const ComplexNumber laplace = (field(es, x + h, y, z)[0] + field(es, x - h, y, z)[0] + field(es, x, y + h, z)[0] + field(es, x, y - h, z)[0] - 4.0 * center) / (h * h);
      const ComplexNumber residual = laplace + (k0 * k0 * GlobalParams.Epsilon_R_in_waveguide - beta * beta) * center;
      ASSERT_LT(std::abs(residual), 0.01 * beta * beta * std::abs(center));
    }
  }
}

// The stored mode belongs to a core of 2.0 x 1.8 (x times y). The longitudinal component has its maximum on the side
// walls of the core, so its location has to coincide with the width of the waveguide in the parameters.
TEST(ModeProfileTests, CoreSizeOfParametersMatchesProfile) {
  prepare_mode();
  ExactSolution es;
  double x_of_maximum = 0;
  double maximum = 0;
  for (double x = 0; x < 2.0; x += 0.01) {
    const double value = std::abs(field(es, x, 0, 0)[2]);
    if (value > maximum) {
      maximum = value;
      x_of_maximum = x;
    }
  }
  ASSERT_NEAR(x_of_maximum, GlobalParams.Width_of_waveguide / 2.0, 0.06);
  ASSERT_NEAR(GlobalParams.Height_of_waveguide, 1.8, 1e-12);
}

TEST(ModeProfileTests, CurlMatchesFiniteDifferences) {
  prepare_mode();
  ExactSolution es;
  // Central differences with the spacing of the data grid in the plane and a small step in z.
  const double h = 0.05;
  const double hz = 1e-4;
  for (double x = -0.6; x <= 0.61; x += 0.4) {
    for (double y = -0.6; y <= 0.61; y += 0.4) {
      const double z = 0.2;
      auto d = [&](unsigned int component, unsigned int direction) {
        Position plus(x, y, z), minus(x, y, z);
        const double step = (direction == 2) ? hz : h;
        plus[direction] += step;
        minus[direction] -= step;
        return (es.val(plus)[component] - es.val(minus)[component]) / (2 * step);
      };
      dealii::Tensor<1, 3, ComplexNumber> reference;
      reference[0] = d(2, 1) - d(1, 2);
      reference[1] = d(0, 2) - d(2, 0);
      reference[2] = d(1, 0) - d(0, 1);
      const dealii::Tensor<1, 3, ComplexNumber> curl = es.curl(Position(x, y, z));
      double difference = 0, norm = 0;
      for (unsigned int i = 0; i < 3; i++) {
        difference += std::norm(curl[i] - reference[i]);
        norm += std::norm(reference[i]);
      }
      ASSERT_LT(std::sqrt(difference), 0.05 * std::sqrt(norm) + 1e-3);
    }
  }
}

TEST(ModeProfileTests, DerivativeTermsAreNotZero) {
  prepare_mode();
  ExactSolution es;
  Position2D p(0.3, 0.2);
  const J_derivative_terms terms = es.get_derivative_terms(p);
  // The profile is a smooth bump, so its second derivatives in the core are negative and clearly non-zero.
  ASSERT_LT(terms.d_f_dyy.real(), -0.1);
  ASSERT_NEAR(terms.beta, ExactSolution::beta(), 1e-12);
  // h = E_z = -i * (stored profile): purely imaginary.
  ASSERT_NEAR(terms.h.real(), 0.0, 1e-12);
}

TEST(SignalTaperTests, DerivativeOfRampScalesWithTaperLength) {
  prepare_mode();
  GlobalParams.Signal_tapering_type = SignalTaperingType::C1;
  GlobalParams.tapering_min_z = 1.0;
  GlobalParams.tapering_max_z = 3.0;
  ExactSolutionRamped ramped;
  const double h = 1e-5;
  for (double z = 1.2; z < 3.0; z += 0.4) {
    const double fd = (ramped.get_ramping_factor_for_position(Position(0, 0, z + h)) - ramped.get_ramping_factor_for_position(Position(0, 0, z - h))) / (2 * h);
    ASSERT_NEAR(ramped.get_ramping_factor_derivative_for_position(Position(0, 0, z)), fd, 1e-6);
  }
}

// A straight waveguide must be described by the identity transformation - reproducibly. (The shape function of the
// vertical displacement used to be initialized with random numbers in this case.)
TEST(ShapeDescriptionTests, StraightWaveguideIsDeterministicAndFlat) {
  for (unsigned int repetition = 0; repetition < 2; repetition++) {
    ShapeFunction shape(0.0, 10.0, 5, false);
    shape.set_constraints(0, 0, 0, 0);
    shape.initialize();
    for (double z = 0; z <= 10.0; z += 0.25) {
      ASSERT_NEAR(shape.evaluate_at(z), 0.0, 1e-14);
      ASSERT_NEAR(shape.evaluate_derivative_at(z), 0.0, 1e-14);
    }
  }
}

TEST(ShapeDescriptionTests, DerivativeIsConsistentWithValues) {
  ShapeFunction shape(0.0, 10.0, 5, false);
  shape.set_constraints(0, 1.5, 0, 0);
  shape.initialize();
  shape.set_free_dof_value(0, 0.2);
  shape.set_free_dof_value(1, 0.45);
  const double h = 1e-5;
  for (double z = 0.33; z < 4.9; z += 0.41) {
    const double fd = (shape.evaluate_at(z + h) - shape.evaluate_at(z - h)) / (2 * h);
    ASSERT_NEAR(shape.evaluate_derivative_at(z), fd, 1e-5);
  }
  // The constrained degree of freedom has to make the function reach the prescribed end value.
  ASSERT_NEAR(shape.evaluate_at(5.0), 1.5, 1e-10);
}

TEST(ShapeDescriptionTests, TransformationOfStraightWaveguideIsIdentity) {
  GlobalParams = Parameters();
  GlobalParams.Number_of_sectors = 5;
  GlobalParams.Vertical_displacement_of_waveguide = 0;
  Geometry.global_z_range = std::pair<double, double>(0.0, 10.0);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  for (double z = 0.1; z < 10.0; z += 1.3) {
    Position p(0.3, -0.2, z);
    const dealii::Tensor<2, 3, double> tensor = transformation.get_Space_Transformation_Tensor(p);
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        ASSERT_NEAR(tensor[i][j], (i == j) ? 1.0 : 0.0, 1e-14);
      }
    }
  }
}

TEST(ShapeDescriptionTests, JacobianMatchesFiniteDifferencesOfTransformation) {
  GlobalParams = Parameters();
  GlobalParams.Number_of_sectors = 5;
  GlobalParams.Vertical_displacement_of_waveguide = 1.0;
  Geometry.global_z_range = std::pair<double, double>(0.0, 10.0);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  const double h = 1e-6;
  for (double z = 0.37; z < 4.9; z += 0.61) {
    Position p(0.3, -0.2, z);
    const dealii::Tensor<2, 3, double> J = transformation.get_J(p);
    for (unsigned int j = 0; j < 3; j++) {
      Position plus = p, minus = p;
      plus[j] += h;
      minus[j] -= h;
      const dealii::Tensor<1, 3, double> d = (transformation.phys_to_math(plus) - transformation.phys_to_math(minus)) / (2 * h);
      for (unsigned int i = 0; i < 3; i++) {
        ASSERT_NEAR(J[i][j], d[i], 1e-5);
      }
    }
  }
}

// Ranks are numbered x-first. The neighbor in y is therefore (number of processes in x) ranks away.
TEST(ProcessLayoutTests, NeighborsInNonCubicLayout) {
  GlobalParams = Parameters();
  GlobalParams.Blocks_in_x_direction = 2;
  GlobalParams.Blocks_in_y_direction = 3;
  GlobalParams.Blocks_in_z_direction = 2;
  GlobalParams.MPI_Rank = 3; // (x, y, z) = (1, 1, 0)
  GlobalParams.Index_in_x_direction = 1;
  GlobalParams.Index_in_y_direction = 1;
  GlobalParams.Index_in_z_direction = 0;
  MPICommunicator communicator;
  ASSERT_EQ(communicator.get_neighbor_for_interface(Direction::MinusX).second, 2u);
  ASSERT_EQ(communicator.get_neighbor_for_interface(Direction::MinusY).second, 1u);
  ASSERT_EQ(communicator.get_neighbor_for_interface(Direction::PlusY).second, 5u);
  ASSERT_EQ(communicator.get_neighbor_for_interface(Direction::PlusZ).second, 9u);
  ASSERT_FALSE(communicator.get_neighbor_for_interface(Direction::PlusX).first);
  ASSERT_FALSE(communicator.get_neighbor_for_interface(Direction::MinusZ).first);
}
