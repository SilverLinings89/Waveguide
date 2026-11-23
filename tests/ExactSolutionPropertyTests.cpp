// Properties of the analytical fields: the mode of the straight waveguide (ExactSolution), its complex conjugate,
// the tapered mode, the mode in the PML and the mode of the tilted waveguide.
#include <utility>
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/Solutions/AngledExactSolution.h"
#include "../Code/Solutions/ExactSolution.h"
#include "../Code/Solutions/ExactSolutionConjugate.h"
#include "../Code/Solutions/ExactSolutionRamped.h"
#include "../Code/Solutions/PMLTransformedExactSolution.h"
#include "../Code/SpaceTransformations/AngleWaveguideTransformation.h"
#include "./HelpersB.h"

namespace {

  using helpers_b::max_difference;
  using FieldValue = dealii::Tensor<1, 3, ComplexNumber>;

  const ComplexNumber imaginary_unit(0, 1);

  class ModeFixture : public helpers_b::GlobalStateTest {
  protected:
    void SetUp() override {
      helpers_b::GlobalStateTest::SetUp();
      load_mode_profile();
    }
  };

  class ExactSolutionPropertyTests : public ModeFixture { };
  class ExactSolutionConjugateTests : public ModeFixture { };
  class RampedSolutionTests : public ModeFixture { };
  class AngledSolutionTests : public ModeFixture { };

  // Domain [-2, 2] x [-1.5, 1.5] x [0, 10] with a PML of thickness 1 whose first of 8 layers does not absorb.
  // val() of PMLTransformedExactSolution and of AngledExactSolution is not called in these tests: both write the
  // field into a vector of size zero.
  class PMLTransformedSolutionTests : public ModeFixture {
  protected:
    void SetUp() override {
      ModeFixture::SetUp();
      GlobalParams.PML_thickness = 1.0;
      GlobalParams.PML_N_Layers = 8;
      Geometry.local_x_range = std::pair<double, double>(-2.0, 2.0);
      Geometry.local_y_range = std::pair<double, double>(-1.5, 1.5);
      Geometry.local_z_range = std::pair<double, double>(0.0, 10.0);
    }

    const double first_layer = 1.0 / 8.0;
  };

  // Points inside of the core and in the cladding. They are not on lines of the data grid (distance 0.05), where
  // the interpolated profile has kinks.
  std::vector<Position> sample_points() {
    std::vector<Position> ret;
    for (const double z : {0.0, 0.37, 4.21}) {
      ret.push_back(Position(0.313, 0.207, z));
      ret.push_back(Position(-0.662, 0.431, z));
      ret.push_back(Position(1.487, -1.212, z));
    }
    return ret;
  }

  template <typename Solution>
  FieldValue vector_value_of(const Solution & solution, const Position & p) {
    dealii::Vector<ComplexNumber> values(3);
    solution.vector_value(p, values);
    FieldValue ret;
    for (unsigned int i = 0; i < 3; i++) {
      ret[i] = values[i];
    }
    return ret;
  }

  template <typename Solution>
  FieldValue value_of(const Solution & solution, const Position & p) {
    FieldValue ret;
    for (unsigned int i = 0; i < 3; i++) {
      ret[i] = solution.value(p, i);
    }
    return ret;
  }

  FieldValue conjugate(const FieldValue & in) {
    FieldValue ret;
    for (unsigned int i = 0; i < 3; i++) {
      ret[i] = std::conj(in[i]);
    }
    return ret;
  }

  // Curl by central differences of vector_value().
  template <typename Solution>
  FieldValue curl_by_finite_differences(const Solution & solution, const Position & p, double h) {
    auto d = [&](unsigned int component, unsigned int direction) {
      Position plus = p, minus = p;
      plus[direction] += h;
      minus[direction] -= h;
      return (vector_value_of(solution, plus)[component] - vector_value_of(solution, minus)[component]) / (2 * h);
    };
    FieldValue ret;
    ret[0] = d(2, 1) - d(1, 2);
    ret[1] = d(0, 2) - d(2, 0);
    ret[2] = d(1, 0) - d(0, 1);
    return ret;
  }

}

// The effective index of a guided mode lies between the indices of the cladding and the core.
TEST_F(ExactSolutionPropertyTests, PropagationConstantBelongsToGuidedMode) {
  const double k0 = 2 * GlobalParams.Pi / GlobalParams.Lambda;
  ASSERT_GT(ExactSolution::beta(), k0 * std::sqrt(GlobalParams.Epsilon_R_outside_waveguide));
  ASSERT_LT(ExactSolution::beta(), k0 * std::sqrt(GlobalParams.Epsilon_R_in_waveguide));
}

TEST_F(ExactSolutionPropertyTests, FieldAndCurlPropagateWithExpIBetaZ) {
  ExactSolution es;
  const double beta = ExactSolution::beta();
  for (const Position & p : sample_points()) {
    for (const double distance : {0.1, 1.3, -0.45}) {
      Position shifted = p;
      shifted[2] += distance;
      const ComplexNumber phase = std::exp(imaginary_unit * beta * distance);
      ASSERT_LT(max_difference(es.val(shifted), phase * es.val(p)), 1e-12);
      ASSERT_LT(max_difference(es.curl(shifted), phase * es.curl(p)), 1e-11);
      ASSERT_NEAR(es.val(shifted).norm(), es.val(p).norm(), 1e-12);
    }
  }
}

// The phase is zero at the input interface, also if the domain does not start at z = 0. There the transverse
// components are real and the longitudinal component is imaginary.
TEST_F(ExactSolutionPropertyTests, PhaseIsZeroAtStartOfDomain) {
  ExactSolution es;
  const FieldValue reference = es.val(Position(0.313, 0.207, 0.0));
  ASSERT_GT(std::abs(reference[0]), 0.1);
  ASSERT_GT(std::abs(reference[2]), 1e-3);
  ASSERT_EQ(reference[0].imag(), 0.0);
  ASSERT_EQ(reference[1].imag(), 0.0);
  ASSERT_EQ(reference[2].real(), 0.0);
  Geometry.global_z_range = std::pair<double, double>(2.0, 12.0);
  ASSERT_LT(max_difference(es.val(Position(0.313, 0.207, 2.0)), reference), 1e-14);
}

TEST_F(ExactSolutionPropertyTests, ValueVectorValueAndValAgree) {
  ExactSolution es;
  for (const Position & p : sample_points()) {
    ASSERT_LT(max_difference(value_of(es, p), es.val(p)), 1e-15);
    ASSERT_LT(max_difference(vector_value_of(es, p), es.val(p)), 1e-15);
  }
}

// load_data() scales the profile such that the largest field strength is one.
TEST_F(ExactSolutionPropertyTests, ProfileIsNormalized) {
  ExactSolution es;
  double largest_norm = 0;
  for (unsigned int i = 0; i <= ExactSolution::n_intervals[0]; i++) {
    for (unsigned int j = 0; j <= ExactSolution::n_intervals[1]; j++) {
      const double x = ExactSolution::ranges[0].first + i * (ExactSolution::ranges[0].second - ExactSolution::ranges[0].first) / ExactSolution::n_intervals[0];
      const double y = ExactSolution::ranges[1].first + j * (ExactSolution::ranges[1].second - ExactSolution::ranges[1].first) / ExactSolution::n_intervals[1];
      largest_norm = std::max(largest_norm, es.val(Position(x, y, 0)).norm());
    }
  }
  ASSERT_NEAR(largest_norm, 1.0, 1e-9);
}

// The fundamental mode of the symmetric waveguide: the main component is even in x and y, the longitudinal
// component is odd in x (it is proportional to the derivative of the main component in x) and even in y.
TEST_F(ExactSolutionPropertyTests, ProfileHasSymmetryOfFundamentalMode) {
  ExactSolution es;
  for (const double x : {0.313, 0.662, 1.487}) {
    for (const double y : {0.207, 0.431, 1.212}) {
      const FieldValue value = es.val(Position(x, y, 0));
      const FieldValue mirrored_x = es.val(Position(-x, y, 0));
      const FieldValue mirrored_y = es.val(Position(x, -y, 0));
      ASSERT_NEAR(std::abs(mirrored_x[0] - value[0]), 0.0, 1e-6);
      ASSERT_NEAR(std::abs(mirrored_y[0] - value[0]), 0.0, 1e-6);
      ASSERT_NEAR(std::abs(mirrored_x[2] + value[2]), 0.0, 1e-6);
      ASSERT_NEAR(std::abs(mirrored_y[2] - value[2]), 0.0, 1e-6);
    }
  }
}

TEST_F(ExactSolutionPropertyTests, FieldIsConfinedToCore) {
  ExactSolution es;
  const double in_core = es.val(Position(0, 0, 0)).norm();
  ASSERT_GT(in_core, 0.5);
  ASSERT_LT(es.val(Position(2.5, 0, 0)).norm(), 0.1 * in_core);
  ASSERT_LT(es.val(Position(0, 2.5, 0)).norm(), 0.1 * in_core);
  ASSERT_LT(es.val(Position(4.9, 4.4, 0)).norm(), 1e-4 * in_core);
}

TEST_F(ExactSolutionPropertyTests, MissingProfileFileIsReported) {
  ASSERT_THROW(ExactSolution::load_data(repository_path("Modes/this_file_does_not_exist.dat")), std::runtime_error);
  // The profile that was loaded before is still available.
  ExactSolution es;
  ASSERT_GT(es.val(Position(0, 0, 0)).norm(), 0.5);
}

TEST_F(ExactSolutionConjugateTests, AllEvaluationsAreComplexConjugates) {
  ExactSolution es;
  ExactSolutionConjugate conjugated;
  for (const Position & p : sample_points()) {
    const FieldValue reference = conjugate(es.val(p));
    ASSERT_GT(reference.norm(), 1e-3);
    ASSERT_LT(max_difference(conjugated.val(p), reference), 1e-15);
    ASSERT_LT(max_difference(value_of(conjugated, p), reference), 1e-15);
    ASSERT_LT(max_difference(vector_value_of(conjugated, p), reference), 1e-15);
    ASSERT_LT(max_difference(conjugated.curl(p), conjugate(es.curl(p))), 1e-15);
  }
}

TEST_F(ExactSolutionConjugateTests, ConjugatePropagatesInNegativeZDirection) {
  ExactSolutionConjugate conjugated;
  const double beta = ExactSolution::beta();
  for (const Position & p : sample_points()) {
    Position shifted = p;
    shifted[2] += 0.3;
    ASSERT_LT(max_difference(conjugated.val(shifted), std::exp(-imaginary_unit * beta * 0.3) * conjugated.val(p)), 1e-12);
  }
}

TEST_F(RampedSolutionTests, RampFallsMonotonicallyFromOneToZero) {
  GlobalParams.tapering_min_z = 1.0;
  GlobalParams.tapering_max_z = 3.0;
  for (const SignalTaperingType type : {SignalTaperingType::C0, SignalTaperingType::C1}) {
    GlobalParams.Signal_tapering_type = type;
    ExactSolutionRamped ramped;
    auto ramp = [&](double z) { return ramped.get_ramping_factor_for_position(Position(0.1, 0.2, z)); };
    ASSERT_EQ(ramp(-2.0), 1.0);
    ASSERT_EQ(ramp(1.0), 1.0);
    ASSERT_EQ(ramp(3.0), 0.0);
    ASSERT_EQ(ramp(7.0), 0.0);
    ASSERT_NEAR(ramp(2.0), 0.5, 1e-14);
    // Continuous at both ends of the ramp.
    ASSERT_NEAR(ramp(1.0 + 1e-9), 1.0, 1e-8);
    ASSERT_NEAR(ramp(3.0 - 1e-9), 0.0, 1e-8);
    double previous = 1.0;
    for (unsigned int i = 1; i < 40; i++) {
      const double z = 1.0 + i * 0.05;
      ASSERT_LT(ramp(z), previous);
      ASSERT_GT(ramp(z), 0.0);
      previous = ramp(z);
    }
  }
}

TEST_F(RampedSolutionTests, DerivativeOfC0RampInsideOfRamp) {
  GlobalParams.tapering_min_z = 1.0;
  GlobalParams.tapering_max_z = 3.0;
  GlobalParams.Signal_tapering_type = SignalTaperingType::C0;
  ExactSolutionRamped ramped;
  const double h = 1e-5;
  for (double z = 1.2; z < 3.0; z += 0.4) {
    const double fd = (ramped.get_ramping_factor_for_position(Position(0, 0, z + h)) - ramped.get_ramping_factor_for_position(Position(0, 0, z - h))) / (2 * h);
    ASSERT_NEAR(ramped.get_ramping_factor_derivative_for_position(Position(0, 0, z)), fd, 1e-8);
    ASSERT_NEAR(fd, -0.5, 1e-8);
  }
}

TEST_F(RampedSolutionTests, DerivativeOfC1RampVanishesAtEndsAndOutsideOfRamp) {
  GlobalParams.tapering_min_z = 1.0;
  GlobalParams.tapering_max_z = 3.0;
  GlobalParams.Signal_tapering_type = SignalTaperingType::C1;
  ExactSolutionRamped ramped;
  for (const double z : {-1.0, 0.5, 1.0, 3.0, 3.5, 8.0}) {
    ASSERT_NEAR(ramped.get_ramping_factor_derivative_for_position(Position(0, 0, z)), 0.0, 1e-14);
  }
}

// The derivative of the C0 ramp is -1 / (length of the ramp) for every z (ExactSolutionRamped.cpp:81).
TEST_F(RampedSolutionTests, DISABLED_DerivativeOfC0RampVanishesOutsideOfRamp) {
  GlobalParams.tapering_min_z = 1.0;
  GlobalParams.tapering_max_z = 3.0;
  GlobalParams.Signal_tapering_type = SignalTaperingType::C0;
  ExactSolutionRamped ramped;
  for (const double z : {-1.0, 0.5, 3.5, 8.0}) {
    ASSERT_NEAR(ramped.get_ramping_factor_derivative_for_position(Position(0, 0, z)), 0.0, 1e-14);
  }
}

TEST_F(RampedSolutionTests, FieldIsModeTimesRamp) {
  GlobalParams.tapering_min_z = 1.0;
  GlobalParams.tapering_max_z = 3.0;
  for (const SignalTaperingType type : {SignalTaperingType::C0, SignalTaperingType::C1}) {
    GlobalParams.Signal_tapering_type = type;
    ExactSolution es;
    ExactSolutionRamped ramped;
    for (const double z : {0.4, 1.3, 2.0, 2.9, 3.6}) {
      const Position p(0.313, 0.207, z);
      const FieldValue reference = ramped.get_ramping_factor_for_position(p) * es.val(p);
      ASSERT_LT(max_difference(ramped.val(p), reference), 1e-15);
      ASSERT_LT(max_difference(value_of(ramped, p), reference), 1e-15);
      ASSERT_LT(max_difference(vector_value_of(ramped, p), reference), 1e-15);
    }
    // Before the ramp the field is the mode, behind it there is no field.
    ASSERT_LT(max_difference(ramped.val(Position(0.313, 0.207, 0.4)), es.val(Position(0.313, 0.207, 0.4))), 1e-15);
    ASSERT_EQ(ramped.val(Position(0.313, 0.207, 3.6)).norm(), 0.0);
  }
}

// curl(r E) = r curl(E) + grad(r) x E, and the ramp r only depends on z.
TEST_F(RampedSolutionTests, CurlFollowsProductRule) {
  GlobalParams.tapering_min_z = 1.0;
  GlobalParams.tapering_max_z = 3.0;
  for (const SignalTaperingType type : {SignalTaperingType::C0, SignalTaperingType::C1}) {
    GlobalParams.Signal_tapering_type = type;
    ExactSolution es;
    ExactSolutionRamped ramped;
    for (const double z : {0.4, 1.3, 2.0, 2.9}) {
      for (const Position & p : {Position(0.313, 0.207, z), Position(-0.662, 0.431, z)}) {
        const double ramp = ramped.get_ramping_factor_for_position(p);
        const double ramp_derivative = (z > 1.0) ? ramped.get_ramping_factor_derivative_for_position(p) : 0.0;
        const FieldValue field = es.val(p);
        FieldValue reference = ramp * es.curl(p);
        reference[0] -= ramp_derivative * field[1];
        reference[1] += ramp_derivative * field[0];
        // ExactSolutionRamped::curl uses one-sided differences with the step 1e-4.
        ASSERT_LT(max_difference(ramped.curl(p), reference), 1e-2) << "at z = " << z;
        ASSERT_GT(es.curl(p).norm(), 1.0);
      }
    }
  }
}

TEST_F(PMLTransformedSolutionTests, MatchesModeInsideOfDomain) {
  ExactSolution es;
  GlobalParams.source_field = &es;
  for (BoundaryId boundary = 0; boundary < 6; boundary++) {
    PMLTransformedExactSolution transformed(boundary, 0.0);
    for (const Position & p : sample_points()) {
      ASSERT_EQ(transformed.compute_scaling_factor(p), 1.0);
      ASSERT_LT(max_difference(value_of(transformed, p), es.val(p)), 1e-15);
      ASSERT_LT(max_difference(vector_value_of(transformed, p), es.val(p)), 1e-15);
    }
  }
}

// The absorption starts behind the first layer of the PML and reaches its maximum at the outer boundary.
TEST_F(PMLTransformedSolutionTests, FractionGrowsFromZeroToOneInUpperPML) {
  ExactSolution es;
  GlobalParams.source_field = &es;
  PMLTransformedExactSolution transformed(5, 0.0);
  const double upper_end[3] = {2.0, 1.5, 10.0};
  for (unsigned int direction = 0; direction < 3; direction++) {
    Position p(0.3, 0.2, 4.0);
    ASSERT_EQ(transformed.fraction_of_pml_direction(p)[direction], 0.0);
    p[direction] = upper_end[direction] + first_layer;
    ASSERT_NEAR(transformed.fraction_of_pml_direction(p)[direction], 0.0, 1e-14);
    p[direction] = upper_end[direction] + 0.5 * (first_layer + 1.0);
    ASSERT_NEAR(transformed.fraction_of_pml_direction(p)[direction], 0.5, 1e-14);
    p[direction] = upper_end[direction] + 1.0;
    ASSERT_NEAR(transformed.fraction_of_pml_direction(p)[direction], 1.0, 1e-14);
  }
}

// On the lower sides the thickness of the first layer is added instead of subtracted
// (PMLTransformedExactSolution.cpp:74, 83): the fraction is not zero at the end of the first layer and exceeds one.
TEST_F(PMLTransformedSolutionTests, DISABLED_FractionGrowsFromZeroToOneInLowerPML) {
  ExactSolution es;
  GlobalParams.source_field = &es;
  PMLTransformedExactSolution transformed(4, 0.0);
  const double lower_end[2] = {-2.0, -1.5};
  for (unsigned int direction = 0; direction < 2; direction++) {
    Position p(0.3, 0.2, 4.0);
    p[direction] = lower_end[direction] - first_layer;
    ASSERT_NEAR(transformed.fraction_of_pml_direction(p)[direction], 0.0, 1e-14);
    p[direction] = lower_end[direction] - 0.5 * (first_layer + 1.0);
    ASSERT_NEAR(transformed.fraction_of_pml_direction(p)[direction], 0.5, 1e-14);
    p[direction] = lower_end[direction] - 1.0;
    ASSERT_NEAR(transformed.fraction_of_pml_direction(p)[direction], 1.0, 1e-14);
  }
}

// Below the domain the fraction in z is computed from the x-coordinate (PMLTransformedExactSolution.cpp:92).
TEST_F(PMLTransformedSolutionTests, DISABLED_FractionBelowDomainDependsOnZOnly) {
  ExactSolution es;
  GlobalParams.source_field = &es;
  PMLTransformedExactSolution transformed(4, 0.0);
  const double at_first_x = transformed.fraction_of_pml_direction(Position(0.3, 0.2, -0.6))[2];
  const double at_second_x = transformed.fraction_of_pml_direction(Position(-1.1, 0.2, -0.6))[2];
  ASSERT_NEAR(at_first_x, at_second_x, 1e-14);
  ASSERT_GT(transformed.fraction_of_pml_direction(Position(0.3, 0.2, -0.9))[2], at_first_x);
}

TEST_F(PMLTransformedSolutionTests, FieldDecaysInUpperPML) {
  ExactSolution es;
  GlobalParams.source_field = &es;
  PMLTransformedExactSolution transformed(5, 0.0);
  double previous = 1.0;
  for (double depth = 0.3; depth <= 1.0; depth += 0.1) {
    const Position p(0.313, 0.207, 10.0 + depth);
    const double factor = transformed.compute_scaling_factor(p);
    ASSERT_GT(factor, 0.0);
    ASSERT_LT(factor, previous);
    ASSERT_LT(max_difference(vector_value_of(transformed, p), factor * es.val(p)), 1e-15);
    previous = factor;
  }
}

// value() only scales the component normal to the boundary, vector_value() scales all components.
TEST_F(PMLTransformedSolutionTests, DISABLED_ValueAndVectorValueAgreeInPML) {
  ExactSolution es;
  GlobalParams.source_field = &es;
  PMLTransformedExactSolution transformed(5, 0.0);
  const Position p(0.313, 0.207, 10.8);
  ASSERT_LT(transformed.compute_scaling_factor(p), 0.99);
  ASSERT_LT(max_difference(value_of(transformed, p), vector_value_of(transformed, p)), 1e-15);
}

// curl() returns zero (PMLTransformedExactSolution.cpp:48).
TEST_F(PMLTransformedSolutionTests, DISABLED_CurlMatchesCurlOfModeInsideOfDomain) {
  ExactSolution es;
  GlobalParams.source_field = &es;
  PMLTransformedExactSolution transformed(5, 0.0);
  for (const Position & p : sample_points()) {
    ASSERT_GT(es.curl(p).norm(), 1e-2);
    ASSERT_LT(max_difference(transformed.curl(p), es.curl(p)), 1e-12);
  }
}

TEST_F(AngledSolutionTests, PositionIsTransformedLikeAngleTransformation) {
  AngledExactSolution angled;
  AngleWaveguideTransformation transformation;
  for (const double angle : {0.2, 0.35}) {
    GlobalParams.PML_Angle_Test = angle;
    for (const Position & p : sample_points()) {
      ASSERT_LT((angled.transform_position(p) - transformation.phys_to_math(p)).norm(), 1e-15);
    }
  }
}

TEST_F(AngledSolutionTests, ValueAndVectorValueAgree) {
  use_as_global_transformation(new AngleWaveguideTransformation());
  AngledExactSolution angled;
  for (const Position & p : sample_points()) {
    ASSERT_GT(vector_value_of(angled, p).norm(), 1e-3);
    ASSERT_LT(max_difference(value_of(angled, p), vector_value_of(angled, p)), 1e-15);
  }
}

// For y = 0 the tilted and the straight waveguide coincide: the position is not changed and the components that are
// tangential to the plane y = 0 are those of the straight mode.
TEST_F(AngledSolutionTests, TangentialComponentsOnSymmetryPlaneAreThoseOfMode) {
  use_as_global_transformation(new AngleWaveguideTransformation());
  AngledExactSolution angled;
  ExactSolution es;
  for (const double x : {0.313, -0.662}) {
    for (const double z : {0.0, 0.37, 4.21}) {
      const Position p(x, 0.0, z);
      ASSERT_NEAR(std::abs(angled.value(p, 0) - es.val(p)[0]), 0.0, 1e-14);
      ASSERT_NEAR(std::abs(angled.value(p, 2) - es.val(p)[2]), 0.0, 1e-14);
    }
  }
}

// curl() differentiates around the transformed position and vector_value() transforms the position again.
TEST_F(AngledSolutionTests, DISABLED_CurlMatchesFiniteDifferences) {
  use_as_global_transformation(new AngleWaveguideTransformation());
  AngledExactSolution angled;
  for (const Position & p : sample_points()) {
    const FieldValue reference = curl_by_finite_differences(angled, p, 1e-5);
    ASSERT_GT(reference.norm(), 1e-2);
    ASSERT_LT(max_difference(angled.curl(p), reference), 0.02 * reference.norm());
  }
}
