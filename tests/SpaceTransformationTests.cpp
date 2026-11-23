// Tests for the coordinate transformations between the mathematical system (straight waveguide) and the physical
// system (shaped waveguide). The properties tested here do not depend on the shape: the two maps are inverse to each
// other, the Jacobian is the derivative of phys_to_math, and the material tensor J J^T / det(J) is symmetric with
// determinant 1 / det(J).
#include <utility>
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/Helpers/ParameterOverride.h"
#include "../Code/Helpers/ShapeDescription.h"
#include "../Code/SpaceTransformations/AngleWaveguideTransformation.h"
#include "../Code/SpaceTransformations/BendTransformation.h"
#include "../Code/SpaceTransformations/PredefinedShapeTransformation.h"
#include "../Code/SpaceTransformations/WaveguideTransformation.h"
#include "./HelpersB.h"

namespace {

  using helpers_b::identity_tensor;
  using helpers_b::jacobian_by_finite_differences;
  using helpers_b::max_difference;

  class WaveguideTransformationTests : public helpers_b::GlobalStateTest { };
  class AngleTransformationTests : public helpers_b::GlobalStateTest { };
  class BendTransformationTests : public helpers_b::GlobalStateTest { };
  class PredefinedShapeTests : public helpers_b::GlobalStateTest { };
  class SpaceTransformationTests : public helpers_b::GlobalStateTest { };

  const unsigned int n_sectors = 5;

  // Points in both halves of the domain [0, 10] in z, away from the nodes of the shape functions.
  std::vector<Position> sample_points() {
    std::vector<Position> ret;
    for (double z = 0.37; z < 10.0; z += 0.83) {
      ret.push_back(Position(0.3, -0.2, z));
      ret.push_back(Position(-1.1, 0.7, z));
    }
    return ret;
  }

  void configure_waveguide(double displacement, bool constant_height, bool constant_width) {
    GlobalParams.Number_of_sectors = n_sectors;
    GlobalParams.Vertical_displacement_of_waveguide = displacement;
    GlobalParams.keep_waveguide_height_constant = constant_height;
    GlobalParams.keep_waveguide_width_constant = constant_width;
  }

  // Gives the height and the width (if they are variable) a profile that is not constant.
  void perturb_height_and_width(WaveguideTransformation & transformation) {
    const unsigned int n_free_per_function = n_sectors - 2;
    const double slopes[3] = {0.05, -0.02, 0.03};
    for (unsigned int i = n_free_per_function; i < transformation.n_free_dofs(); i++) {
      transformation.set_free_dof(i, slopes[i % 3] * ((i < 2 * n_free_per_function) ? 1.0 : -0.7));
    }
  }

  void expect_maps_are_inverse(const SpaceTransformation & transformation, const std::vector<Position> & points) {
    for (const Position & p : points) {
      ASSERT_LT((transformation.math_to_phys(transformation.phys_to_math(p)) - p).norm(), 1e-12);
      ASSERT_LT((transformation.phys_to_math(transformation.math_to_phys(p)) - p).norm(), 1e-12);
    }
  }

  void expect_jacobian_matches_finite_differences(SpaceTransformation & transformation, const std::vector<Position> & points, double tolerance) {
    for (Position p : points) {
      ASSERT_LT(max_difference(transformation.get_J(p), jacobian_by_finite_differences(transformation, p)), tolerance) << "at z = " << p[2];
      ASSERT_LT(max_difference(transformation.get_J(p) * transformation.get_J_inverse(p), identity_tensor()), 1e-12);
    }
  }

  void expect_tensor_is_symmetric_with_inverse_determinant(SpaceTransformation & transformation, const std::vector<Position> & points) {
    for (Position p : points) {
      const dealii::Tensor<2, 3, double> tensor = transformation.get_Space_Transformation_Tensor(p);
      ASSERT_LT(max_difference(tensor, dealii::transpose(tensor)), 1e-13);
      ASSERT_NEAR(dealii::determinant(tensor) * dealii::determinant(transformation.get_J(p)), 1.0, 1e-12);
      const dealii::Tensor<2, 3, ComplexNumber> complex_tensor = transformation.get_Tensor(p);
      const dealii::Tensor<2, 3, ComplexNumber> real_tensor_as_complex = tensor;
      ASSERT_LT(max_difference(complex_tensor, real_tensor_as_complex), 1e-14);
    }
  }

  double vertical_displacement_at(const SpaceTransformation & transformation, double z) {
    return transformation.math_to_phys(Position(0, 0, z))[1];
  }

  // Shape with the displacement m(z) = amplitude * sin^2(pi z / length) on equidistant nodes. The displacement and
  // its derivative are zero at both ends.
  void set_sine_shape(unsigned int n_nodes, double node_distance, double amplitude) {
    const double length = (n_nodes - 1) * node_distance;
    helpers_b::reset_parameters_and_ranges(length);
    const double pi = GlobalParams.Pi;
    GlobalParams.sd.Sectors = n_nodes;
    for (unsigned int i = 0; i < n_nodes; i++) {
      const double z = i * node_distance;
      GlobalParams.sd.z.push_back(z);
      GlobalParams.sd.m.push_back(amplitude * std::pow(std::sin(pi * z / length), 2));
      GlobalParams.sd.v.push_back(amplitude * pi / length * std::sin(2 * pi * z / length));
    }
    GlobalParams.Sector_thickness = node_distance;
    GlobalParams.Number_of_sectors = n_nodes - 1;
    GlobalParams.Use_Predefined_Shape = true;
  }

  std::vector<Position> points_between_nodes(double length) {
    std::vector<Position> ret;
    for (double z = 0.31; z < length; z += 0.77) {
      ret.push_back(Position(0.3, -0.2, z));
    }
    return ret;
  }

  std::vector<std::string> lines_of_shape_file() {
    std::vector<std::string> ret;
    std::ifstream input(repository_path("Modes/test.csv"));
    std::string line;
    while (std::getline(input, line)) {
      if (line.size() > 0) ret.push_back(line);
    }
    return ret;
  }

  // The path of the solver: Parameters::complete_data() reads the shape from Modes/test.csv.
  void load_predefined_shape(unsigned int shape_number) {
    helpers_b::reset_parameters_and_ranges(100.0);
    GlobalParams.Use_Predefined_Shape = true;
    GlobalParams.Number_of_Predefined_Shape = shape_number;
    GlobalParams.Point_Source_Type = 99; // no source field is required here
    GlobalParams.complete_data();
  }

}

TEST_F(WaveguideTransformationTests, MapsAreInverseForVerticalShift) {
  configure_waveguide(1.0, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  expect_maps_are_inverse(transformation, sample_points());
  // The shape is not trivial: in the middle of the domain the waveguide is displaced.
  ASSERT_GT(std::abs(vertical_displacement_at(transformation, 4.0)), 0.1);
}

TEST_F(WaveguideTransformationTests, MapsAreInverseForVariableHeightAndWidth) {
  configure_waveguide(0.6, false, false);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  perturb_height_and_width(transformation);
  expect_maps_are_inverse(transformation, sample_points());
  // Height and width are really variable: a point off the axis is moved in x.
  ASSERT_GT(std::abs(transformation.math_to_phys(Position(1.0, 0.0, 2.4))[0] - 1.0), 1e-3);
}

TEST_F(WaveguideTransformationTests, JacobianMatchesFiniteDifferencesForVariableHeight) {
  configure_waveguide(0.6, false, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  perturb_height_and_width(transformation);
  expect_jacobian_matches_finite_differences(transformation, sample_points(), 1e-5);
}

TEST_F(WaveguideTransformationTests, JacobianMatchesFiniteDifferencesForVariableHeightAndWidth) {
  configure_waveguide(0.6, false, false);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  perturb_height_and_width(transformation);
  expect_jacobian_matches_finite_differences(transformation, sample_points(), 1e-5);
}

// get_J() divides by the height function (WaveguideTransformation.cpp:229), which is not initialized for a
// constant height (it is zero).
TEST_F(WaveguideTransformationTests, DISABLED_JacobianMatchesFiniteDifferencesForConstantHeightAndVariableWidth) {
  configure_waveguide(0.6, true, false);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  expect_jacobian_matches_finite_differences(transformation, sample_points(), 1e-5);
}

TEST_F(WaveguideTransformationTests, TensorIsSymmetricWithDeterminantOne) {
  // A pure shift has det(J) = 1.
  configure_waveguide(1.0, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  for (Position p : sample_points()) {
    ASSERT_NEAR(dealii::determinant(transformation.get_J(p)), 1.0, 1e-14);
  }
  expect_tensor_is_symmetric_with_inverse_determinant(transformation, sample_points());
}

TEST_F(WaveguideTransformationTests, TensorIsSymmetricWithInverseDeterminantForVariableHeightAndWidth) {
  configure_waveguide(0.6, false, false);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  perturb_height_and_width(transformation);
  expect_tensor_is_symmetric_with_inverse_determinant(transformation, sample_points());
}

TEST_F(WaveguideTransformationTests, StraightWaveguideIsIdentityForVariableHeightAndWidth) {
  configure_waveguide(0.0, false, false);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  for (Position p : sample_points()) {
    ASSERT_LT((transformation.math_to_phys(p) - p).norm(), 1e-14);
    ASSERT_LT((transformation.phys_to_math(p) - p).norm(), 1e-14);
    ASSERT_LT(max_difference(transformation.get_Space_Transformation_Tensor(p), identity_tensor()), 1e-14);
  }
}

// The vertical displacement is a hump: zero at both ends of the domain, the prescribed value in the middle and
// symmetric to the middle.
TEST_F(WaveguideTransformationTests, DisplacementFulfillsEndConditions) {
  configure_waveguide(1.3, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  ASSERT_NEAR(vertical_displacement_at(transformation, 0.0), 0.0, 1e-12);
  ASSERT_NEAR(vertical_displacement_at(transformation, 5.0), 1.3, 1e-10);
  ASSERT_NEAR(vertical_displacement_at(transformation, 10.0), 0.0, 1e-12);
  for (double z = 0.4; z < 5.0; z += 0.7) {
    ASSERT_NEAR(vertical_displacement_at(transformation, z), vertical_displacement_at(transformation, 10.0 - z), 1e-12);
  }
  // The waveguide leaves the input connector and passes the middle without a slope.
  Position start(0, 0, 1e-9), middle(0, 0, 5.0 - 1e-9);
  ASSERT_NEAR(transformation.get_J(start)[1][2], 0.0, 1e-6);
  ASSERT_NEAR(transformation.get_J(middle)[1][2], 0.0, 1e-6);
}

// The shape function mirrors at (upper end) / 2 instead of the middle of [z_min, z_max] (ShapeFunction.cpp:32).
TEST_F(WaveguideTransformationTests, DISABLED_DisplacementFulfillsEndConditionsInShiftedDomain) {
  helpers_b::reset_parameters_and_ranges(10.0, 2.0);
  configure_waveguide(1.3, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  ASSERT_NEAR(vertical_displacement_at(transformation, 2.0), 0.0, 1e-12);
  ASSERT_NEAR(vertical_displacement_at(transformation, 7.0), 1.3, 1e-10);
  ASSERT_NEAR(vertical_displacement_at(transformation, 12.0), 0.0, 1e-12);
}

TEST_F(WaveguideTransformationTests, NumberOfShapeDofs) {
  // Every shape function has (number of sectors - 2) free parameters.
  configure_waveguide(1.0, true, true);
  {
    WaveguideTransformation transformation;
    transformation.estimate_and_initialize();
    ASSERT_EQ(transformation.n_free_dofs(), n_sectors - 2);
    ASSERT_EQ(transformation.get_dof_values().size(), transformation.n_dofs());
    ASSERT_GT(transformation.n_dofs(), transformation.n_free_dofs());
  }
  configure_waveguide(1.0, false, true);
  {
    WaveguideTransformation transformation;
    transformation.estimate_and_initialize();
    ASSERT_EQ(transformation.n_free_dofs(), 2 * (n_sectors - 2));
    ASSERT_EQ(transformation.get_dof_values().size(), transformation.n_dofs());
  }
  configure_waveguide(1.0, false, false);
  {
    WaveguideTransformation transformation;
    transformation.estimate_and_initialize();
    ASSERT_EQ(transformation.n_free_dofs(), 3 * (n_sectors - 2));
    ASSERT_EQ(transformation.get_dof_values().size(), transformation.n_dofs());
  }
}

TEST_F(WaveguideTransformationTests, DofValuesContainConstraintsAndFreeDofs) {
  configure_waveguide(1.3, false, false);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  perturb_height_and_width(transformation);
  const dealii::Vector<double> values = transformation.get_dof_values();
  for (unsigned int i = 0; i < transformation.n_dofs(); i++) {
    ASSERT_EQ(values[i], transformation.get_dof(i));
  }
  // Per shape function: value at the start, slopes at the nodes (the first and the last one are prescribed, the
  // last but one is computed), value in the middle. The free dofs are the slopes 2, ..., n - 4.
  const unsigned int n_per_function = transformation.n_dofs() / 3;
  const unsigned int n_free_per_function = transformation.n_free_dofs() / 3;
  const double start_values[3] = {0.0, 1.0, 1.0};
  const double middle_values[3] = {1.3, 1.0, 1.0};
  for (unsigned int function = 0; function < 3; function++) {
    const unsigned int first = function * n_per_function;
    ASSERT_NEAR(transformation.get_dof(first), start_values[function], 1e-14);
    ASSERT_NEAR(transformation.get_dof(first + 1), 0.0, 1e-14);
    ASSERT_NEAR(transformation.get_dof(first + n_per_function - 2), 0.0, 1e-14);
    ASSERT_NEAR(transformation.get_dof(first + n_per_function - 1), middle_values[function], 1e-14);
    for (unsigned int i = 0; i < n_free_per_function; i++) {
      ASSERT_EQ(transformation.get_free_dof(function * n_free_per_function + i), transformation.get_dof(first + 2 + i));
    }
  }
}

TEST_F(WaveguideTransformationTests, SetFreeDofChangesOneParameterAndKeepsEndConditions) {
  configure_waveguide(1.3, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  for (unsigned int dof = 0; dof < transformation.n_free_dofs(); dof++) {
    std::vector<double> before;
    for (unsigned int i = 0; i < transformation.n_free_dofs(); i++) {
      before.push_back(transformation.get_free_dof(i));
    }
    const double displacement_before = vertical_displacement_at(transformation, 2.4);
    transformation.set_free_dof(dof, before[dof] + 0.25);
    for (unsigned int i = 0; i < transformation.n_free_dofs(); i++) {
      ASSERT_NEAR(transformation.get_free_dof(i), before[i] + ((i == dof) ? 0.25 : 0.0), 1e-14);
    }
    ASSERT_GT(std::abs(vertical_displacement_at(transformation, 2.4) - displacement_before), 1e-3);
    ASSERT_NEAR(vertical_displacement_at(transformation, 0.0), 0.0, 1e-12);
    ASSERT_NEAR(vertical_displacement_at(transformation, 5.0), 1.3, 1e-10);
    ASSERT_NEAR(vertical_displacement_at(transformation, 10.0), 0.0, 1e-12);
  }
  // The Jacobian follows the changed shape.
  expect_jacobian_matches_finite_differences(transformation, sample_points(), 1e-5);
}

// The displacement is linear in every shape parameter, so the Jacobian of a pure shift is linear as well and the
// tensor J J^T is a quadratic polynomial: its central difference quotient is exact.
TEST_F(WaveguideTransformationTests, DerivativeOfTensorWithRespectToShapeParameter) {
  configure_waveguide(1.0, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  const double step = 0.1;
  for (unsigned int dof = 0; dof < transformation.n_free_dofs(); dof++) {
    const double value = transformation.get_free_dof(dof);
    for (Position p : sample_points()) {
      transformation.set_free_dof(dof, value - step);
      const dealii::Tensor<2, 3, double> J_minus = transformation.get_J(p);
      const dealii::Tensor<2, 3, double> T_minus = transformation.get_Space_Transformation_Tensor(p);
      transformation.set_free_dof(dof, value + step);
      const dealii::Tensor<2, 3, double> J_plus = transformation.get_J(p);
      const dealii::Tensor<2, 3, double> T_plus = transformation.get_Space_Transformation_Tensor(p);
      transformation.set_free_dof(dof, value);
      const dealii::Tensor<2, 3, double> J = transformation.get_J(p);
      ASSERT_LT(max_difference(0.5 * (J_plus + J_minus), J), 1e-13);
      // d/da (J J^T) = J' J^T + J J'^T with J' = dJ/da.
      const dealii::Tensor<2, 3, double> dJ = (J_plus - J_minus) / (2 * step);
      const dealii::Tensor<2, 3, double> dT = dJ * dealii::transpose(J) + J * dealii::transpose(dJ);
      ASSERT_LT(max_difference((T_plus - T_minus) / (2 * step), dT), 1e-12);
    }
  }
}

// get_Tensor_for_step() reads the old value with the index of all dofs (get_dof) but writes with the index of the
// free dofs (set_free_dof), which is shifted by two (SpaceTransformation.cpp:35).
TEST_F(WaveguideTransformationTests, DISABLED_TensorForStepIsDifferenceOfTensors) {
  configure_waveguide(1.0, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  const unsigned int dof = 1;
  const double step = 0.01;
  Position p(0.3, -0.2, 2.4);
  const double value = transformation.get_free_dof(dof);
  const dealii::Tensor<2, 3, double> before = transformation.get_Space_Transformation_Tensor(p);
  transformation.set_free_dof(dof, value + step);
  const dealii::Tensor<2, 3, double> after = transformation.get_Space_Transformation_Tensor(p);
  transformation.set_free_dof(dof, value);
  const dealii::Tensor<2, 3, ComplexNumber> expected = after - before;
  ASSERT_GT(expected.norm(), 1e-4);
  ASSERT_LT(max_difference(transformation.get_Tensor_for_step(p, dof, step), expected), 1e-12);
}

// See above: the free dof is reset to the value of another dof.
TEST_F(WaveguideTransformationTests, DISABLED_TensorForStepLeavesShapeUnchanged) {
  configure_waveguide(1.0, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  Position p(0.3, -0.2, 2.4);
  const dealii::Vector<double> before = transformation.get_dof_values();
  transformation.get_Tensor_for_step(p, 1, 0.01);
  transformation.get_inverse_Tensor_for_step(p, 1, 0.01);
  const dealii::Vector<double> after = transformation.get_dof_values();
  for (unsigned int i = 0; i < before.size(); i++) {
    ASSERT_NEAR(after[i], before[i], 1e-14);
  }
}

// With a constant height the free dofs behind those of the displacement are still mapped to the height function
// (WaveguideTransformation.cpp:249), which is not part of the transformation in this case.
TEST_F(WaveguideTransformationTests, DISABLED_FreeDofsOfWidthAreAddressedForConstantHeight) {
  configure_waveguide(0.0, true, false);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  ASSERT_EQ(transformation.n_free_dofs(), 2 * (n_sectors - 2));
  const double x_before = transformation.math_to_phys(Position(1.0, 0.0, 2.4))[0];
  transformation.set_free_dof(n_sectors - 2 + 1, 0.05);
  ASSERT_GT(std::abs(transformation.math_to_phys(Position(1.0, 0.0, 2.4))[0] - x_before), 1e-4);
}

// The parameter "Initial shape parameters" of the case file: the slopes of the vertical displacement at the inner
// nodes.
TEST_F(WaveguideTransformationTests, InitialShapeParametersAreApplied) {
  configure_waveguide(0.8, true, true);
  GlobalParams.Number_of_sectors = 6;
  GlobalParams.initial_shape_parameters = {0.1, 0.3, 0.2, 0.05};
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  ASSERT_EQ(transformation.n_free_dofs(), 4u);
  // The shape functions use sectors of half the length, because the second half of the domain is the mirror image.
  const double node_distance = 10.0 / (2 * 6);
  for (unsigned int i = 0; i < 4; i++) {
    ASSERT_EQ(transformation.get_free_dof(i), GlobalParams.initial_shape_parameters[i]);
    Position near_node(0, 0, (i + 1) * node_distance + 1e-8);
    ASSERT_NEAR(-transformation.get_J(near_node)[1][2], GlobalParams.initial_shape_parameters[i], 1e-6);
  }
  ASSERT_NEAR(vertical_displacement_at(transformation, 0.0), 0.0, 1e-12);
  ASSERT_NEAR(vertical_displacement_at(transformation, 5.0), 0.8, 1e-10);
  ASSERT_NEAR(vertical_displacement_at(transformation, 10.0), 0.0, 1e-12);
  expect_maps_are_inverse(transformation, sample_points());
  expect_jacobian_matches_finite_differences(transformation, sample_points(), 1e-5);
}

// The way of the parameter from the case file to the transformation.
TEST_F(WaveguideTransformationTests, InitialShapeParametersAreReadFromCaseFile) {
  const std::filesystem::path directory = std::filesystem::temp_directory_path() / ("waveguide_tests_shape_parameters_" + std::to_string(getpid()));
  std::filesystem::create_directories(directory);
  const std::string run_file = (directory / "run.prm").string();
  const std::string case_file = (directory / "case.prm").string();
  {
    std::ofstream run(run_file);
    run << "subsection Run parameters\n  set Logging Level = Production One\nend\n";
    std::ofstream with_parameters(case_file);
    with_parameters << "subsection Case parameters\n  set source type = 1\n  set geometry size z = 10\n  set Number of shape sectors = 6\n"
                    << "  set vertical waveguide displacement = 0.8\n  set Initial shape parameters = 0.1, 0.3, 0.2, 0.05\nend\n";
  }
  ParameterOverride no_overrides;
  GlobalParams = GetParameters(run_file, case_file, no_overrides);
  ASSERT_EQ(GlobalParams.initial_shape_parameters, (std::vector<double>{0.1, 0.3, 0.2, 0.05}));
  Geometry.global_z_range = std::pair<double, double>(0.0, 10.0);
  {
    WaveguideTransformation transformation;
    transformation.estimate_and_initialize();
    ASSERT_EQ(transformation.n_free_dofs(), 4u);
    for (unsigned int i = 0; i < 4; i++) {
      ASSERT_EQ(transformation.get_free_dof(i), GlobalParams.initial_shape_parameters[i]);
    }
    ASSERT_NEAR(vertical_displacement_at(transformation, 5.0), 0.8, 1e-10);
  }
  // Without the parameter the list is empty and the default shape is used.
  {
    std::ofstream without_parameters(case_file);
    without_parameters << "subsection Case parameters\n  set source type = 1\n  set geometry size z = 10\n  set Number of shape sectors = 6\nend\n";
  }
  GlobalParams = GetParameters(run_file, case_file, no_overrides);
  ASSERT_TRUE(GlobalParams.initial_shape_parameters.empty());
  std::filesystem::remove_all(directory);
}

TEST_F(WaveguideTransformationTests, InitialShapeParametersChangeTheShape) {
  configure_waveguide(0.8, true, true);
  WaveguideTransformation default_shape;
  default_shape.estimate_and_initialize();
  GlobalParams.initial_shape_parameters = {0.0, 0.4, 0.0};
  WaveguideTransformation shape;
  shape.estimate_and_initialize();
  ASSERT_GT(std::abs(vertical_displacement_at(shape, 2.4) - vertical_displacement_at(default_shape, 2.4)), 1e-2);
}

TEST_F(WaveguideTransformationTests, WrongNumberOfInitialShapeParametersIsRejected) {
  configure_waveguide(0.8, true, true);
  GlobalParams.initial_shape_parameters = {0.1, 0.3};
  WaveguideTransformation transformation;
  ASSERT_THROW(transformation.estimate_and_initialize(), dealii::ExceptionBase);
}

TEST_F(SpaceTransformationTests, OperatorAppliesMathToPhysByDefault) {
  configure_waveguide(1.0, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  const Position p(0.3, -0.2, 2.4);
  ASSERT_LT((transformation(p) - transformation.math_to_phys(p)).norm(), 1e-14);
}

// operator() always applies math_to_phys (SpaceTransformation.cpp:57).
TEST_F(SpaceTransformationTests, DISABLED_OperatorAppliesPhysToMathAfterSwitch) {
  configure_waveguide(1.0, true, true);
  WaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  const Position p(0.3, -0.2, 2.4);
  ASSERT_GT((transformation.math_to_phys(p) - transformation.phys_to_math(p)).norm(), 1e-2);
  transformation.switch_application_mode(false);
  ASSERT_LT((transformation(p) - transformation.phys_to_math(p)).norm(), 1e-14);
  transformation.switch_application_mode(true);
  ASSERT_LT((transformation(p) - transformation.math_to_phys(p)).norm(), 1e-14);
}

TEST_F(SpaceTransformationTests, SectorOfZCoordinate) {
  GlobalParams.Number_of_sectors = 5;
  GlobalParams.Sector_thickness = 2.0;
  AngleWaveguideTransformation transformation;
  std::pair<int, double> sector = transformation.Z_to_Sector_and_local_z(3.0);
  ASSERT_EQ(sector.first, 1);
  ASSERT_NEAR(sector.second, 0.5, 1e-14);
  sector = transformation.Z_to_Sector_and_local_z(9.5);
  ASSERT_EQ(sector.first, 4);
  ASSERT_NEAR(sector.second, 0.75, 1e-14);
  // Outside of the domain: the start of the first and the end of the last sector.
  sector = transformation.Z_to_Sector_and_local_z(-0.3);
  ASSERT_EQ(sector.first, 0);
  ASSERT_EQ(sector.second, 0.0);
  sector = transformation.Z_to_Sector_and_local_z(10.4);
  ASSERT_EQ(sector.first, 4);
  ASSERT_EQ(sector.second, 1.0);
}

// The start of the domain is added to z instead of subtracted (SpaceTransformation.cpp:17).
TEST_F(SpaceTransformationTests, DISABLED_SectorOfZCoordinateInShiftedDomain) {
  helpers_b::reset_parameters_and_ranges(10.0, 2.0);
  GlobalParams.Number_of_sectors = 5;
  GlobalParams.Sector_thickness = 2.0;
  AngleWaveguideTransformation transformation;
  const std::pair<int, double> sector = transformation.Z_to_Sector_and_local_z(5.0);
  ASSERT_EQ(sector.first, 1);
  ASSERT_NEAR(sector.second, 0.5, 1e-14);
}

TEST_F(AngleTransformationTests, MapsAreInverse) {
  AngleWaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  expect_maps_are_inverse(transformation, sample_points());
  GlobalParams.PML_Angle_Test = 0.35;
  expect_maps_are_inverse(transformation, sample_points());
  // The waveguide is tilted: z changes proportionally to y.
  ASSERT_NEAR(transformation.math_to_phys(Position(0, 2.0, 1.0))[2], 1.0 + 0.35 * 2.0, 1e-14);
}

TEST_F(AngleTransformationTests, JacobianMatchesFiniteDifferences) {
  AngleWaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  expect_jacobian_matches_finite_differences(transformation, sample_points(), 1e-8);
  ASSERT_EQ(transformation.n_dofs(), 0u);
  ASSERT_EQ(transformation.n_free_dofs(), 0u);
}

// The Jacobian contains the constant 0.2 instead of GlobalParams.PML_Angle_Test (AngleWaveguideTransformation.cpp:22).
TEST_F(AngleTransformationTests, DISABLED_JacobianFollowsAngleParameter) {
  GlobalParams.PML_Angle_Test = 0.35;
  AngleWaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  expect_jacobian_matches_finite_differences(transformation, sample_points(), 1e-8);
}

TEST_F(AngleTransformationTests, TensorIsSymmetricWithDeterminantOne) {
  AngleWaveguideTransformation transformation;
  transformation.estimate_and_initialize();
  for (Position p : sample_points()) {
    // A shear has det(J) = 1.
    ASSERT_NEAR(transformation.get_det(p), 1.0, 1e-14);
    ASSERT_NEAR(dealii::determinant(transformation.get_J(p)), 1.0, 1e-14);
    ASSERT_GT(max_difference(transformation.get_Space_Transformation_Tensor(p), identity_tensor()), 0.1);
  }
  expect_tensor_is_symmetric_with_inverse_determinant(transformation, sample_points());
}

// BendTransformation is a stub: both maps return the origin and the tensor is zero.
TEST_F(BendTransformationTests, DISABLED_MapsAreInverseAndTensorIsRegular) {
  BendTransformation transformation;
  transformation.estimate_and_initialize();
  expect_maps_are_inverse(transformation, sample_points());
  expect_jacobian_matches_finite_differences(transformation, sample_points(), 1e-6);
  expect_tensor_is_symmetric_with_inverse_determinant(transformation, sample_points());
}

TEST_F(PredefinedShapeTests, MapsAreInverse) {
  set_sine_shape(41, 0.5, 0.5);
  PredefinedShapeTransformation transformation;
  transformation.estimate_and_initialize();
  expect_maps_are_inverse(transformation, points_between_nodes(20.0));
  ASSERT_GT(std::abs(vertical_displacement_at(transformation, 9.0)), 0.3);
}

TEST_F(PredefinedShapeTests, NodeValuesAndEndConditionsAreReproduced) {
  for (const double node_distance : {1.0, 0.5}) {
    set_sine_shape(41, node_distance, 0.5);
    PredefinedShapeTransformation transformation;
    transformation.estimate_and_initialize();
    ASSERT_EQ(transformation.case_sectors.size(), 40u);
    for (unsigned int i = 0; i < 41; i++) {
      ASSERT_NEAR(transformation.get_m(GlobalParams.sd.z[i]), GlobalParams.sd.m[i], 1e-12);
      ASSERT_NEAR(transformation.get_v(GlobalParams.sd.z[i]), GlobalParams.sd.v[i], 1e-12);
    }
    const double length = 40 * node_distance;
    // Outside of the domain the shape continues with its end values.
    for (const double z : {-0.7, 0.0, length, length + 0.7}) {
      ASSERT_NEAR(transformation.get_m(z), 0.0, 1e-12);
      ASSERT_NEAR(transformation.get_v(z), 0.0, 1e-12);
      Position p(0.3, -0.2, z);
      ASSERT_LT((transformation.math_to_phys(p) - p).norm(), 1e-12);
      ASSERT_LT(max_difference(transformation.get_Space_Transformation_Tensor(p), identity_tensor()), 1e-12);
    }
  }
}

TEST_F(PredefinedShapeTests, TensorIsSymmetricWithDeterminantOne) {
  set_sine_shape(41, 0.5, 0.5);
  PredefinedShapeTransformation transformation;
  transformation.estimate_and_initialize();
  for (Position p : points_between_nodes(20.0)) {
    ASSERT_NEAR(dealii::determinant(transformation.get_J(p)), 1.0, 1e-14);
    ASSERT_LT(max_difference(transformation.get_J(p) * transformation.get_J_inverse(p), identity_tensor()), 1e-13);
  }
  expect_tensor_is_symmetric_with_inverse_determinant(transformation, points_between_nodes(20.0));
}

// The displacement between two nodes is the cubic polynomial with the values m and the slopes v of the nodes. For
// nodes with the distance 1 its derivative at a node is the slope v of that node.
TEST_F(PredefinedShapeTests, JacobianMatchesFiniteDifferencesAtNodesForUnitNodeDistance) {
  set_sine_shape(41, 1.0, 1.0);
  PredefinedShapeTransformation transformation;
  transformation.estimate_and_initialize();
  double largest_slope = 0;
  for (unsigned int node = 1; node < 40; node++) {
    Position p(0.3, -0.2, node * 1.0);
    const dealii::Tensor<2, 3, double> J = transformation.get_J(p);
    ASSERT_LT(max_difference(J, jacobian_by_finite_differences(transformation, p)), 1e-5) << "at node " << node;
    largest_slope = std::max(largest_slope, std::abs(J[1][2]));
  }
  ASSERT_GT(largest_slope, 0.05);
}

// The local coordinate of a sector runs from 0 to 1, but the slopes v are used as derivatives with respect to it
// without the factor (length of the sector): the derivative of the displacement at a node is v / (length of the
// sector), here 2 v (Sector::get_m(), Sector.cpp:223).
TEST_F(PredefinedShapeTests, DISABLED_JacobianMatchesFiniteDifferencesAtNodesForOtherNodeDistances) {
  set_sine_shape(41, 0.5, 0.5);
  PredefinedShapeTransformation transformation;
  transformation.estimate_and_initialize();
  for (unsigned int node = 1; node < 40; node++) {
    Position p(0.3, -0.2, node * 0.5);
    ASSERT_LT(max_difference(transformation.get_J(p), jacobian_by_finite_differences(transformation, p)), 1e-5) << "at node " << node;
  }
}

// Between the nodes the Jacobian interpolates the slopes v with a cubic polynomial of its own (Sector::get_v(), Sector.cpp:236)
// instead of the derivative of the polynomial of the displacement. Here the two differ by up to 1.3 percent of the
// largest slope.
TEST_F(PredefinedShapeTests, DISABLED_JacobianMatchesFiniteDifferencesBetweenNodes) {
  set_sine_shape(41, 1.0, 1.0);
  PredefinedShapeTransformation transformation;
  transformation.estimate_and_initialize();
  for (Position p : points_between_nodes(40.0)) {
    ASSERT_LT(max_difference(transformation.get_J(p), jacobian_by_finite_differences(transformation, p)), 1e-5) << "at z = " << p[2];
  }
}

// All shapes of Modes/test.csv: complete data, 100 micrometers long, and the waveguide enters and leaves the domain
// on the axis without a slope.
TEST_F(PredefinedShapeTests, AllShapesOfRepositoryLoadAndFulfillEndConditions) {
  const std::vector<std::string> lines = lines_of_shape_file();
  ASSERT_GE(lines.size(), 36u);
  for (unsigned int shape = 0; shape < lines.size(); shape++) {
    ShapeDescription sd;
    sd.SetByString(lines[shape]);
    ASSERT_GT(sd.Sectors, 2) << "shape " << shape;
    ASSERT_EQ(sd.z.size(), (std::size_t) sd.Sectors);
    ASSERT_EQ(sd.m.size(), (std::size_t) sd.Sectors);
    ASSERT_EQ(sd.v.size(), (std::size_t) sd.Sectors);
    ASSERT_EQ(sd.z.front(), 0.0);
    ASSERT_NEAR(sd.z.back(), 100.0, 1e-9) << "shape " << shape;
    ASSERT_EQ(sd.m.front(), 0.0);
    ASSERT_EQ(sd.m.back(), 0.0);
    ASSERT_EQ(sd.v.front(), 0.0);
    ASSERT_EQ(sd.v.back(), 0.0);
    double largest_displacement = 0;
    for (int i = 0; i < sd.Sectors; i++) {
      ASSERT_TRUE(std::isfinite(sd.z[i]) && std::isfinite(sd.m[i]) && std::isfinite(sd.v[i]));
      // A slope of 45 degrees or more is not a plausible waveguide.
      ASSERT_LT(std::abs(sd.v[i]), 1.0) << "shape " << shape;
      largest_displacement = std::max(largest_displacement, std::abs(sd.m[i]));
    }
    // None of the shapes is the straight waveguide.
    ASSERT_GT(largest_displacement, 0.1) << "shape " << shape;
  }
}

// The solver can select the shapes 0 to 35 ("Predefined case number").
TEST_F(PredefinedShapeTests, SelectableShapesAreLoadedIntoTransformation) {
  const std::vector<std::string> lines = lines_of_shape_file();
  ASSERT_GE(lines.size(), 36u);
  for (unsigned int shape = 0; shape < 36; shape++) {
    ShapeDescription expected;
    expected.SetByString(lines[shape]);
    load_predefined_shape(shape);
    ASSERT_EQ(GlobalParams.sd.Sectors, expected.Sectors) << "shape " << shape;
    ASSERT_EQ(GlobalParams.sd.z, expected.z);
    ASSERT_EQ(GlobalParams.sd.m, expected.m);
    ASSERT_EQ(GlobalParams.sd.v, expected.v);
    PredefinedShapeTransformation transformation;
    transformation.estimate_and_initialize();
    ASSERT_EQ(transformation.case_sectors.size(), (std::size_t) expected.Sectors - 1);
    // At the input interface and in the PML in front of it the waveguide is straight.
    for (const double z : {-0.4, 0.0}) {
      ASSERT_NEAR(transformation.get_m(z), 0.0, 1e-12) << "shape " << shape;
      ASSERT_NEAR(transformation.get_v(z), 0.0, 1e-12) << "shape " << shape;
      Position p(0.3, -0.2, z);
      ASSERT_LT((transformation.math_to_phys(p) - p).norm(), 1e-12);
      ASSERT_LT(max_difference(transformation.get_Space_Transformation_Tensor(p), identity_tensor()), 1e-12);
    }
  }
}

// For z >= (end of the domain) the sector with the index Number_of_sectors - 1 is evaluated
// (SpaceTransformation.cpp:20), but Parameters::complete_data() does not set Number_of_sectors for a predefined
// shape. With its default value the displacement and the slope of the second node are used at the output interface
// and in the PML behind it.
TEST_F(PredefinedShapeTests, DISABLED_SelectableShapesAreStraightAtOutputInterface) {
  for (unsigned int shape = 0; shape < 36; shape++) {
    load_predefined_shape(shape);
    PredefinedShapeTransformation transformation;
    transformation.estimate_and_initialize();
    for (const double z : {100.0, 100.4}) {
      ASSERT_NEAR(transformation.get_m(z), 0.0, 1e-12) << "shape " << shape;
      ASSERT_NEAR(transformation.get_v(z), 0.0, 1e-12) << "shape " << shape;
      Position p(0.3, -0.2, z);
      ASSERT_LT(max_difference(transformation.get_Space_Transformation_Tensor(p), identity_tensor()), 1e-12);
    }
  }
}

// The nodes of the file are not equidistant. The transformation assumes sectors of the length z[1] - z[0]
// (Parameters.cpp:65, SpaceTransformation.cpp:17), so the displacement is evaluated at shifted positions: for
// this shape (hump of 5.5 micrometers) it is off by up to 0.44 micrometers at the nodes.
TEST_F(PredefinedShapeTests, DISABLED_NodeValuesOfRepositoryShapeAreReproduced) {
  load_predefined_shape(3);
  PredefinedShapeTransformation transformation;
  transformation.estimate_and_initialize();
  for (int i = 0; i < GlobalParams.sd.Sectors; i++) {
    ASSERT_NEAR(transformation.get_m(GlobalParams.sd.z[i]), GlobalParams.sd.m[i], 1e-3) << "node " << i << " at z = " << GlobalParams.sd.z[i];
  }
}

// In the shapes 26, 28, 29, 31, 33, 34 and 35 of Modes/test.csv some nodes are not in ascending order of z.
TEST_F(PredefinedShapeTests, DISABLED_NodesOfRepositoryShapesAreAscending) {
  const std::vector<std::string> lines = lines_of_shape_file();
  for (unsigned int shape = 0; shape < lines.size(); shape++) {
    ShapeDescription sd;
    sd.SetByString(lines[shape]);
    for (int i = 1; i < sd.Sectors; i++) {
      ASSERT_GT(sd.z[i], sd.z[i - 1]) << "shape " << shape << ", node " << i;
    }
  }
}
