// Tests for mode files written by the Python package modesolver (header, permittivity column) and for the choice
// of the cross-section of the waveguide (rectangular, circular, from the mode file).
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Helpers/ParameterOverride.h"
#include "../Code/Helpers/ParameterReader.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/Solutions/ExactSolution.h"
#include "./HelpersA.h"

namespace {

  const double beta_of_test_file = 5.9;

  // A mode file in the new format on the grid x = -3, -2, ..., 3 and y = -2, -1, ..., 2. E_x = 2 inside the core
  // |x| <= 1, |y| <= 1 and 1 outside, so the largest norm is 2. The permittivity is 4 in the core and 2 outside,
  // except for the column x = 3, where it is 3 (to see the continuation beyond the grid).
  std::string mode_file_content(bool with_permittivity, const std::string & line_end = "\n") {
    std::ostringstream out;
    out << "# format = waveguide mode file 2" << line_end << "# a comment without a value" << line_end;
    out << "# beta = " << beta_of_test_file << line_end << "# wavelength = 1.55" << line_end;
    out << "# structure = test core 2 x 2 um" << line_end << "# mode = test mode" << line_end;
    for (int j = -2; j <= 2; j++) {
      for (int i = -3; i <= 3; i++) {
        const bool core = std::abs(i) <= 1 && std::abs(j) <= 1;
        out << i << " " << j << " " << (core ? 2.0 : 1.0) << " 0 " << 0.1 * i;
        if (with_permittivity) out << " " << (i == 3 ? 3.0 : (core ? 4.0 : 2.0));
        out << line_end;
      }
    }
    return out.str();
  }

  // Loads the mode file of the repository again after every test, because the tables of ExactSolution are static.
  class ModeFileTest : public ::testing::Test {
   protected:
    helpers_a::ScopedGlobalParams global_params;
    helpers_a::ScopedGeometry geometry;
    helpers_a::TemporaryDirectory directory{"mode_file"};

    std::string write(bool with_permittivity, const std::string & line_end = "\n") {
      helpers_a::write_file(directory.file("mode.dat"), mode_file_content(with_permittivity, line_end));
      return directory.file("mode.dat");
    }

    // Parameters of one process for a waveguide mode, with a domain that the test file covers.
    Parameters waveguide_parameters(const std::string & mode_file) {
      Parameters p;
      p.MPI_Rank = 0;
      p.NumberProcesses = 1;
      p.Point_Source_Type = 3;
      p.Logging_Level = LoggingLevel::PRODUCTION_ONE;
      p.Geometry_Size_X = 4.0;
      p.Geometry_Size_Y = 3.0;
      p.mode_file = mode_file;
      return p;
    }

    void complete(Parameters & p) {
      testing::internal::CaptureStdout();
      try {
        p.complete_data();
      } catch (...) {
        testing::internal::GetCapturedStdout();
        throw;
      }
      testing::internal::GetCapturedStdout();
      delete p.source_field;
      p.source_field = nullptr;
    }

    void TearDown() override {
      ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
    }
  };

}

TEST_F(ModeFileTest, HeaderAndTablesOfNewFormat) {
  ExactSolution::load_data(write(true));
  EXPECT_DOUBLE_EQ(ExactSolution::file_beta, beta_of_test_file);
  EXPECT_DOUBLE_EQ(ExactSolution::file_wavelength, 1.55);
  EXPECT_EQ(ExactSolution::file_structure, "test core 2 x 2 um");
  EXPECT_EQ(ExactSolution::file_mode, "test mode");
  EXPECT_TRUE(ExactSolution::has_permittivity);
  EXPECT_EQ(ExactSolution::n_intervals[0], 6u);
  EXPECT_EQ(ExactSolution::n_intervals[1], 4u);
  EXPECT_DOUBLE_EQ(ExactSolution::ranges[0].first, -3.0);
  EXPECT_DOUBLE_EQ(ExactSolution::ranges[0].second, 3.0);
  EXPECT_DOUBLE_EQ(ExactSolution::ranges[1].first, -2.0);
  EXPECT_DOUBLE_EQ(ExactSolution::ranges[1].second, 2.0);
  // The values are divided by the largest norm of a point: sqrt(2^2 + 0.1^2) at (1, 0) and (-1, 0).
  const double norm = std::sqrt(4.0 + 0.01);
  EXPECT_NEAR(ExactSolution::data_table_x[3][2], 2.0 / norm, 1e-14);
  EXPECT_NEAR(ExactSolution::data_table_x[0][0], 1.0 / norm, 1e-14);
  EXPECT_NEAR(ExactSolution::data_table_z[6][0], 0.3 / norm, 1e-14);
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_range.first, 2.0);
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_range.second, 4.0);
}

TEST_F(ModeFileTest, WindowsLineEndsAreAccepted) {
  ExactSolution::load_data(write(true, "\r\n"));
  EXPECT_DOUBLE_EQ(ExactSolution::file_beta, beta_of_test_file);
  EXPECT_EQ(ExactSolution::file_mode, "test mode");
  EXPECT_EQ(ExactSolution::n_intervals[0], 6u);
  EXPECT_EQ(ExactSolution::n_intervals[1], 4u);
}

TEST_F(ModeFileTest, FileOfTheRepositoryHasNoHeaderAndNoPermittivity) {
  ExactSolution::load_data(write(true));
  ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
  EXPECT_EQ(ExactSolution::file_beta, 0.0);
  EXPECT_EQ(ExactSolution::file_wavelength, 0.0);
  EXPECT_EQ(ExactSolution::file_structure, "");
  EXPECT_FALSE(ExactSolution::has_permittivity);
  EXPECT_EQ(ExactSolution::n_intervals[0], 200u);
  EXPECT_EQ(ExactSolution::n_intervals[1], 180u);
}

TEST_F(ModeFileTest, PermittivityIsTakenFromNearestPointAndContinuedOutside) {
  ExactSolution::load_data(write(true));
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_at(0.0, 0.0), 4.0);
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_at(1.4, -1.4), 4.0);   // nearest point (1, -1)
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_at(1.6, 0.0), 2.0);    // nearest point (2, 0)
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_at(0.0, 1.6), 2.0);
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_at(3.0, 0.0), 3.0);
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_at(50.0, 0.4), 3.0);   // beyond the grid: the value of its boundary
  EXPECT_DOUBLE_EQ(ExactSolution::permittivity_at(-50.0, -50.0), 2.0);
}

TEST_F(ModeFileTest, PathsAreResolvedInWorkingDirectoryAndThenInRepository) {
  EXPECT_EQ(ExactSolution::resolve_path("/some/absolute/path.dat"), "/some/absolute/path.dat");
  EXPECT_EQ(ExactSolution::resolve_path("Modes/output_mode.dat"), repository_path("Modes/output_mode.dat"));
  EXPECT_EQ(ExactSolution::resolve_path("no_such_folder/mode.dat"), repository_path("no_such_folder/mode.dat"));
  EXPECT_THROW(ExactSolution::load_data(ExactSolution::resolve_path("no_such_folder/mode.dat")), std::runtime_error);
}

TEST_F(ModeFileTest, PropagationConstantOfFileReplacesEpsilonEffective) {
  Parameters p = waveguide_parameters(write(true));
  p.Epsilon_R_effective = 1.0;
  complete(p);
  const double n_eff = beta_of_test_file * 1.55 / (2.0 * p.Pi);
  EXPECT_NEAR(p.Epsilon_R_effective, n_eff * n_eff, 1e-13);
  GlobalParams = p;
  EXPECT_NEAR(ExactSolution::beta(), beta_of_test_file, 1e-13);
}

TEST_F(ModeFileTest, FileWithoutHeaderKeepsEpsilonEffective) {
  Parameters p = waveguide_parameters("Modes/output_mode.dat");
  p.Epsilon_R_effective = 2.2;
  complete(p);
  EXPECT_DOUBLE_EQ(p.Epsilon_R_effective, 2.2);
}

TEST_F(ModeFileTest, ModeForAnotherWavelengthIsRefused) {
  Parameters p = waveguide_parameters(write(true));
  p.Lambda = 1.31;
  EXPECT_THROW(complete(p), std::runtime_error);
}

TEST_F(ModeFileTest, GridThatDoesNotCoverTheDomainIsRefused) {
  Parameters p = waveguide_parameters(write(true));
  p.Geometry_Size_X = 6.5;
  EXPECT_THROW(complete(p), std::runtime_error);
  p = waveguide_parameters(write(true));
  p.Geometry_Size_Y = 4.5;
  EXPECT_THROW(complete(p), std::runtime_error);
  p = waveguide_parameters(write(true));
  p.Geometry_Size_X = 6.0;
  p.Geometry_Size_Y = 4.0;
  EXPECT_NO_THROW(complete(p));
}

TEST_F(ModeFileTest, CrossSectionFromFileNeedsPermittivityColumn) {
  Parameters p = waveguide_parameters(write(false));
  p.cross_section = CrossSectionType::ModeFile;
  EXPECT_THROW(complete(p), std::runtime_error);
  p = waveguide_parameters(write(true));
  p.cross_section = CrossSectionType::ModeFile;
  EXPECT_NO_THROW(complete(p));
  // Other sources do not load a mode file, so there is no permittivity to take.
  ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
  p = waveguide_parameters(write(true));
  p.Point_Source_Type = 1;
  p.cross_section = CrossSectionType::ModeFile;
  EXPECT_THROW(complete(p), std::runtime_error);
}

TEST_F(ModeFileTest, MissingModeFileIsReported) {
  Parameters p = waveguide_parameters(directory.file("does_not_exist.dat"));
  EXPECT_THROW(complete(p), std::runtime_error);
}

TEST_F(ModeFileTest, RectangularAndCircularCore) {
  GlobalParams.Width_of_waveguide = 2.0;
  GlobalParams.Height_of_waveguide = 1.0;
  GlobalParams.Epsilon_R_in_waveguide = 2.5;
  GlobalParams.Epsilon_R_outside_waveguide = 1.5;
  GlobalParams.Omega = 3.0;
  GlobalParams.cross_section = CrossSectionType::Rectangular;
  EXPECT_TRUE(Geometry.math_coordinate_in_waveguide({0.9, 0.4, 7.0}));
  EXPECT_FALSE(Geometry.math_coordinate_in_waveguide({0.9, 0.6, 7.0}));
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({0.9, 0.4, 0.0}), 2.5);
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({0.9, 0.6, 0.0}), 1.5);
  // Circular: the diameter is the width; the height is not used.
  GlobalParams.cross_section = CrossSectionType::Circular;
  EXPECT_TRUE(Geometry.math_coordinate_in_waveguide({0.0, 0.99, 7.0}));
  EXPECT_TRUE(Geometry.math_coordinate_in_waveguide({0.7, 0.7, -3.0}));
  EXPECT_FALSE(Geometry.math_coordinate_in_waveguide({0.9, 0.6, 7.0}));
  EXPECT_FALSE(Geometry.math_coordinate_in_waveguide({-1.01, 0.0, 7.0}));
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({0.7, 0.7, 0.0}), 2.5);
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({0.9, 0.6, 0.0}), 1.5);
  EXPECT_DOUBLE_EQ(Geometry.eps_kappa_2({0.7, 0.7, 0.0}), 2.5 * 9.0);
  EXPECT_DOUBLE_EQ(Geometry.eps_kappa_2({0.9, 0.6, 0.0}), 1.5 * 9.0);
}

TEST_F(ModeFileTest, CrossSectionFromFileGivesTheMaterialOfTheSolver) {
  ExactSolution::load_data(write(true));
  GlobalParams.cross_section = CrossSectionType::ModeFile;
  GlobalParams.Epsilon_R_in_waveguide = 9.0;       // not used for this cross-section
  GlobalParams.Epsilon_R_outside_waveguide = 8.0;
  GlobalParams.Omega = 2.0;
  Geometry.global_x_range = std::pair<double, double>(-2.0, 2.0);
  Geometry.global_y_range = std::pair<double, double>(-1.5, 1.5);
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({0.2, -0.3, 5.0}), 4.0);
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({1.8, 0.0, 5.0}), 2.0);
  EXPECT_DOUBLE_EQ(Geometry.eps_kappa_2({0.2, -0.3, 5.0}), 16.0);
  EXPECT_TRUE(Geometry.math_coordinate_in_waveguide({0.2, -0.3, 5.0}));
  EXPECT_FALSE(Geometry.math_coordinate_in_waveguide({1.8, 0.0, 5.0}));
  // A point in an absorbing layer beside the domain gets the material of the nearest point of the domain (x = 2,
  // permittivity 2), not the material that the file has further out (3 at x = 3).
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({3.0, 0.0, 5.0}), 2.0);
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({0.0, 2.4, 5.0}), 2.0);
  // With a domain that ends inside the core, the core continues into the layer.
  Geometry.global_y_range = std::pair<double, double>(-1.4, 1.4);
  EXPECT_DOUBLE_EQ(Geometry.get_epsilon_for_point({0.0, 2.4, 5.0}), 4.0);
}

TEST_F(ModeFileTest, CaseFileEntriesAreRead) {
  helpers_a::write_file(directory.file("run.prm"), "subsection Run parameters\nend\n");
  ParameterOverride no_override;
  for (const auto & entry : std::vector<std::pair<std::string, CrossSectionType>>{
           {"rectangular", CrossSectionType::Rectangular}, {"circular", CrossSectionType::Circular}, {"mode file", CrossSectionType::ModeFile}}) {
    helpers_a::write_file(directory.file("case.prm"), "subsection Case parameters\n  set source type = 3\n  set geometry size x = 4\n  set geometry size y = 3\n  set mode file = " +
                          write(true) + "\n  set waveguide cross section = " + entry.first + "\nend\n");
    testing::internal::CaptureStdout();
    Parameters p = GetParameters(directory.file("run.prm"), directory.file("case.prm"), no_override);
    testing::internal::GetCapturedStdout();
    EXPECT_EQ(p.cross_section, entry.second);
    EXPECT_EQ(p.mode_file, directory.file("mode.dat"));
    delete p.source_field;
  }
  // The defaults keep the behaviour of earlier versions.
  const Parameters defaults = Parameters();
  EXPECT_EQ(defaults.mode_file, "Modes/output_mode.dat");
  EXPECT_EQ(defaults.cross_section, CrossSectionType::Rectangular);
}
