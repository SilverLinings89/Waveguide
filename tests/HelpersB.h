#pragma once

// Helpers of the tests for the space transformations, the exact solutions, the geometry manager, the mode classes
// and the mesh generator.
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <utility>
#include <deal.II/base/tensor.h>
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/Solutions/ExactSolution.h"
#include "../Code/SpaceTransformations/SpaceTransformation.h"

namespace helpers_b {

  // Parameters of a single process with their default values and the ranges of the domain that belong to them.
  // The domain is [-size_x/2, size_x/2] x [-size_y/2, size_y/2] x [z_shift, z_shift + size_z].
  inline void reset_parameters_and_ranges(double size_z = 10.0, double z_shift = 0.0) {
    GlobalParams = Parameters();
    GlobalParams.Logging_Level = LoggingLevel::PRODUCTION_ONE;
    GlobalParams.NumberProcesses = 1;
    GlobalParams.MPI_Rank = 0;
    GlobalParams.Index_in_x_direction = 0;
    GlobalParams.Index_in_y_direction = 0;
    GlobalParams.Index_in_z_direction = 0;
    GlobalParams.Geometry_Size_Z = size_z;
    GlobalParams.global_z_shift = z_shift;
    GlobalParams.source_field = nullptr;
    const double half_x = GlobalParams.Geometry_Size_X / 2.0;
    const double half_y = GlobalParams.Geometry_Size_Y / 2.0;
    Geometry.global_x_range = std::pair<double, double>(-half_x, half_x);
    Geometry.global_y_range = std::pair<double, double>(-half_y, half_y);
    Geometry.global_z_range = std::pair<double, double>(z_shift, z_shift + size_z);
    Geometry.local_x_range = Geometry.global_x_range;
    Geometry.local_y_range = Geometry.global_y_range;
    Geometry.local_z_range = Geometry.global_z_range;
  }

  // Process layout with n_x x n_y x n_z blocks. Ranks are numbered x-first: rank = i_x + n_x * (i_y + n_y * i_z).
  inline void set_process_layout(unsigned int n_x, unsigned int n_y, unsigned int n_z, unsigned int rank) {
    GlobalParams.Blocks_in_x_direction = n_x;
    GlobalParams.Blocks_in_y_direction = n_y;
    GlobalParams.Blocks_in_z_direction = n_z;
    GlobalParams.NumberProcesses = n_x * n_y * n_z;
    GlobalParams.MPI_Rank = rank;
    GlobalParams.Index_in_x_direction = rank % n_x;
    GlobalParams.Index_in_y_direction = (rank / n_x) % n_y;
    GlobalParams.Index_in_z_direction = rank / (n_x * n_y);
  }

  // Base class of the fixtures. Every test starts from default parameters and leaves default parameters behind,
  // because the tests of this binary share the global objects.
  class GlobalStateTest : public ::testing::Test {
  protected:
    void SetUp() override {
      reset_parameters_and_ranges();
    }

    void TearDown() override {
      if (owned_transformation != nullptr) {
        if (GlobalSpaceTransformation == owned_transformation) {
          GlobalSpaceTransformation = nullptr;
        }
        delete owned_transformation;
        owned_transformation = nullptr;
      }
      reset_parameters_and_ranges(Parameters().Geometry_Size_Z);
      GlobalParams.Logging_Level = Parameters().Logging_Level;
    }

    // Installs the transformation as GlobalSpaceTransformation. It is deleted at the end of the test.
    void use_as_global_transformation(SpaceTransformation * transformation) {
      owned_transformation = transformation;
      GlobalSpaceTransformation = transformation;
    }

    void load_mode_profile() {
      ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
    }

  private:
    SpaceTransformation * owned_transformation = nullptr;
  };

  // Largest difference of two entries. Infinity if an entry is not finite.
  template <typename Number>
  double max_difference(const dealii::Tensor<2, 3, Number> & a, const dealii::Tensor<2, 3, Number> & b) {
    double ret = 0;
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        const double difference = std::abs(a[i][j] - b[i][j]);
        if (!std::isfinite(difference)) {
          return std::numeric_limits<double>::infinity();
        }
        ret = std::max(ret, difference);
      }
    }
    return ret;
  }

  template <typename Number>
  double max_difference(const dealii::Tensor<1, 3, Number> & a, const dealii::Tensor<1, 3, Number> & b) {
    double ret = 0;
    for (unsigned int i = 0; i < 3; i++) {
      const double difference = std::abs(a[i] - b[i]);
      if (!std::isfinite(difference)) {
        return std::numeric_limits<double>::infinity();
      }
      ret = std::max(ret, difference);
    }
    return ret;
  }

  inline dealii::Tensor<2, 3, double> identity_tensor() {
    dealii::Tensor<2, 3, double> ret;
    for (unsigned int i = 0; i < 3; i++) {
      ret[i][i] = 1.0;
    }
    return ret;
  }

  // Jacobian of phys_to_math by central differences: J[i][j] = d math_i / d phys_j.
  inline dealii::Tensor<2, 3, double> jacobian_by_finite_differences(const SpaceTransformation & transformation, const Position & p, double h = 1e-6) {
    dealii::Tensor<2, 3, double> ret;
    for (unsigned int j = 0; j < 3; j++) {
      Position plus = p, minus = p;
      plus[j] += h;
      minus[j] -= h;
      const dealii::Tensor<1, 3, double> d = (transformation.phys_to_math(plus) - transformation.phys_to_math(minus)) / (2 * h);
      for (unsigned int i = 0; i < 3; i++) {
        ret[i][j] = d[i];
      }
    }
    return ret;
  }

}
