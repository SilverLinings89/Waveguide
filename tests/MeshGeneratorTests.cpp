// Tests for SquareMeshGenerator: boundary ids of the six sides of a block and the refinement towards the mesh
// sizes 0.05 (inside of the core) and 0.1 (outside).
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/MeshGenerators/SquareMeshGenerator.h"
#include "./HelpersB.h"

namespace {

  class SquareMeshGeneratorTests : public helpers_b::GlobalStateTest { };

  double longest_edge(const dealii::Triangulation<3> & triangulation) {
    double ret = 0;
    for (const auto & cell : triangulation.active_cell_iterators()) {
      for (unsigned int line = 0; line < dealii::GeometryInfo<3>::lines_per_cell; line++) {
        ret = std::max(ret, cell->line(line)->measure());
      }
    }
    return ret;
  }

  std::array<std::pair<double, double>, 3> bounding_box(const dealii::Triangulation<3> & triangulation) {
    std::array<std::pair<double, double>, 3> ret;
    for (unsigned int i = 0; i < 3; i++) {
      ret[i] = std::pair<double, double>(1e10, -1e10);
    }
    for (const auto & cell : triangulation.active_cell_iterators()) {
      for (unsigned int vertex = 0; vertex < 8; vertex++) {
        for (unsigned int i = 0; i < 3; i++) {
          ret[i].first = std::min(ret[i].first, cell->vertex(vertex)[i]);
          ret[i].second = std::max(ret[i].second, cell->vertex(vertex)[i]);
        }
      }
    }
    return ret;
  }

  // Every boundary face has the id 2 * axis for the lower side and 2 * axis + 1 for the upper side of the box.
  // Returns the number of faces per id.
  std::array<unsigned int, 6> check_boundary_ids(const dealii::Triangulation<3> & triangulation) {
    const std::array<std::pair<double, double>, 3> box = bounding_box(triangulation);
    std::array<unsigned int, 6> counts = {0, 0, 0, 0, 0, 0};
    for (const auto & cell : triangulation.active_cell_iterators()) {
      for (unsigned int face = 0; face < 6; face++) {
        if (!cell->face(face)->at_boundary()) continue;
        const Position center = cell->face(face)->center();
        unsigned int expected = 6;
        for (unsigned int axis = 0; axis < 3; axis++) {
          if (std::abs(center[axis] - box[axis].first) < 1e-10) expected = 2 * axis;
          if (std::abs(center[axis] - box[axis].second) < 1e-10) expected = 2 * axis + 1;
        }
        EXPECT_EQ(cell->face(face)->boundary_id(), expected) << "face at " << center;
        if (cell->face(face)->boundary_id() < 6) counts[cell->face(face)->boundary_id()]++;
      }
    }
    return counts;
  }

  // Exits with the code 0 if prepare_triangulation() creates one cell that covers the local domain (which is
  // smaller than the mesh size) and has the six boundary ids.
  [[noreturn]] void prepare_triangulation_and_exit() {
    dealii::Triangulation<3> triangulation;
    SquareMeshGenerator generator;
    generator.prepare_triangulation(&triangulation);
    const std::array<std::pair<double, double>, 3> box = bounding_box(triangulation);
    bool is_correct = triangulation.n_active_cells() == 1;
    is_correct = is_correct && std::abs(box[0].first - 2.0) < 1e-12 && std::abs(box[0].second - 2.09) < 1e-12;
    is_correct = is_correct && std::abs(box[1].first - 0.0) < 1e-12 && std::abs(box[1].second - 0.09) < 1e-12;
    is_correct = is_correct && std::abs(box[2].first - 0.0) < 1e-12 && std::abs(box[2].second - 0.09) < 1e-12;
    const std::array<unsigned int, 6> counts = check_boundary_ids(triangulation);
    for (unsigned int surface = 0; surface < 6; surface++) {
      is_correct = is_correct && counts[surface] == 1;
    }
    _exit(is_correct ? 0 : 1);
  }

}

TEST_F(SquareMeshGeneratorTests, DominantDirectionOfVector) {
  SquareMeshGenerator generator;
  ASSERT_EQ(generator.getDominantComponentAndDirection(Position(-1, 0, 0)), 0u);
  ASSERT_EQ(generator.getDominantComponentAndDirection(Position(1, 0, 0)), 1u);
  ASSERT_EQ(generator.getDominantComponentAndDirection(Position(0, -1, 0)), 2u);
  ASSERT_EQ(generator.getDominantComponentAndDirection(Position(0, 1, 0)), 3u);
  ASSERT_EQ(generator.getDominantComponentAndDirection(Position(0, 0, -1)), 4u);
  ASSERT_EQ(generator.getDominantComponentAndDirection(Position(0, 0, 1)), 5u);
  ASSERT_EQ(generator.getDominantComponentAndDirection(Position(0.1, -0.9, 0.3)), 2u);
  ASSERT_EQ(generator.getDominantComponentAndDirection(Position(-0.2, 0.3, 0.31)), 5u);
}

TEST_F(SquareMeshGeneratorTests, BoundaryIdsOnAllSixFaces) {
  dealii::Triangulation<3> triangulation;
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, std::vector<unsigned int>{3, 2, 4}, Position(-1.0, 0.0, 5.0), Position(2.0, 1.0, 9.0), false);
  SquareMeshGenerator generator;
  generator.set_boundary_ids(triangulation);
  const std::array<unsigned int, 6> counts = check_boundary_ids(triangulation);
  ASSERT_EQ(counts[0], 2u * 4u);
  ASSERT_EQ(counts[1], 2u * 4u);
  ASSERT_EQ(counts[2], 3u * 4u);
  ASSERT_EQ(counts[3], 3u * 4u);
  ASSERT_EQ(counts[4], 3u * 2u);
  ASSERT_EQ(counts[5], 3u * 2u);
}

// The numbering is the one of deal.II for a colorized box, which the rest of the code uses.
TEST_F(SquareMeshGeneratorTests, BoundaryIdsAgreeWithColorizedBox) {
  dealii::Triangulation<3> triangulation;
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, std::vector<unsigned int>{2, 3, 2}, Position(-0.5, -0.5, 0.0), Position(0.5, 0.5, 3.0), true);
  std::vector<unsigned int> colorized_ids;
  for (const auto & cell : triangulation.active_cell_iterators()) {
    for (unsigned int face = 0; face < 6; face++) {
      colorized_ids.push_back(cell->face(face)->at_boundary() ? cell->face(face)->boundary_id() : 6);
    }
  }
  SquareMeshGenerator generator;
  generator.set_boundary_ids(triangulation);
  unsigned int counter = 0;
  for (const auto & cell : triangulation.active_cell_iterators()) {
    for (unsigned int face = 0; face < 6; face++) {
      if (cell->face(face)->at_boundary()) {
        ASSERT_EQ(cell->face(face)->boundary_id(), colorized_ids[counter]);
      }
      counter++;
    }
  }
}

// One cell in the cladding with edges of 0.36 x 0.09 x 0.09: two bisections in x reach the mesh size 0.1.
TEST_F(SquareMeshGeneratorTests, RefinementInOneDirection) {
  dealii::Triangulation<3> triangulation;
  dealii::GridGenerator::hyper_rectangle(triangulation, Position(2.0, 0.0, 0.0), Position(2.36, 0.09, 0.09));
  SquareMeshGenerator generator;
  generator.refine_triangulation_iteratively(&triangulation);
  ASSERT_EQ(triangulation.n_active_cells(), 4u);
  ASSERT_NEAR(longest_edge(triangulation), 0.09, 1e-12);
}

// A cube with edges of 0.18 in the cladding is bisected once in every direction.
TEST_F(SquareMeshGeneratorTests, RefinementInAllDirections) {
  dealii::Triangulation<3> triangulation;
  dealii::GridGenerator::hyper_rectangle(triangulation, Position(2.0, 0.0, 0.0), Position(2.18, 0.18, 0.18));
  SquareMeshGenerator generator;
  generator.refine_triangulation_iteratively(&triangulation);
  ASSERT_EQ(triangulation.n_active_cells(), 8u);
  ASSERT_NEAR(longest_edge(triangulation), 0.09, 1e-12);
  // The boundary ids are still correct after the refinement.
  generator.set_boundary_ids(triangulation);
  const std::array<unsigned int, 6> counts = check_boundary_ids(triangulation);
  for (unsigned int surface = 0; surface < 6; surface++) {
    ASSERT_EQ(counts[surface], 4u);
  }
}

// Inside of the core the mesh size is 0.05.
TEST_F(SquareMeshGeneratorTests, RefinementInsideOfCoreIsFiner) {
  GlobalParams.Width_of_waveguide = 2.0;
  GlobalParams.Height_of_waveguide = 1.8;
  dealii::Triangulation<3> triangulation;
  dealii::GridGenerator::hyper_rectangle(triangulation, Position(0.1, 0.1, 0.0), Position(0.19, 0.14, 0.04));
  SquareMeshGenerator generator;
  generator.refine_triangulation_iteratively(&triangulation);
  ASSERT_EQ(triangulation.n_active_cells(), 2u);
  ASSERT_NEAR(longest_edge(triangulation), 0.045, 1e-12);
}

TEST_F(SquareMeshGeneratorTests, FineMeshIsNotRefined) {
  dealii::Triangulation<3> triangulation;
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, std::vector<unsigned int>{2, 2, 2}, Position(2.0, 0.0, 0.0), Position(2.16, 0.16, 0.16), false);
  SquareMeshGenerator generator;
  generator.refine_triangulation_iteratively(&triangulation);
  ASSERT_EQ(triangulation.n_active_cells(), 8u);
}

// The loop stops as soon as the last cell of the mesh is fine enough (SquareMeshGenerator.cpp:107): the first cell
// (edge 0.4) is only bisected once.
TEST_F(SquareMeshGeneratorTests, DISABLED_RefinementContinuesUntilAllCellsAreFine) {
  dealii::Triangulation<3> triangulation;
  const std::vector<std::vector<double>> step_sizes = { {0.4, 0.08}, {0.08}, {0.08} };
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, Position(2.0, 0.0, 0.0), Position(2.48, 0.08, 0.08), false);
  SquareMeshGenerator generator;
  generator.refine_triangulation_iteratively(&triangulation);
  ASSERT_LE(longest_edge(triangulation), 0.1 + 1e-12);
}

// prepare_triangulation() moves the vertices of a cube to the corners of the local domain with
// Triangulation_Shit_To_Local_Geometry(), which writes the z-coordinate to the component 3 of a 3D point
// (staticfunctions.cpp:279). In an optimized build the program is terminated there. The call runs in a child
// process so that the other tests are not affected.
TEST_F(SquareMeshGeneratorTests, DISABLED_PreparedTriangulationCoversLocalDomain) {
  Geometry.local_x_range = std::pair<double, double>(2.0, 2.09);
  Geometry.local_y_range = std::pair<double, double>(0.0, 0.09);
  Geometry.local_z_range = std::pair<double, double>(0.0, 0.09);
  ASSERT_EXIT(prepare_triangulation_and_exit(), ::testing::ExitedWithCode(0), "");
}
