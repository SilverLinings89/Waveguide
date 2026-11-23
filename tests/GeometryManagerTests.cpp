// Tests for the decomposition of the domain into one block per process (GeometryManager, MPICommunicator,
// DofIndexData). The process layout is set in the parameters, so all ranks can be checked in a single process.
#include <utility>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>
#include <mpi.h>
#include "../Code/BoundaryCondition/PMLSurface.h"
#include "../Code/Core/InnerDomain.h"
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GeometryManager.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Hierarchy/DofIndexData.h"
#include "../Code/Hierarchy/MPICommunicator.h"
#include "./HelpersB.h"

namespace {

  using helpers_b::set_process_layout;

  class GeometryManagerTests : public helpers_b::GlobalStateTest { };
  class MPICommunicatorTests : public helpers_b::GlobalStateTest { };
  class DofIndexDataTests : public helpers_b::GlobalStateTest { };

  using Layout = std::array<unsigned int, 3>;

  const std::vector<Layout> layouts = { Layout{1, 1, 1}, Layout{1, 1, 4}, Layout{1, 3, 2}, Layout{2, 3, 2}, Layout{3, 2, 2}, Layout{2, 2, 3} };

  const std::array<Direction, 6> all_directions = { Direction::MinusX, Direction::PlusX, Direction::MinusY, Direction::PlusY, Direction::MinusZ, Direction::PlusZ };

  unsigned int n_processes(const Layout & layout) {
    return layout[0] * layout[1] * layout[2];
  }

  void set_process(const Layout & layout, unsigned int rank) {
    set_process_layout(layout[0], layout[1], layout[2], rank);
  }

  using Box = std::array<std::pair<double, double>, 3>;

  Box box_of_current_process() {
    GeometryManager geometry;
    return Box{ geometry.compute_x_range(), geometry.compute_y_range(), geometry.compute_z_range() };
  }

  // Rank of the process that is one block further in the given direction, computed from the block indices.
  // The second value is false if there is no such block.
  std::pair<bool, unsigned int> expected_neighbor(const Layout & layout, unsigned int rank, Direction direction) {
    int index[3] = { (int)(rank % layout[0]), (int)((rank / layout[0]) % layout[1]), (int)(rank / (layout[0] * layout[1])) };
    const unsigned int axis = direction / 2;
    index[axis] += (direction % 2 == 0) ? -1 : 1;
    if (index[axis] < 0 || index[axis] >= (int) layout[axis]) {
      return std::pair<bool, unsigned int>(false, 0);
    }
    return std::pair<bool, unsigned int>(true, index[0] + layout[0] * (index[1] + layout[1] * index[2]));
  }

  Direction opposite(Direction direction) {
    return all_directions[(direction % 2 == 0) ? direction + 1 : direction - 1];
  }

  // The hierarchy uses one sweeping direction for processes that are only split in z, two if they are also split
  // in y and three if they are split in all directions.
  struct HierarchyCase {
    unsigned int sweeping_level;
    Layout layout;
  };

  const std::vector<HierarchyCase> hierarchy_cases = { {1, Layout{1, 1, 4}}, {2, Layout{1, 3, 2}}, {3, Layout{2, 3, 2}}, {3, Layout{3, 2, 2}} };

  // Removes everything that Geometry.initialize() created, so that the next call starts from scratch.
  void clear_global_geometry() {
    for (unsigned int level = 0; level < Geometry.levels.size(); level++) {
      for (unsigned int surface = 0; surface < 6; surface++) {
        Geometry.levels[level].surfaces[surface].reset();
      }
    }
    for (unsigned int surface = 0; surface < 6; surface++) {
      Geometry.surface_meshes[surface].clear();
    }
    Geometry.are_surface_meshes_initialized = false;
  }

}

TEST_F(GeometryManagerTests, SingleProcessOwnsTheGlobalDomain) {
  GlobalParams.Geometry_Size_X = 6.0;
  GlobalParams.Geometry_Size_Y = 5.4;
  GlobalParams.Geometry_Size_Z = 100.0;
  for (const double shift : {0.0, -50.0}) {
    GlobalParams.global_z_shift = shift;
    GeometryManager geometry;
    geometry.set_x_range(geometry.compute_x_range());
    geometry.set_y_range(geometry.compute_y_range());
    geometry.set_z_range(geometry.compute_z_range());
    // The cross-section is centered around the axis of the waveguide, the domain starts at the shift in z.
    ASSERT_EQ(geometry.global_x_range, (std::pair<double, double>(-3.0, 3.0)));
    ASSERT_EQ(geometry.global_y_range, (std::pair<double, double>(-2.7, 2.7)));
    ASSERT_EQ(geometry.global_z_range, (std::pair<double, double>(shift, shift + 100.0)));
    ASSERT_EQ(geometry.local_x_range, geometry.global_x_range);
    ASSERT_EQ(geometry.local_y_range, geometry.global_y_range);
    ASSERT_EQ(geometry.local_z_range, geometry.global_z_range);
  }
}

TEST_F(GeometryManagerTests, SubdomainsTileTheGlobalDomain) {
  GlobalParams.Geometry_Size_X = 6.0;
  GlobalParams.Geometry_Size_Y = 5.4;
  GlobalParams.Geometry_Size_Z = 12.0;
  for (const double shift : {0.0, 1.5}) {
    GlobalParams.global_z_shift = shift;
    const double lower_corner[3] = {-3.0, -2.7, shift};
    const double size[3] = {6.0, 5.4, 12.0};
    for (const Layout & layout : layouts) {
      std::vector<Box> boxes;
      double volume = 0;
      for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
        set_process(layout, rank);
        const Box box = box_of_current_process();
        const unsigned int index[3] = {GlobalParams.Index_in_x_direction, GlobalParams.Index_in_y_direction, GlobalParams.Index_in_z_direction};
        double box_volume = 1;
        for (unsigned int axis = 0; axis < 3; axis++) {
          const double length = size[axis] / layout[axis];
          ASSERT_NEAR(box[axis].first, lower_corner[axis] + index[axis] * length, 1e-12) << "rank " << rank << ", axis " << axis;
          ASSERT_NEAR(box[axis].second, lower_corner[axis] + (index[axis] + 1) * length, 1e-12) << "rank " << rank << ", axis " << axis;
          box_volume *= box[axis].second - box[axis].first;
        }
        volume += box_volume;
        boxes.push_back(box);
      }
      // The volumes add up to the volume of the domain and no two blocks overlap.
      ASSERT_NEAR(volume, 6.0 * 5.4 * 12.0, 1e-9);
      for (unsigned int a = 0; a < boxes.size(); a++) {
        for (unsigned int b = a + 1; b < boxes.size(); b++) {
          bool separated = false;
          for (unsigned int axis = 0; axis < 3; axis++) {
            if (boxes[a][axis].second <= boxes[b][axis].first + 1e-12 || boxes[b][axis].second <= boxes[a][axis].first + 1e-12) {
              separated = true;
            }
          }
          ASSERT_TRUE(separated) << "ranks " << a << " and " << b << " overlap";
        }
      }
    }
  }
}

TEST_F(GeometryManagerTests, GlobalNeighborsShareTheirInterface) {
  for (const Layout & layout : layouts) {
    for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
      for (const Direction direction : all_directions) {
        set_process(layout, rank);
        GeometryManager geometry;
        const Box own_box = box_of_current_process();
        const std::pair<bool, unsigned int> neighbor = geometry.get_global_neighbor_for_interface(direction);
        const std::pair<bool, unsigned int> expected = expected_neighbor(layout, rank, direction);
        ASSERT_EQ(neighbor.first, expected.first) << "rank " << rank << ", direction " << direction;
        ASSERT_EQ(geometry.get_level_neighbor_for_interface(direction, 0).first, expected.first);
        const unsigned int axis = direction / 2;
        if (!expected.first) {
          // Without a neighbor the face is part of the boundary of the global domain.
          geometry.set_x_range(own_box[0]);
          geometry.set_y_range(own_box[1]);
          geometry.set_z_range(own_box[2]);
          const Box global_box = Box{geometry.global_x_range, geometry.global_y_range, geometry.global_z_range};
          if (direction % 2 == 0) {
            ASSERT_NEAR(own_box[axis].first, global_box[axis].first, 1e-12);
          } else {
            ASSERT_NEAR(own_box[axis].second, global_box[axis].second, 1e-12);
          }
          continue;
        }
        ASSERT_EQ(neighbor.second, expected.second) << "rank " << rank << ", direction " << direction;
        ASSERT_EQ(geometry.get_level_neighbor_for_interface(direction, 0).second, expected.second);
        // Seen from the neighbor, this process is the neighbor in the opposite direction.
        set_process(layout, neighbor.second);
        const std::pair<bool, unsigned int> back = geometry.get_global_neighbor_for_interface(opposite(direction));
        ASSERT_TRUE(back.first);
        ASSERT_EQ(back.second, rank);
        const Box neighbor_box = box_of_current_process();
        for (unsigned int other_axis = 0; other_axis < 3; other_axis++) {
          if (other_axis == axis) {
            if (direction % 2 == 0) {
              ASSERT_NEAR(neighbor_box[axis].second, own_box[axis].first, 1e-12);
            } else {
              ASSERT_NEAR(neighbor_box[axis].first, own_box[axis].second, 1e-12);
            }
          } else {
            ASSERT_NEAR(neighbor_box[other_axis].first, own_box[other_axis].first, 1e-12);
            ASSERT_NEAR(neighbor_box[other_axis].second, own_box[other_axis].second, 1e-12);
          }
        }
      }
    }
  }
}

// Level 1 of get_level_neighbor_for_interface() is the communicator of one slab of constant z, level 2 the one of
// a row of constant y and z. Ranks in these communicators follow the order of the global ranks.
TEST_F(GeometryManagerTests, LevelNeighborsInXDirection) {
  for (const Layout & layout : layouts) {
    for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
      set_process(layout, rank);
      GeometryManager geometry;
      const unsigned int rank_in_slab = rank % (layout[0] * layout[1]);
      const unsigned int rank_in_row = rank % layout[0];
      for (const Direction direction : {Direction::MinusX, Direction::PlusX}) {
        const bool has_neighbor = expected_neighbor(layout, rank, direction).first;
        const int step = (direction == Direction::MinusX) ? -1 : 1;
        for (const unsigned int level : {1u, 2u}) {
          const std::pair<bool, unsigned int> neighbor = geometry.get_level_neighbor_for_interface(direction, level);
          ASSERT_EQ(neighbor.first, has_neighbor);
          if (has_neighbor) {
            ASSERT_EQ(neighbor.second, ((level == 1) ? rank_in_slab : rank_in_row) + step) << "rank " << rank << ", level " << level;
          }
        }
      }
      // The communicators do not contain the neighbors in the directions that were split off.
      for (const Direction direction : {Direction::MinusZ, Direction::PlusZ}) {
        ASSERT_FALSE(geometry.get_level_neighbor_for_interface(direction, 1).first);
        ASSERT_FALSE(geometry.get_level_neighbor_for_interface(direction, 2).first);
      }
      for (const Direction direction : {Direction::MinusY, Direction::PlusY}) {
        ASSERT_FALSE(geometry.get_level_neighbor_for_interface(direction, 2).first);
        ASSERT_EQ(geometry.get_level_neighbor_for_interface(direction, 1).first, expected_neighbor(layout, rank, direction).first);
      }
    }
  }
}

// The offset of the neighbor in y is the number of blocks in y instead of the number of blocks in x
// (GeometryManager.cpp:323, 330). Only correct if both numbers are equal.
TEST_F(GeometryManagerTests, DISABLED_LevelOneNeighborsInYDirection) {
  for (const Layout & layout : layouts) {
    for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
      set_process(layout, rank);
      GeometryManager geometry;
      const unsigned int rank_in_slab = rank % (layout[0] * layout[1]);
      for (const Direction direction : {Direction::MinusY, Direction::PlusY}) {
        const std::pair<bool, unsigned int> neighbor = geometry.get_level_neighbor_for_interface(direction, 1);
        if (neighbor.first) {
          const int step = (direction == Direction::MinusY) ? -(int) layout[0] : (int) layout[0];
          ASSERT_EQ(neighbor.second, rank_in_slab + step) << "layout " << layout[0] << "x" << layout[1] << "x" << layout[2] << ", rank " << rank;
        }
      }
    }
  }
}

TEST_F(GeometryManagerTests, LevelOneNeighborsInYDirectionForEqualBlockCounts) {
  const Layout layout{2, 2, 3};
  for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
    set_process(layout, rank);
    GeometryManager geometry;
    const unsigned int rank_in_slab = rank % 4;
    for (const Direction direction : {Direction::MinusY, Direction::PlusY}) {
      const std::pair<bool, unsigned int> neighbor = geometry.get_level_neighbor_for_interface(direction, 1);
      if (neighbor.first) {
        ASSERT_EQ(neighbor.second, rank_in_slab + ((direction == Direction::MinusY) ? -2 : 2));
      }
    }
  }
}

TEST_F(GeometryManagerTests, SurfacesOfSingleProcess) {
  GeometryManager geometry;
  for (BoundaryId surface = 0; surface < 6; surface++) {
    ASSERT_EQ(geometry.get_surface_type(surface, 0), SurfaceType::ABC_SURFACE);
    ASSERT_EQ(geometry.get_surface_type(surface, 1), SurfaceType::ABC_SURFACE);
  }
  // The mode is either coupled in by a Dirichlet condition on the input side or by a tapered source term.
  GlobalParams.Signal_coupling_method = SignalCouplingMethod::Dirichlet;
  for (BoundaryId surface = 0; surface < 6; surface++) {
    ASSERT_EQ(geometry.get_surface_type(surface, 0), (surface == 4) ? SurfaceType::DIRICHLET_SURFACE : SurfaceType::ABC_SURFACE);
  }
  // Perfectly conducting side walls are surfaces without degrees of freedom.
  GlobalParams.pec_side_walls = true;
  for (BoundaryId surface = 0; surface < 4; surface++) {
    ASSERT_EQ(geometry.get_surface_type(surface, 0), SurfaceType::OPEN_SURFACE);
  }
  ASSERT_EQ(geometry.get_surface_type(4, 0), SurfaceType::DIRICHLET_SURFACE);
  ASSERT_EQ(geometry.get_surface_type(5, 0), SurfaceType::ABC_SURFACE);
}

// On the highest level all processes are coupled: every interface between two blocks is a neighbor surface and
// the boundary of the global domain never is. On level 0 (one block alone) there are no neighbor surfaces.
TEST_F(GeometryManagerTests, NeighborSurfacesOfHierarchy) {
  for (const HierarchyCase & hierarchy_case : hierarchy_cases) {
    GlobalParams.Sweeping_Level = hierarchy_case.sweeping_level;
    const Layout & layout = hierarchy_case.layout;
    for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
      for (const Direction direction : all_directions) {
        set_process(layout, rank);
        GeometryManager geometry;
        const BoundaryId surface = geometry.get_boundary_for_direction(direction);
        const std::pair<bool, unsigned int> neighbor = expected_neighbor(layout, rank, direction);
        ASSERT_NE(geometry.get_surface_type(surface, 0), SurfaceType::NEIGHBOR_SURFACE);
        ASSERT_EQ(geometry.get_surface_type(surface, hierarchy_case.sweeping_level) == SurfaceType::NEIGHBOR_SURFACE, neighbor.first) << "rank " << rank << ", surface " << surface;
        bool is_neighbor_on_lower_level = false;
        for (unsigned int level = 0; level <= hierarchy_case.sweeping_level; level++) {
          const SurfaceType type = geometry.get_surface_type(surface, level);
          if (!neighbor.first) {
            ASSERT_EQ(type, SurfaceType::ABC_SURFACE);
            continue;
          }
          // A coupling that exists on one level also exists on all higher levels.
          if (is_neighbor_on_lower_level) {
            ASSERT_EQ(type, SurfaceType::NEIGHBOR_SURFACE);
          }
          is_neighbor_on_lower_level = (type == SurfaceType::NEIGHBOR_SURFACE);
          // Both sides of an interface agree on whether they are coupled.
          set_process(layout, neighbor.second);
          const SurfaceType type_of_neighbor = geometry.get_surface_type(geometry.get_boundary_for_direction(opposite(direction)), level);
          ASSERT_EQ(type == SurfaceType::NEIGHBOR_SURFACE, type_of_neighbor == SurfaceType::NEIGHBOR_SURFACE) << "rank " << rank << ", surface " << surface << ", level " << level;
          set_process(layout, rank);
        }
      }
    }
  }
}

// Sweeping preconditioner: where a block is not coupled to its neighbor, the interface towards the lower neighbor
// carries no degrees of freedom (open) and the one towards the upper neighbor is an absorbing boundary.
TEST_F(GeometryManagerTests, UncoupledInterfacesAreOpenBelowAndAbsorbingAbove) {
  for (const HierarchyCase & hierarchy_case : hierarchy_cases) {
    GlobalParams.Sweeping_Level = hierarchy_case.sweeping_level;
    const Layout & layout = hierarchy_case.layout;
    for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
      set_process(layout, rank);
      GeometryManager geometry;
      for (unsigned int level = 0; level <= hierarchy_case.sweeping_level; level++) {
        geometry.set_surface_types_and_properties(level);
        for (const Direction direction : all_directions) {
          const BoundaryId surface = geometry.get_boundary_for_direction(direction);
          const SurfaceType type = geometry.levels[level].surface_type[surface];
          ASSERT_EQ(type, geometry.get_surface_type(surface, level));
          ASSERT_EQ(geometry.levels[level].is_surface_truncated[surface], type == SurfaceType::ABC_SURFACE);
          if (expected_neighbor(layout, rank, direction).first && type != SurfaceType::NEIGHBOR_SURFACE) {
            ASSERT_EQ(type, (direction % 2 == 0) ? SurfaceType::OPEN_SURFACE : SurfaceType::ABC_SURFACE) << "rank " << rank << ", surface " << surface << ", level " << level;
          }
        }
      }
    }
  }
}

TEST_F(GeometryManagerTests, BoundaryIdsAndDirections) {
  GeometryManager geometry;
  for (const Direction direction : all_directions) {
    const BoundaryId surface = geometry.get_boundary_for_direction(direction);
    ASSERT_LT(surface, 6u);
    ASSERT_EQ(geometry.get_direction_for_boundary_id(surface), direction);
    // Two opposing surfaces have the ids 2 * axis (lower side) and 2 * axis + 1 (upper side).
    ASSERT_EQ(geometry.get_boundary_for_direction(opposite(direction)) / 2, surface / 2);
    ASSERT_NE(geometry.get_boundary_for_direction(opposite(direction)), surface);
  }
  ASSERT_EQ(geometry.get_boundary_for_direction(Direction::MinusX), 0u);
  ASSERT_EQ(geometry.get_boundary_for_direction(Direction::PlusY), 3u);
  ASSERT_EQ(geometry.get_boundary_for_direction(Direction::MinusZ), 4u);
}

TEST_F(GeometryManagerTests, MaterialOfCoreAndCladding) {
  GlobalParams.Width_of_waveguide = 2.0;
  GlobalParams.Height_of_waveguide = 1.8;
  GlobalParams.Epsilon_R_in_waveguide = 2.3409;
  GlobalParams.Epsilon_R_outside_waveguide = 1.8496;
  GlobalParams.Omega = 2 * GlobalParams.Pi / 1.55;
  GeometryManager geometry;
  // The core is the rectangle [-1, 1] x [-0.9, 0.9] for every z.
  for (const double z : {-3.0, 0.0, 47.1}) {
    ASSERT_TRUE(geometry.math_coordinate_in_waveguide(Position(0, 0, z)));
    ASSERT_TRUE(geometry.math_coordinate_in_waveguide(Position(0.99, -0.89, z)));
    ASSERT_TRUE(geometry.math_coordinate_in_waveguide(Position(-0.99, 0.89, z)));
    ASSERT_FALSE(geometry.math_coordinate_in_waveguide(Position(1.01, 0, z)));
    ASSERT_FALSE(geometry.math_coordinate_in_waveguide(Position(0, -0.91, z)));
    ASSERT_FALSE(geometry.math_coordinate_in_waveguide(Position(0.95, 0.95, z)));
    ASSERT_FALSE(geometry.math_coordinate_in_waveguide(Position(-1.5, -1.5, z)));
  }
  const Position in_core(0.3, -0.2, 1.0), in_cladding(1.4, 0.2, 1.0);
  ASSERT_EQ(geometry.get_epsilon_for_point(in_core), 2.3409);
  ASSERT_EQ(geometry.get_epsilon_for_point(in_cladding), 1.8496);
  const double omega_squared = std::pow(2 * GlobalParams.Pi / 1.55, 2);
  ASSERT_NEAR(geometry.kappa_2(), omega_squared, 1e-13);
  ASSERT_NEAR(geometry.eps_kappa_2(in_core), 2.3409 * omega_squared, 1e-12);
  ASSERT_NEAR(geometry.eps_kappa_2(in_cladding), 1.8496 * omega_squared, 1e-12);
  const dealii::Tensor<2, 3> tensor = geometry.get_epsilon_tensor(in_cladding);
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      ASSERT_EQ(tensor[i][j], (i == j) ? 1.8496 : 0.0);
    }
  }
}

// Geometry.initialize() for one process with PML on all sides: mesh sizes, surface meshes and the extension of the
// domain by the PML.
TEST_F(GeometryManagerTests, InitializationOfSingleProcessWithPML) {
  GlobalParams.Geometry_Size_X = 4.0;
  GlobalParams.Geometry_Size_Y = 3.0;
  GlobalParams.Geometry_Size_Z = 2.5;
  GlobalParams.Cells_in_x = 4;
  GlobalParams.Cells_in_y = 3;
  GlobalParams.Cells_in_z = 5;
  GlobalParams.Nedelec_element_order = 0;
  GlobalParams.Sweeping_Level = 1;
  GlobalParams.BoundaryCondition = BoundaryConditionType::PML;
  GlobalParams.PML_thickness = 0.8;
  GlobalParams.PML_N_Layers = 2;
  GlobalMPI = MPICommunicator();
  GlobalMPI.initialize();
  clear_global_geometry();
  Geometry.initialize();

  ASSERT_EQ(Geometry.local_x_range, (std::pair<double, double>(-2.0, 2.0)));
  ASSERT_EQ(Geometry.local_y_range, (std::pair<double, double>(-1.5, 1.5)));
  ASSERT_EQ(Geometry.local_z_range, (std::pair<double, double>(0.0, 2.5)));
  ASSERT_NEAR(Geometry.h_x, 1.0, 1e-14);
  ASSERT_NEAR(Geometry.h_y, 1.0, 1e-14);
  ASSERT_NEAR(Geometry.h_z, 0.5, 1e-14);
  const double extremal_coordinates[6] = {-2.0, 2.0, -1.5, 1.5, 0.0, 2.5};
  const unsigned int cells[3] = {4, 3, 5};
  ASSERT_TRUE(Geometry.are_surface_meshes_initialized);
  for (unsigned int surface = 0; surface < 6; surface++) {
    ASSERT_EQ(Geometry.surface_extremal_coordinate[surface], extremal_coordinates[surface]);
    ASSERT_EQ(Geometry.levels[0].surface_type[surface], SurfaceType::ABC_SURFACE);
    ASSERT_TRUE(Geometry.levels[0].is_surface_truncated[surface]);
    // The surface mesh has one cell per face of the 4 x 3 x 5 mesh on that side.
    const unsigned int axis = surface / 2;
    ASSERT_EQ(Geometry.surface_meshes[surface].n_active_cells(), cells[(axis + 1) % 3] * cells[(axis + 2) % 3]);
  }
  // Lowest order Nedelec elements have one degree of freedom per edge.
  const unsigned int n_edges = 4 * (3 + 1) * (5 + 1) + 3 * (4 + 1) * (5 + 1) + 5 * (4 + 1) * (3 + 1);
  ASSERT_EQ(Geometry.levels[0].inner_domain->triangulation.n_active_cells(), 4u * 3u * 5u);
  ASSERT_EQ(Geometry.levels[0].inner_domain->dof_handler.n_dofs(), n_edges);
  ASSERT_GT(Geometry.levels[0].n_total_level_dofs, n_edges);
  ASSERT_EQ(Geometry.levels[0].n_local_dofs, Geometry.levels[0].n_total_level_dofs);

  // Every PML domain extends the computational domain by the PML thickness in its own direction. The edges and
  // corners between two or three PML domains belong to one of them, so that all of them together fill the layer
  // of the PML thickness around the domain exactly once.
  double volume_of_all_pml_domains = 0;
  double lowest_of_all[3] = {1e10, 1e10, 1e10}, highest_of_all[3] = {-1e10, -1e10, -1e10};
  for (unsigned int surface = 0; surface < 6; surface++) {
    const PMLSurface * pml = dynamic_cast<PMLSurface *>(Geometry.levels[0].surfaces[surface].get());
    ASSERT_NE(pml, nullptr);
    const std::pair<double, double> ranges[3] = {pml->x_range, pml->y_range, pml->z_range};
    const unsigned int axis = surface / 2;
    if (surface % 2 == 0) {
      ASSERT_NEAR(ranges[axis].first, extremal_coordinates[surface] - 0.8, 1e-12);
      ASSERT_NEAR(ranges[axis].second, extremal_coordinates[surface], 1e-12);
    } else {
      ASSERT_NEAR(ranges[axis].first, extremal_coordinates[surface], 1e-12);
      ASSERT_NEAR(ranges[axis].second, extremal_coordinates[surface] + 0.8, 1e-12);
    }
    double lowest[3] = {1e10, 1e10, 1e10}, highest[3] = {-1e10, -1e10, -1e10};
    for (const auto & cell : pml->triangulation.active_cell_iterators()) {
      volume_of_all_pml_domains += cell->measure();
      for (unsigned int vertex = 0; vertex < 8; vertex++) {
        for (unsigned int i = 0; i < 3; i++) {
          lowest[i] = std::min(lowest[i], cell->vertex(vertex)[i]);
          highest[i] = std::max(highest[i], cell->vertex(vertex)[i]);
        }
      }
    }
    for (unsigned int i = 0; i < 3; i++) {
      if (i == axis) {
        ASSERT_NEAR(highest[i] - lowest[i], 0.8, 1e-12);
      } else {
        ASSERT_LE(lowest[i], extremal_coordinates[2 * i] + 1e-12);
        ASSERT_GE(highest[i], extremal_coordinates[2 * i + 1] - 1e-12);
      }
      lowest_of_all[i] = std::min(lowest_of_all[i], lowest[i]);
      highest_of_all[i] = std::max(highest_of_all[i], highest[i]);
    }
  }
  for (unsigned int i = 0; i < 3; i++) {
    ASSERT_NEAR(lowest_of_all[i], extremal_coordinates[2 * i] - 0.8, 1e-12);
    ASSERT_NEAR(highest_of_all[i], extremal_coordinates[2 * i + 1] + 0.8, 1e-12);
  }
  ASSERT_NEAR(volume_of_all_pml_domains, (4.0 + 1.6) * (3.0 + 1.6) * (2.5 + 1.6) - 4.0 * 3.0 * 2.5, 1e-9);

  // The inner domain is not deleted (InnerDomain has no virtual destructor), as in the solver.
  Geometry.levels[0].inner_domain = nullptr;
  clear_global_geometry();
  GlobalMPI = MPICommunicator();
}

TEST_F(MPICommunicatorTests, NeighborsAreReciprocalInAllLayouts) {
  for (const Layout & layout : layouts) {
    for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
      for (const Direction direction : all_directions) {
        set_process(layout, rank);
        MPICommunicator communicator;
        const std::pair<bool, unsigned int> neighbor = communicator.get_neighbor_for_interface(direction);
        const std::pair<bool, unsigned int> expected = expected_neighbor(layout, rank, direction);
        ASSERT_EQ(neighbor.first, expected.first) << "rank " << rank << ", direction " << direction;
        if (expected.first) {
          ASSERT_EQ(neighbor.second, expected.second) << "rank " << rank << ", direction " << direction;
          set_process(layout, neighbor.second);
          const std::pair<bool, unsigned int> back = communicator.get_neighbor_for_interface(opposite(direction));
          ASSERT_TRUE(back.first);
          ASSERT_EQ(back.second, rank);
        }
      }
    }
  }
}

// One communicator per level: level 0 is the process alone, the highest level all processes.
TEST_F(MPICommunicatorTests, CommunicatorsOfSingleProcess) {
  for (const unsigned int sweeping_level : {1u, 2u, 3u}) {
    GlobalParams.Sweeping_Level = sweeping_level;
    MPICommunicator communicator;
    communicator.initialize();
    ASSERT_EQ(communicator.communicators_by_level.size(), sweeping_level + 1);
    ASSERT_EQ(communicator.rank_on_level.size(), sweeping_level + 1);
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(communicator.communicators_by_level.front(), MPI_COMM_SELF, &comparison);
    ASSERT_EQ(comparison, MPI_IDENT);
    MPI_Comm_compare(communicator.communicators_by_level.back(), MPI_COMM_WORLD, &comparison);
    ASSERT_EQ(comparison, MPI_IDENT);
    for (unsigned int level = 0; level <= sweeping_level; level++) {
      int size = 0;
      MPI_Comm_size(communicator.communicators_by_level[level], &size);
      ASSERT_EQ(size, 1);
      ASSERT_EQ(communicator.rank_on_level[level], 0u);
    }
    // The communicators between the first and the last one were created by initialize().
    for (unsigned int level = 1; level < sweeping_level; level++) {
      MPI_Comm_free(&communicator.communicators_by_level[level]);
    }
  }
}

// Only for one sweeping level: with more levels the constructor copies objects that own memory without a copy
// constructor (LevelDofIndexData), which is freed twice.
TEST_F(DofIndexDataTests, NeighborFlagsFollowProcessLayout) {
  GlobalParams.Sweeping_Level = 1;
  for (const Layout & layout : layouts) {
    for (unsigned int rank = 0; rank < n_processes(layout); rank++) {
      set_process(layout, rank);
      DofIndexData data;
      ASSERT_EQ(data.indexCountsByLevel.size(), 1u);
      for (const Direction direction : all_directions) {
        ASSERT_EQ(data.isSurfaceNeighbor[direction], expected_neighbor(layout, rank, direction).first) << "rank " << rank << ", direction " << direction;
      }
    }
  }
}
