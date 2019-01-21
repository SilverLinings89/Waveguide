#ifndef SquareMeshGeneratorCppFlag
#define SquareMeshGeneratorCppFlag

#include "SquareMeshGenerator.h"
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/std_cxx11/array.h>
#include <deal.II/base/std_cxx11/bind.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include "../Helpers/staticfunctions.h"

using namespace dealii;

void mesh_i(const Triangulation<3, 3> &tria, const std::string &filename) {
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << 3 << std::endl
            << " no. of cells: " << tria.n_active_cells() << std::endl;

  std::ofstream out(filename.c_str());
  GridOut grid_out;
  grid_out.write_vtk(tria, out);
  out.close();
  std::cout << " written to " << filename << std::endl << std::endl;
}

SquareMeshGenerator::SquareMeshGenerator(SpaceTransformation *in_ct)
    : MaxDistX((GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In) * 1.4 /
               2.0),
      MaxDistY((GlobalParams.M_C_Dim2Out + GlobalParams.M_C_Dim2In) * 1.4 /
               2.0) {
  ct = in_ct;
  Layers = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  origin = Point<3>(-1, -1, -1);
  p1 = Point<3>(-1, -1, -1);
  p2 = Point<3>(1, 1, 1);
  edges[0][0] = 2;
  edges[0][1] = 0;
  edges[0][2] = 0;

  edges[1][0] = 0;
  edges[1][1] = 2;
  edges[1][2] = 0;

  edges[2][0] = 0;
  edges[2][1] = 0;
  edges[2][2] = 2;

  subs.push_back(1);
  subs.push_back(1);
  subs.push_back(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
}

SquareMeshGenerator::~SquareMeshGenerator() {}

void SquareMeshGenerator::set_boundary_ids(Triangulation<3> &tria) const {
  Triangulation<3>::active_cell_iterator cell2 = tria.begin_active(),
                                         endc2 = tria.end();
  tria.set_all_manifold_ids(0);
  for (; cell2 != endc2; ++cell2) {
    if (cell2->at_boundary()) {
      for (int j = 0; j < 6; j++) {
        Point<3> ctr = cell2->face(j)->center();
        if (cell2->face(j)->at_boundary()) {
          cell2->face(j)->set_all_boundary_ids(1);

          if (std::abs(ctr(2) - GlobalParams.Minimum_Z) < 0.00001) {
            cell2->face(j)->set_all_boundary_ids(2);
          }
          if (std::abs(ctr(2) - GlobalParams.Maximum_Z) < 0.00001) {
            cell2->face(j)->set_all_boundary_ids(3);
          }
        }
      }
    }
  }
}

void SquareMeshGenerator::prepare_triangulation(Triangulation<3, 3> *in_tria) {
  deallog.push("SquareMeshGenerator:prepare_triangulation");
  deallog << "Starting Mesh preparation" << std::endl;
  Triangulation<2, 2> surface;

  GridGenerator::subdivided_hyper_cube<2, 2>(surface, 3,
                                             -GlobalParams.M_R_XLength / 2.0,
                                             GlobalParams.M_R_XLength / 2.0);
  GridTools::transform(&Triangulation_Stretch_Computational_Rectangle, surface);

  Triangulation<2, 2>::active_cell_iterator

      cell = surface.begin_active(),
      endc = surface.end();

  double len = 2.0 / Layers;
  const double outside_max_edge_length = 1.0;
  const double inside_max_edge_length = 0.5;
  bool found_one = true;
  int refinements = 0;
  while (found_one) {
    found_one = false;
    cell = surface.begin_active();
    endc = surface.end();
    for (; cell != endc; ++cell) {
      Point<2> location_cell_center = cell->center();
      Point<3> location3D(location_cell_center[0], location_cell_center[1], 0);
      if (math_coordinate_in_waveguide(location3D)) {
        cell->set_material_id(1);
        bool first = false;
        bool second = false;
        for (unsigned int i = 0; i < 4; i++) {
          Point<2, double> dir(
              cell->line(i)->vertex(1)[0] - cell->line(i)->vertex(0)[0],
              cell->line(i)->vertex(1)[1] - cell->line(i)->vertex(0)[1]);
          double len = dir.norm();
          if (abs(dir[0]) > abs(dir[1])) {
            if (len > inside_max_edge_length) {
              first = true;
            }
          } else {
            if (len > inside_max_edge_length) {
              second = true;
            }
          }
        }
        if (first || second) {
          cell->set_refine_flag(RefinementCase<2>::cut_xy);
          found_one = true;
        }
      } else {
        cell->set_material_id(0);
        bool first = false;
        bool second = false;
        for (unsigned int i = 0; i < 4; i++) {
          Point<2, double> dir(
              cell->line(i)->vertex(1)[0] - cell->line(i)->vertex(0)[0],
              cell->line(i)->vertex(1)[1] - cell->line(i)->vertex(0)[1]);
          double len = dir.norm();
          if (abs(dir[0]) > abs(dir[1])) {
            if (len > outside_max_edge_length) {
              first = true;
            }
          } else {
            if (len > outside_max_edge_length) {
              second = true;
            }
          }
        }
        if (first || second) {
          cell->set_refine_flag(RefinementCase<2>::cut_xy);
          found_one = true;
        }
      }
    }
    surface.execute_coarsening_and_refinement();
    refinements++;
  }
  unsigned int layers = (unsigned int)std::round(10 * floor(GlobalParams.SystemLength / GlobalParams.NumberProcesses) /
      GlobalParams.M_W_Lambda);
  double length = GlobalParams.LayerThickness / (double)layers;
  deallog << "Concluded in " << refinements
          << " refinement steps. Extruding mesh. Building " << layers
          << " layers of thickness " << length << std::endl;

  // At this point the 2D surface Mesh is complete. Starting extrusion now.
  std::vector<double> slice_coords;
  double first_slice = GlobalParams.Minimum_Z +
                       GlobalParams.MPI_Rank * GlobalParams.LayerThickness;
  for (unsigned int i = 0; i <= layers; i++) {
    slice_coords.push_back(first_slice + ((double)i) * length);
  }
  GridGenerator::extrude_triangulation(surface, slice_coords, *in_tria);
  //  dealii::Tensor<1, 3, double> shift_vector;
  //  shift_vector[0] = 0;
  //  shift_vector[1] = 0;
  //  shift_vector[2] = GlobalParams.SystemLength / GlobalParams.NumberProcesses
  //  *
  //                        (GlobalParams.MPI_Rank + 0.5) -
  //                    GlobalParams.Minimum_Z;
  //  GridTools::shift(shift_vector, *in_tria);
  set_boundary_ids(*in_tria);
  deallog << "Done" << std::endl;
  deallog.pop();
  mesh_i(*in_tria, "grid_out.vtk");
}

bool SquareMeshGenerator::math_coordinate_in_waveguide(
    Point<3, double> in_position) const {
  return std::abs(in_position[0]) <
             (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) / 2.0 &&
         std::abs(in_position[1]) <
             (GlobalParams.M_C_Dim2In + GlobalParams.M_C_Dim2Out) / 2.0;
}

bool SquareMeshGenerator::phys_coordinate_in_waveguide(
    Point<3, double> in_position) const {
  std::cout
      << "NOT IMPLEMENTED: SquareMeshGenerator::phys_coordinate_in_waveguide"
      << std::endl;
  Point<3, double> temp = in_position;
  temp[1] -= ct->get_m(in_position[2]);
  double r = ct->get_r(in_position[2]);
  return (abs(temp[0]) < r && abs(temp[1]) < r);
  return false;
}

void SquareMeshGenerator::refine_global(Triangulation<3> *in_tria,
                                        unsigned int times) {
  in_tria->refine_global(times);
}

void SquareMeshGenerator::refine_proximity(Triangulation<3> *in_tria,
                                           unsigned int times, double factor) {
  double X = (GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In) *
             (1.0 + factor) / 2.0;
  double Y = (GlobalParams.M_C_Dim2Out + GlobalParams.M_C_Dim2In) *
             (1.0 + factor) / 2.0;
  for (unsigned int i = 0; i < times; i++) {
    cell = in_tria->begin_active();
    for (; cell != endc; ++cell) {
      if (std::abs(cell->center()[0]) < X && std::abs(cell->center()[1]) < Y) {
        cell->set_refine_flag();
      }
    }
    in_tria->execute_coarsening_and_refinement();
  }
}

void SquareMeshGenerator::refine_internal(Triangulation<3> *in_tria,
                                          unsigned int times) {
  for (unsigned int i = 0; i < times; i++) {
    cell = in_tria->begin_active();
    for (; cell != endc; ++cell) {
      if (math_coordinate_in_waveguide(cell->center())) {
        cell->set_refine_flag();
      }
    }
    in_tria->execute_coarsening_and_refinement();
  }
}

#endif
