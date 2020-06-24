#include "SquareMeshGenerator.h"
#include <deal.II/base/multithread_info.h>
#include <array>
#include <functional>
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

SquareMeshGenerator::SquareMeshGenerator() {}

SquareMeshGenerator::~SquareMeshGenerator() {}

// This function returns a direction - as specified in the Direction enum in the
// GeometryManager header - for a given cell and face. it only takes a vector as
// an argument which is the vector pointing from cell center to face center. It
// then derives which bound-id to give that face (i.e. 0 if it is the boundary
// in the minus X direction etc.)

unsigned int SquareMeshGenerator::getDominantComponentAndDirection(
    Point<3> in_dir) const {
  unsigned int comp = 0;
  if (std::abs(in_dir[0]) >= std::abs(in_dir[1]) &&
      std::abs(in_dir[0]) >= std::abs(in_dir[2])) {
    if (in_dir[0] < 0) {
      comp = 0;
    } else {
      comp = 1;
    }
  } else {
    if (std::abs(in_dir[1]) >= std::abs(in_dir[2])) {
      if (in_dir[1] < 0) {
        comp = 2;
      } else {
        comp = 3;
      }
    } else {
      if (in_dir[2] < 0) {
        comp = 4;
      } else {
        comp = 5;
      }
    }
  }
  return comp;
}

void SquareMeshGenerator::set_boundary_ids(Triangulation<3> &tria) const {
  Triangulation<3>::active_cell_iterator cell2 = tria.begin_active(),
                                         endc2 = tria.end();
  tria.set_all_manifold_ids(0);
  for (; cell2 != endc2; ++cell2) {
    if (cell2->at_boundary()) {
      for (int j = 0; j < 6; j++) {
        Point<3> ctr = cell2->face(j)->center();
        if (cell2->face(j)->at_boundary()) {
          dealii::Point<3, double> d2 = -cell2->center() + ctr;
          unsigned int dominant_direction =
              getDominantComponentAndDirection(d2);
          cell2->face(j)->set_all_boundary_ids(dominant_direction);
        }
      }
    }
  }
}

void SquareMeshGenerator::prepare_triangulation(Triangulation<3, 3> *in_tria) {
  GridGenerator::hyper_cube(*in_tria, -1.0, 1.0, false);
  GridTools::transform(&Triangulation_Shit_To_Local_Geometry, *in_tria);
  set_boundary_ids(*in_tria);

  in_tria->signals.post_refinement.connect(
      std::bind(&SquareMeshGenerator::set_boundary_ids,
          std::cref(*this), std::ref(*in_tria)));

  refine_triangulation_iteratively(in_tria);

  set_boundary_ids(*in_tria);
}

void SquareMeshGenerator::refine_triangulation_iteratively(
    Triangulation<3, 3> *in_tria) {
  bool refinement_required = true;
  Triangulation<3>::active_cell_iterator cell = in_tria->begin_active(),
                                         endc = in_tria->end();
  while (refinement_required && in_tria->n_active_cells() < 100000) {
    refinement_required = false;
    for (cell = in_tria->begin_active(); cell != endc; ++cell) {
      refinement_required = check_and_mark_one_cell_for_refinement(cell);
    }
    in_tria->execute_coarsening_and_refinement();
  }
}

bool SquareMeshGenerator::check_and_mark_one_cell_for_refinement(
    Triangulation<3>::active_cell_iterator cell) {
  bool found_refinable_cell = false;
  double h_max = hmax_for_cell_center(cell->center());
  for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
    for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
      dealii::Tensor<1, 3> dir =
          cell->face(i)->line(j)->vertex(0) - cell->face(i)->line(j)->vertex(1);
      dir[0] = std::abs(dir[0]);
      dir[1] = std::abs(dir[1]);
      dir[2] = std::abs(dir[2]);
      if (cell->face(i)->line(j)->measure() > h_max) {
        found_refinable_cell = true;
        if (dir[0] > dir[1] && dir[0] > dir[2]) {
          cell->set_refine_flag(RefinementCase<3>::cut_x);
        } else {
          if (dir[1] > dir[2]) {
            cell->set_refine_flag(RefinementCase<3>::cut_y);
          } else {
            cell->set_refine_flag(RefinementCase<3>::cut_z);
          }
        }
      }
    }
  }

  return found_refinable_cell;
}
