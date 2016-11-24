#ifndef MESHGENERATOR_H_
#define MESHGENERATOR_H_

#include "../Helpers/Parameters.h"
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include "../SpaceTransformations/SpaceTransformation.h"

using namespace dealii;

class MeshGenerator {
  parallel::distributed::Triangulation<3> * p_triangulation;
  Parameters * p;
  parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
  unsigned int Layers;

  MeshGenerator(SpaceTransformation & ct);

  void refine_global(unsigned int times);

  void refine_internal(unsigned int times);

  void refine_exxternal(unsigned int times);

  bool math_coordinate_in_waveguide(Point<3> position);

  bool phys_coordinate_in_waveguide(Point<3> position);

  void set_boundary_ids() const ;
};






#endif MESHGENERATOR_H_
