 
#ifndef SquareMeshGeneratorCppFlag
#define SquareMeshGeneratorCppFlag

#include "../Helpers/staticfunctions.cpp"

#include <deal.II/base/tensor.h>
#include <deal.II/base/std_cxx11/bind.h>
#include <deal.II/base/std_cxx11/array.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>

using namespace dealii;

SquareMeshGenerator::SquareMeshGenerator(SpaceTransformation * in_ct) :
    MaxDistX((GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*1.4/2.0),
    MaxDistY((GlobalParams.M_C_Dim2Out + GlobalParams.M_C_Dim2In)*1.4/2.0)
    {
  ct = in_ct;
  Layers = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  Point<3> origin(-1,-1,-1);
  std_cxx11::array< Tensor< 1, 3 >, 3 > edges;
  edges[0][0] = 2;
  edges[0][1] = 0;
  edges[0][2] = 0;

  edges[1][0] = 0;
  edges[1][1] = 2;
  edges[1][2] = 0;

  edges[2][0] = 0;
  edges[2][1] = 0;
  edges[2][2] = 2;

  std::vector<unsigned int> subs(3);
  subs[0] = 1;
  subs[1] = 1;
  subs[2] = Layers;

}

void SquareMeshGenerator::set_boundary_ids() {
  return;
}

void SquareMeshGenerator::prepare_triangulation(parallel::distributed::Triangulation<3> * in_tria){
    const std_cxx11::array< Tensor< 1, 3 >, 3 > edges2(edges);

    GridGenerator::subdivided_parallelepiped<3,3>(* in_tria, origin, edges2, subs, false);

    in_tria->repartition();
    //  parallel::shared::Triangulation<3>::active_cell_iterator cell, endc;
    in_tria->refine_global(3);

    in_tria->signals.post_refinement.connect
            (std_cxx11::bind (&Waveguide::set_boundary_ids,
                              std_cxx11::cref(*this),
                              std_cxx11::ref(in_tria)));


    in_tria->set_all_manifold_ids(0);

    GridTools::transform( &Triangulation_Stretch_to_circle , *in_tria);

    unsigned int man = 1;

    in_tria->set_manifold (man, round_description);

    in_tria->set_all_manifold_ids(0);

    parallel::shared::Triangulation<3>::active_cell_iterator

    cell = in_tria->begin_active(),
    endc = in_tria->end();

    int layers_per_sector = 4;
    layers_per_sector /= GlobalParams.R_Global;
    int reps = log2(layers_per_sector);
    if( layers_per_sector > 0 && pow(2,reps) != layers_per_sector) {
      std::cout << "The number of layers per sector has to be a power of 2. At least 2 layers are recommended for neccessary sparsity in the pattern for preconditioner to work." << std::endl;
      exit(0);
    }

    double len = 2.0 / Layers;

    cell = in_tria->begin_active();
    for (; cell!=endc; ++cell){
      unsigned int temp  = (int) std::floor((cell->center(true, false)[2] + 1.0)/len);
      if( temp >=  Layers || temp < 0) std::cout << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
    }

    GridTools::transform(& Triangulation_Stretch_X, * in_tria);
    GridTools::transform(& Triangulation_Stretch_Y, * in_tria);
    GridTools::transform(& Triangulation_Stretch_Computational_Radius, * in_tria);

    double MaxDistX = (GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*1.4/2.0;
    double MaxDistY = (GlobalParams.M_C_Dim2Out + GlobalParams.M_C_Dim2In)*1.4/2.0;
    for(int i = 0; i < GlobalParams.R_Local; i++) {
      cell = in_tria->begin_active();
      for (; cell!=endc; ++cell){
        if(std::abs(cell->center(true, false)[0])< MaxDistX && std::abs(cell->center(true, false)[1])< MaxDistY ){
          cell->set_refine_flag();
        }
      }
      in_tria->execute_coarsening_and_refinement();
    }

    for(int i = 0; i < GlobalParams.R_Interior; i++) {
      cell = in_tria->begin_active();
      for (; cell!=endc; ++cell){
        if( Distance2D(cell->center(true, false))< (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0)  {
          cell->set_refine_flag();
        }
      }
      in_tria->execute_coarsening_and_refinement();
    }


    // mesh_info(triangulation, solutionpath + "/grid" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".vtk");

    GridTools::transform(& Triangulation_Stretch_Z, * in_tria);


    GridTools::transform(& Triangulation_Shift_Z , * in_tria);

    z_min = 10000000.0;
    z_max = -10000000.0;
    cell = in_tria->begin_active();
    endc = in_tria->end();

    for (; cell!=endc; ++cell){
      if(cell->is_locally_owned()){
        for(int face = 0; face < 6; face++) {
          z_min = std::min(z_min, cell->face(face)->center()[2]);
          z_max = std::max(z_max, cell->face(face)->center()[2]);
        }
      }
    }


    // mesh_info(triangulation, solutionpath + "/grid" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".vtk");

    cell = in_tria->begin_active();
    endc = in_tria->end();
}

bool SquareMeshGenerator::math_coordinate_in_waveguide(Point<3,double> in_position) {
  return std::abs(cell->center(true, false)[0])< MaxDistX && std::abs(cell->center(true, false)[1])< MaxDistY ;
}

bool SquareMeshGenerator::phys_coordinate_in_waveguide(Point<3,double> in_position) {
  return  this->math_coordinate_in_waveguide(ct->phys_to_math(in_position));
}
#endif SquareMeshGeneratorCppFlag
