#ifndef RoundMeshGeneratorCppFlag
#define RoundMeshGeneratorCppFlag

#include "RoundMeshGenerator.h"
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
#include <deal.II/distributed/tria.h>
#include "../Helpers/staticfunctions.h"

using namespace dealii;

RoundMeshGenerator::RoundMeshGenerator(SpaceTransformation * in_ct) :
        MaxDistX((GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*1.4/2.0),
        MaxDistY((GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*1.4/2.0)
        {
  Layers = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  origin = Point<3>(-1,-1,-1);
  edges[0][0] = 2;
  edges[0][1] = 0;
  edges[0][2] = 0;

  edges[1][0] = 0;
  edges[1][1] = 2;
  edges[1][2] = 0;

  edges[2][0] = 0;
  edges[2][1] = 0;
  edges[2][2] = 2;

  // const std_cxx11::array< Tensor< 1, 3 >, 3 > edges2(edges);

  subs.push_back(1);
  subs.push_back(1);
  subs.push_back(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
  ct = in_ct;

}

RoundMeshGenerator::~RoundMeshGenerator() {

}

void RoundMeshGenerator::set_boundary_ids(parallel::distributed::Triangulation<3> & tria) const {

  parallel::distributed::Triangulation<3>::active_cell_iterator cell2 = tria.begin_active(),
  endc2 = tria.end();
  tria.set_all_manifold_ids(0);
  for (; cell2!=endc2; ++cell2){
    if (Distance2D(cell2->center() ) < 0.25 ) {
      cell2->set_all_manifold_ids(1);
      cell2->set_manifold_id(1);
    }
  }
  unsigned int man = 1;

  tria.set_manifold (man, round_description);

  cell2 = tria.begin_active();

  for (; cell2!=endc2; ++cell2){
    if(cell2->at_boundary()){
      for(int j = 0; j<6; j++){
        if(cell2->face(j)->at_boundary()){
          Point<3> ctr =cell2->face(j)->center(false, false);
          // if(System_Coordinate_in_Waveguide(ctr)){

          cell2->face(j)->set_all_boundary_ids(1);

          if(std::abs(ctr(2) - GlobalParams.M_R_ZLength/2.0 - GlobalParams.M_BC_Zplus*GlobalParams.SectorThickness) < 0.00001) {
            cell2->face(j)->set_all_boundary_ids(2);
          }
          if(std::abs(ctr(2) + GlobalParams.M_R_ZLength/2.0) < 0.00001) {
            cell2->face(j)->set_all_boundary_ids(3);
          }
        }
      }
    }
  }
}

void RoundMeshGenerator::prepare_triangulation(parallel::distributed::Triangulation<3> * in_tria){

  deallog.push("RoundMeshGenerator:prepare_triangulation");
  deallog << "Starting Mesh preparation"<<std::endl;

  const std_cxx11::array< Tensor< 1, 3 >, 3 > edges2(edges);

  GridGenerator::subdivided_parallelepiped<3,3>(* in_tria, origin, edges2, subs, false);

  in_tria->repartition();

  in_tria->signals.post_refinement.connect
            (std_cxx11::bind (&RoundMeshGenerator::set_boundary_ids,
                              std_cxx11::cref(*this),
                              std_cxx11::ref(*in_tria)));

  in_tria->refine_global(3);

  in_tria->set_all_manifold_ids(0);

  GridTools::transform( &Triangulation_Stretch_to_circle , *in_tria);

  unsigned int man = 1;

  in_tria->set_manifold (man, round_description);

  in_tria->set_all_manifold_ids(0);
  cell = in_tria->begin_active();

  endc = in_tria->end();
  for (; cell!=endc; ++cell){
    if (Distance2D(cell->center() ) < 0.25 ) {
      cell->set_all_manifold_ids(1);
      cell->set_manifold_id(1);
    }
  }


  in_tria->set_manifold (man, round_description);

  in_tria->set_all_manifold_ids(0);
  cell = in_tria->begin_active();
  endc = in_tria->end();
  for (; cell!=endc; ++cell){
    if (Distance2D(cell->center() ) < 0.25 ) {
      cell->set_all_manifold_ids(1);
      cell->set_manifold_id(1);
    }
  }


  in_tria->set_manifold (man, round_description);
  parallel::distributed::Triangulation<3>::active_cell_iterator
  cell = in_tria->begin_active(),
  endc = in_tria->end();

  double len = 2.0 / Layers;

  cell = in_tria->begin_active();
  for (; cell!=endc; ++cell){
    int temp  = (int) std::floor((cell->center(true, false)[2] + 1.0)/len);
    if( temp >=  (int)Layers || temp < 0) std::cout << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
  }

  GridTools::transform(& Triangulation_Stretch_X, * in_tria);
  GridTools::transform(& Triangulation_Stretch_Y, * in_tria);
  GridTools::transform(& Triangulation_Stretch_Computational_Radius, * in_tria);

  if(GlobalParams.R_Global > 0) {
    in_tria->refine_global(GlobalParams.R_Global);
  }

  double MaxDistFromBoundary = (GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*1.4/2.0;
  for(int i = 0; i < GlobalParams.R_Local; i++) {
    cell = in_tria->begin_active();
    for (; cell!=endc; ++cell){
      if(std::abs(Distance2D(cell->center(true, false)) - (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0 ) < MaxDistFromBoundary) {
        cell->set_refine_flag();
      }
    }
    in_tria->execute_coarsening_and_refinement();
    MaxDistFromBoundary = (MaxDistFromBoundary + ((GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)/2.0))/2.0 ;
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



    cell = in_tria->begin_active();
    endc = in_tria->end();

    /// mesh_info(*in_tria, "Output"+std::to_string(GlobalParams.MPI_Rank)+".vtk");

    set_boundary_ids(*in_tria);

    deallog << "Done" <<std::endl;
    deallog.pop();
}

bool RoundMeshGenerator::math_coordinate_in_waveguide(Point<3,double> in_position) const {
  return Distance2D(in_position)< (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0 ;
}

bool RoundMeshGenerator::phys_coordinate_in_waveguide(Point<3,double> in_position) const {
  Point<3,double> temp = in_position;
  temp[1] -= ct->get_m(in_position[2]);
  double r = ct->get_r(in_position[2]);
  return (abs(temp[0]) < r && abs(temp[1]) < r );
}

void RoundMeshGenerator::refine_global(parallel::distributed::Triangulation<3> * in_tria, unsigned int times) {
  in_tria->refine_global(times);
}

void RoundMeshGenerator::refine_proximity(parallel::distributed::Triangulation<3> * in_tria, unsigned int times, double factor) {
  for (unsigned int t = 0; t < times; t++) {
    double R = (GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*(1.0 + factor)/2.0;
    cell = in_tria->begin_active();
    for (; cell!=endc; ++cell){
      if(Distance2D(cell->center(true, false))< R) {
        cell->set_refine_flag();
      }
    }
    in_tria->execute_coarsening_and_refinement();
  }
}

void RoundMeshGenerator::refine_internal(parallel::distributed::Triangulation<3> * in_tria, unsigned int times) {
  for(unsigned int i = 0; i < times; i++) {
    cell = in_tria->begin_active();
    for (; cell!=endc; ++cell){
      if(math_coordinate_in_waveguide(cell->center())) {
        cell->set_refine_flag();
      }
    }
    in_tria->execute_coarsening_and_refinement();
  }
}
#endif
