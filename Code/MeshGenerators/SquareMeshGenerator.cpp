 
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

void SquareMeshGenerator::SquareMeshGenerator(SpaceTransformation & in_ct) {
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

  const std_cxx11::array< Tensor< 1, 3 >, 3 > edges2(edges);

  std::vector<unsigned int> subs(3);
  subs[0] = 1;
  subs[1] = 1;
  subs[2] = Layers;
  GridGenerator::subdivided_parallelepiped<3,3>(* p_triangulation,origin, edges2, subs, false);


  p_triangulation->repartition();
  //  parallel::shared::Triangulation<3>::active_cell_iterator cell, endc;
  p_triangulation->refine_global(3);

  p_triangulation->signals.post_refinement.connect
          (std_cxx11::bind (&this->set_boundary_ids,
                            std_cxx11::cref(*this),
                            std_cxx11::ref(p_triangulation)));


  p_triangulation->set_all_manifold_ids(0);

  GridTools::transform( &Triangulation_Stretch_to_circle , *p_triangulation);

  p_triangulation->set_all_manifold_ids(0);

  parallel::shared::Triangulation<3>::active_cell_iterator

  cell = p_triangulation->begin_active(),
  endc = p_triangulation->end();
  for ( ; cell!=endc; ++cell) {
    //cell->set_subdomain_id(0);
  }

  int layers_per_sector = 4;
  layers_per_sector /= GlobalParams.PRM_R_Global;
  int reps = log2(layers_per_sector);
  if( layers_per_sector > 0 && pow(2,reps) != layers_per_sector) {
    std::cout << "The number of layers per sector has to be a power of 2. At least 2 layers are recommended for neccessary sparsity in the pattern for preconditioner to work." << std::endl;
    exit(0);
  }

  double len = 2.0 / Layers;

  cell = p_triangulation->begin_active();
  for (; cell!=endc; ++cell){
    unsigned int temp  = (int) std::floor((cell->center(true, false)[2] + 1.0)/len);
    if( temp >=  Layers || temp < 0) std::cout << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
  }

  GridTools::transform(& Triangulation_Stretch_X, * p_triangulation);
  GridTools::transform(& Triangulation_Stretch_Y, * p_triangulation);
  GridTools::transform(& Triangulation_Stretch_Computational_Radius, * p_triangulation);

  if(GlobalParams.PRM_D_Refinement == "global"){
    p_triangulation->refine_global (GlobalParams.PRM_D_XY);
  } else {

  double MaxDistFromBoundary = (GlobalParams.PRM_M_C_RadiusOut + GlobalParams.PRM_M_C_RadiusIn)*1.4/2.0;
  for(int i = 0; i < GlobalParams.PRM_R_Semi; i++) {
    cell = p_triangulation->begin_active();
    for (; cell!=endc; ++cell){
      if(std::abs(Distance2D(cell->center(true, false)) - (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0 ) < MaxDistFromBoundary) {
        cell->set_refine_flag();
      }
    }
    p_triangulation->execute_coarsening_and_refinement();
    MaxDistFromBoundary = (MaxDistFromBoundary + ((GlobalParams.PRM_M_C_RadiusOut + GlobalParams.PRM_M_C_RadiusIn)/2.0))/2.0 ;
  }

  for(int i = 0; i < GlobalParams.PRM_R_Internal; i++) {
    cell = p_triangulation->begin_active();
    for (; cell!=endc; ++cell){
      if( Distance2D(cell->center(true, false))< (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/2.0)  {
        cell->set_refine_flag();
      }
    }
    p_triangulation->execute_coarsening_and_refinement();
  }
}

  // mesh_info(triangulation, solutionpath + "/grid" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".vtk");

  GridTools::transform(& Triangulation_Stretch_Z, * p_triangulation);


  GridTools::transform(& Triangulation_Shift_Z , * p_triangulation);

  GlobalParams.z_min = 10000000.0;
  GlobalParams.z_max = -10000000.0;
  cell = p_triangulation->begin_active();
  endc = p_triangulation->end();

  for (; cell!=endc; ++cell){
    if(cell->is_locally_owned()){
      for(int face = 0; face < 6; face++) {
        GlobalParams.z_min = std::min(GlobalParams.z_min, cell->face(face)->center()[2]);
        GlobalParams.z_max = std::max(GlobalParams.z_max, cell->face(face)->center()[2]);
      }
    }
  }

  if(GlobalParams.z_min < (-GlobalParams.PRM_M_R_ZLength/2.0 + 0.00001) && GlobalParams.z_max >= -GlobalParams.PRM_M_R_ZLength/2.0 ) {
    GlobalParams.evaluate_in = true;
  } else {
    GlobalParams.evaluate_in = false;
  }

  if(GlobalParams.z_min <= GlobalParams.PRM_M_R_ZLength/(2.0) && GlobalParams.z_max >= GlobalParams.PRM_M_R_ZLength/2.0 ) {
    GlobalParams.evaluate_out = true;
  } else {
    GlobalParams.evaluate_out = false;
  }

  GlobalParams.z_evaluate = (GlobalParams.z_min + GlobalParams.z_max)/2.0;
  // mesh_info(triangulation, solutionpath + "/grid" + static_cast<std::ostringstream*>( &(std::ostringstream() << GlobalParams.MPI_Rank) )->str() + ".vtk");

  cell = p_triangulation->begin_active();
  endc = p_triangulation->end();
}

void SquareMeshGenerator::set_boundary_ids() {
  return;
}

#endif SquareMeshGeneratorCppFlag
