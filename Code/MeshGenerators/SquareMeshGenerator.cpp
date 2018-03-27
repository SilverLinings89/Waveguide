#ifndef SquareMeshGeneratorCppFlag
#define SquareMeshGeneratorCppFlag

#include "SquareMeshGenerator.h"
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
#include <deal.II/grid/grid_out.h>
#include "../Helpers/staticfunctions.h"

using namespace dealii;

SquareMeshGenerator::SquareMeshGenerator(SpaceTransformation * in_ct) :
    MaxDistX((GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*1.4/2.0),
    MaxDistY((GlobalParams.M_C_Dim2Out + GlobalParams.M_C_Dim2In)*1.4/2.0)
    {
  ct = in_ct;
  Layers = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  origin = Point<3>(-1,-1,-1);
  p1 = Point<3>(-1,-1,-1);
  p2 = Point<3>(1,1,1);
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

SquareMeshGenerator::~SquareMeshGenerator() {

}

void SquareMeshGenerator::set_boundary_ids(parallel::distributed::Triangulation<3> & tria) const {

	parallel::distributed::Triangulation<3>::active_cell_iterator cell2 = tria.begin_active(),
	endc2 = tria.end();
	tria.set_all_manifold_ids(0);
    double min_z = 100000.0;
    double max_z = -100000.0;
    parallel::distributed::Triangulation<3>::cell_iterator boundary_searcher_begin = tria.begin();
    parallel::distributed::Triangulation<3>::cell_iterator boundary_searcher_end = tria.end();
    for (; boundary_searcher_begin != boundary_searcher_end; boundary_searcher_begin ++) {
      for (unsigned int i=0; i<GeometryInfo<3>::vertices_per_cell; ++i)
      {
        Point<3> &v = boundary_searcher_begin->vertex(i);
        if( v(2) < min_z) {
          min_z = v(2);
        }
        if( v(2) > max_z) {
          max_z = v(2);
        }
      }
    }
	for (; cell2!=endc2; ++cell2){
	 if(cell2->at_boundary()){
		 for(int j = 0; j<6; j++){
			 Point<3> ctr =cell2->face(j)->center();
			 if(cell2->face(j)->at_boundary()){
				 cell2->face(j)->set_all_boundary_ids(1);

				 if(std::abs(ctr(2) - max_z) < 0.00001) {
					 cell2->face(j)->set_all_boundary_ids(2);
				 }
				 if(std::abs(ctr(2) - min_z) < 0.00001) {
					 cell2->face(j)->set_all_boundary_ids(3);
				 }

			 }
		 }
	 }
	}
}

void SquareMeshGenerator::prepare_triangulation(parallel::distributed::Triangulation<3> * in_tria){

  deallog.push("SquareMeshGenerator:prepare_triangulation");
  deallog << "Starting Mesh preparation"<<std::endl;

  const std_cxx11::array< Tensor< 1, 3 >, 3 > edges2(edges);

  GridGenerator::subdivided_parallelepiped<3,3>(* in_tria, origin, edges2, subs, false);

  in_tria->repartition();

  in_tria->signals.post_refinement.connect
            (std_cxx11::bind (& SquareMeshGenerator::set_boundary_ids,
                              std_cxx11::cref(*this),
                              std_cxx11::ref(*in_tria)));

  in_tria->refine_global(3);

  GridTools::transform(& Triangulation_Stretch_Computational_Rectangle, * in_tria);

  parallel::distributed::Triangulation<3>::active_cell_iterator

  cell = in_tria->begin_active(),
  endc = in_tria->end();

  double len = 2.0 / Layers;

  cell = in_tria->begin_active();
  for (; cell!=endc; ++cell){
    int temp  = (int) std::floor((cell->center()[2] + 1.0)/len);
    if( temp >=  (int)Layers || temp < 0) std::cout << "Critical Error in Mesh partitioning. See make_grid! Solvers might not work." << std::endl;
  }

  if(GlobalParams.R_Global > 0) {
      in_tria->refine_global(GlobalParams.R_Global);
  }


  double MaxDistX = (GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*1.4/2.0;
  double MaxDistY = (GlobalParams.M_C_Dim2Out + GlobalParams.M_C_Dim2In)*1.4/2.0;
  for(int i = 0; i < GlobalParams.R_Local; i++) {
    cell = in_tria->begin_active();
    for (; cell!=endc; ++cell){
      if(std::abs(cell->center()[0])< MaxDistX && std::abs(cell->center()[1])< MaxDistY ){
        cell->set_refine_flag();
      }
    }
    in_tria->execute_coarsening_and_refinement();
    MaxDistX = (GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*1.4/2.0;
    MaxDistY = (GlobalParams.M_C_Dim2Out + GlobalParams.M_C_Dim2In)*1.4/2.0;

  }

  for(int i = 0; i < GlobalParams.R_Interior; i++) {
    cell = in_tria->begin_active();
    for (; cell!=endc; ++cell){
      if( std::abs(cell->center()[0]) < (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0 && std::abs(cell->center()[1]) < (GlobalParams.M_C_Dim2In + GlobalParams.M_C_Dim2Out)/2.0)  {
        cell->set_refine_flag();
      }
    }
    in_tria->execute_coarsening_and_refinement();
  }

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

  set_boundary_ids(*in_tria);

  deallog << "Done" <<std::endl;
  deallog.pop();

}

bool SquareMeshGenerator::math_coordinate_in_waveguide(Point<3,double> in_position) const  {
  return std::abs(in_position[0])< (GlobalParams.M_C_Dim1In+GlobalParams.M_C_Dim1Out)/2.0 && std::abs(in_position[1])< (GlobalParams.M_C_Dim2In+GlobalParams.M_C_Dim2Out)/2.0;
}

bool SquareMeshGenerator::phys_coordinate_in_waveguide(Point<3,double> in_position) const {
  std::cout<< "NOT IMPLEMENTED: SquareMeshGenerator::phys_coordinate_in_waveguide"<<std::endl;
  Point<3,double> temp = in_position;
  temp[1] -= ct->get_m(in_position[2]);
  double r = ct->get_r(in_position[2]);
  return (abs(temp[0]) < r && abs(temp[1]) < r );
  return false;
}

void SquareMeshGenerator::refine_global(parallel::distributed::Triangulation<3> * in_tria, unsigned int times) {
  in_tria->refine_global(times);
}

void SquareMeshGenerator::refine_proximity(parallel::distributed::Triangulation<3> * in_tria, unsigned int times, double factor) {
  double X = (GlobalParams.M_C_Dim1Out + GlobalParams.M_C_Dim1In)*(1.0 + factor)/2.0;
  double Y = (GlobalParams.M_C_Dim2Out + GlobalParams.M_C_Dim2In)*(1.0 + factor)/2.0;
  for(unsigned int i = 0; i < times; i++) {
    cell = in_tria->begin_active();
    for (; cell!=endc; ++cell){
      if(std::abs(cell->center()[0])< X && std::abs(cell->center()[1])< Y) {
        cell->set_refine_flag();
      }
    }
    in_tria->execute_coarsening_and_refinement();
  }
}

void SquareMeshGenerator::refine_internal(parallel::distributed::Triangulation<3> * in_tria, unsigned int times) {
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
