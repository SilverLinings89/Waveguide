#ifndef MESHGENERATOR_H_
#define MESHGENERATOR_H_

#include "../Helpers/Parameters.h"
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include "../SpaceTransformations/SpaceTransformation.h"
#include <deal.II/base/tensor.h>

#include <deal.II/base/std_cxx11/array.h>
using namespace dealii;


/**
 * \class MeshGenerator
 * \brief This is an interface for all the mesh generators in the project describing its role and functionality.
 *
 * Since different shapes of waveguides (in the xz-plane) are interesting in application settings, we wish to introduce a mechanism to model this fact. Therefore all functionality related to the shape of the waveguide are encapsulated of specific objects which do all the heavy lifting. The problem si the fact that a rectangular geometry has more DOFs then a square or circular one (radius versus width and height). This has implications on the space transformation and the optimization scheme and earlier versions were running the risk of getting too flawed by implementing loads of different case models over and over again. This structure leads to a higher readibility of the code and reduses its error-proneness.
 * \author Pascal Kraft
 * \date 28.11.2016
 */
class MeshGenerator {

public:

  parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
  SpaceTransformation * ct;
  unsigned int Layers;
  Point<3> origin;
  std_cxx11::array< Tensor< 1, 3 >, 3 > edges;
  std::vector<unsigned int> subs;

  double z_min, z_max;

  virtual MeshGenerator(SpaceTransformation & in_ct) = 0;

  virtual ~MeshGenerator() =0;

  /**
   * This function is intended to execute a global refinement of the mesh. This means that every cell will be refined in every direction (effectively multiplying the number of DOFs by 8). This version is the most expensive refinement possible and should be used with caution.
   * \param times Number of refinement steps to be performed (gives us a multiplication of the number of degrees of freedom by \f$8^{times}\f$.
   */
  virtual void refine_global(unsigned int times)=0;

  /**
   * This function is intended to execute an internal refinement of the mesh. This means that every cell inside the waveguide will be refined in every direction. This method is rather cheap and only refines where the field is strong, however, the mesh outside the waveguide should not be too coarse to reduce numerical errors.
   * \param times Number of refinement steps to be performed.
   */
  virtual void refine_internal(unsigned int times)=0;

  /**
   * This function is intended to execute a refinement inside and near the waveguide boundary.
   * \param times Number of refinement steps to be performed.
   */
  virtual void refine_proximity(unsigned int times)=0;

  /**
   * This function checks if the given coordinate is inside the waveguide or not. The naming convention of physical and mathematical system find application. In this version, the waveguide has been transformed and the check for a tubal waveguide for example only checks if the radius of a given vector is below the average of input and output radius.
   * \params position This value gives us the location to check for.
   */
  virtual bool math_coordinate_in_waveguide(Point<3> position)=0;

  /**
   * This function checks if the given coordinate is inside the waveguide or not. The naming convention of physical and mathematical system find application. In this version, the waveguide is bent. If we are using a space transformation \f$f\f$ then this function is equal to math_coordinate_in_waveguide(f(x,y,z)).
   * \params position This value gives us the location to check for.
   */
  virtual bool phys_coordinate_in_waveguide(Point<3> position)=0;

  /**
   * This function takes a triangulation object and prepares it for the further computations. It is intended to encapsulate all related work and is explicitely not const.
   * \param in_tria The triangulation that is supposed to be prepared. All further information is derived from the parameter file and not given by parameters.
   */
  virtual void prepare_triangulation(parallel::distributed::Triangulation<3> * in_tria) =0 ;

};


#endif MESHGENERATOR_H_
