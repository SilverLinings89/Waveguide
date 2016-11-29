#ifndef MESHGENERATOR_H_
#define MESHGENERATOR_H_

#include "../Helpers/Parameters.h"
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include "../SpaceTransformations/SpaceTransformation.h"

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
  parallel::distributed::Triangulation<3> * p_triangulation;
  Parameters * p;
  parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
  SpaceTransformation * ct;
  unsigned int Layers;

public:

  MeshGenerator(SpaceTransformation & in_ct);

  /**
   * This function is intended to execute a global refinement of the mesh. This means that every cell will be refined in every direction (effectively multiplying the number of DOFs by 8). This version is the most expensive refinement possible and should be used with caution.
   * \param times Number of refinement steps to be performed (gives us a multiplication of the number of degrees of freedom by \f$8^{times}\f$.
   */
  virtual void refine_global(unsigned int times);

  /**
   * This function is intended to execute an internal refinement of the mesh. This means that every cell inside the waveguide will be refined in every direction. This method is rather cheap and only refines where the field is strong, however, the mesh outside the waveguide should not be too coarse to reduce numerical errors.
   * \param times Number of refinement steps to be performed.
   */
  virtual void refine_internal(unsigned int times);

  /**
   * This function is intended to execute a refinement inside and near the waveguide boundary.
   * \param times Number of refinement steps to be performed.
   */
  virtual void refine_proximity(unsigned int times);

  /**
   * This function checks if the given coordinate is inside the waveguide or not. The naming convention of physical and mathematical system find application. In this version, the waveguide has been transformed and the check for a tubal waveguide for example only checks if the radius of a given vector is below the average of input and output radius.
   * \params position This value gives us the location to check for.
   */
  virtual bool math_coordinate_in_waveguide(Point<3> position);

  /**
   * This function checks if the given coordinate is inside the waveguide or not. The naming convention of physical and mathematical system find application. In this version, the waveguide is bent. If we are using a space transformation \f$f\f$ then this function is equal to math_coordinate_in_waveguide(f(x,y,z)).
   * \params position This value gives us the location to check for.
   */
  virtual bool phys_coordinate_in_waveguide(Point<3> position);

  /**
   * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the x-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
   * \param position Stores the position in which to test for presence of a PML-Material.
   */
  virtual bool  PML_in_X(Point<3> & position);
  /**
   * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the y-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
   * \param position Stores the position in which to test for presence of a PML-Material.
   */
  virtual bool  PML_in_Y(Point<3> & position);
  /**
   * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the z-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
   * \param position Stores the position in which to test for presence of a PML-Material.
   */
  virtual bool  PML_in_Z(Point<3> & position);

  /**
   * Similar to the PML_in_Z only this function is used to generate the artificial PML used in the Preconditioner. These Layers are not only situated at the surface of the computational domain but also inside it at the interfaces of Sectors.
   */
  virtual bool Preconditioner_PML_in_Z(Point<3> &p, unsigned int block);

  /**
   * This function fulfills the same purpose as those with similar names but it is supposed to be used together with Preconditioner_PML_in_Z instead of the versions without "Preconditioner".
   */
  virtual double Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block );

  /**
   * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
   * \param position Stores the position from which to calculate the distance to the PML-surface.
   */
  virtual double  PML_X_Distance(Point<3> & position);
  /**
   * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
   * \param position Stores the position from which to calculate the distance to the PML-surface.
   */

  virtual double  PML_Y_Distance(Point<3> & position);
  /**
   * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
   * \param position Stores the position from which to calculate the distance to the PML-surface.
   */
  virtual double  PML_Z_Distance(Point<3> & position);

  /**
   * This function is a helper during distributed mesh generation.
   *
   */
  virtual void set_boundary_ids() ;
};


#endif MESHGENERATOR_H_
