#ifndef SquareMeshGenerator_h_
#define SquareMeshGenerator_h_

#include "./MeshGenerator.h"

/**
 * \class SquareMeshGenerator
 * \brief This class generates meshes, that are used to discretize a rectangular Waveguide. It is derived from MeshGenerator.
 *
 * The original intention of this project was to model tubular (or cylindrical) waveguides. The motivation behind this thought was the fact, that for this case the modes are known analytically. In applications however modes can be computed numerically and other shapes are easier to fabricate. For example square or rectangular waveguides can be printed in 3D on the scales we currently compute while tubular waveguides on that scale are not yet feasible.
 * \author Pascal Kraft
 * \date 28.11.2016
 */
class SquareMeshGenerator : public MeshGenerator {

  const double MaxDistX;
  const double MaxDistY;

  SquareMeshGenerator(SpaceTransformation * st);

  void set_boundary_ids();

  /**
   * This function is intended to execute a global refinement of the mesh. This means that every cell will be refined in every direction (effectively multiplying the number of DOFs by 8). This version is the most expensive refinement possible and should be used with caution.
   * \param times Number of refinement steps to be performed (gives us a multiplication of the number of degrees of freedom by \f$8^{times}\f$.
   */
  void refine_global(unsigned int times);

  /**
   * This function is intended to execute an internal refinement of the mesh. This means that every cell inside the waveguide will be refined in every direction. This method is rather cheap and only refines where the field is strong, however, the mesh outside the waveguide should not be too coarse to reduce numerical errors.
   * \param times Number of refinement steps to be performed.
   */
  void refine_internal(unsigned int times);

  /**
   * This function is intended to execute a refinement inside and near the waveguide boundary.
   * \param times Number of refinement steps to be performed.
   */
  void refine_proximity(unsigned int times);

  /**
   * This function checks if the given coordinate is inside the waveguide or not. The naming convention of physical and mathematical system find application. In this version, the waveguide has been transformed and the check for a tubal waveguide for example only checks if the radius of a given vector is below the average of input and output radius.
   * \params position This value gives us the location to check for.
   */
  bool math_coordinate_in_waveguide(Point<3> position);

  /**
   * This function checks if the given coordinate is inside the waveguide or not. The naming convention of physical and mathematical system find application. In this version, the waveguide is bent. If we are using a space transformation \f$f\f$ then this function is equal to math_coordinate_in_waveguide(f(x,y,z)).
   * \params position This value gives us the location to check for.
   */
  bool phys_coordinate_in_waveguide(Point<3> position);

  /**
   * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the x-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
   * \param position Stores the position in which to test for presence of a PML-Material.
   */
  bool  PML_in_X(Point<3> & position);

  /**
   * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the y-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
   * \param position Stores the position in which to test for presence of a PML-Material.
   */
  bool  PML_in_Y(Point<3> & position);

  /**
   * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the z-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
   * \param position Stores the position in which to test for presence of a PML-Material.
   */
  bool  PML_in_Z(Point<3> & position);

  /**
   * Similar to the PML_in_Z only this function is used to generate the artificial PML used in the Preconditioner. These Layers are not only situated at the surface of the computational domain but also inside it at the interfaces of Sectors.
   */
  bool Preconditioner_PML_in_Z(Point<3> &p, unsigned int block);

  /**
   * This function fulfills the same purpose as those with similar names but it is supposed to be used together with Preconditioner_PML_in_Z instead of the versions without "Preconditioner".
   */
  double Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block );

  /**
   * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
   * \param position Stores the position from which to calculate the distance to the PML-surface.
   */
  double  PML_X_Distance(Point<3> & position);

  /**
   * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
   * \param position Stores the position from which to calculate the distance to the PML-surface.
   */
  double  PML_Y_Distance(Point<3> & position);

  /**
   * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
   * \param position Stores the position from which to calculate the distance to the PML-surface.
   */
  double  PML_Z_Distance(Point<3> & position);

  /**
   * This function is a helper during distributed mesh generation.
   *
   */
  void set_boundary_ids() ;

  /**
   * This function takes a triangulation object and prepares it for the further computations. It is intended to encapsulate all related work and is explicitely not const.
   * \param in_tria The triangulation that is supposed to be prepared. All further information is derived from the parameter file and not given by parameters.
   */
  void prepare_triangulation(parallel::distributed::Triangulation<3> * in_tria)  ;



};

#endif SquareMeshGenerator_h_
 
