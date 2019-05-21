#ifndef RoundMeshGenerator_H_
#define RoundMeshGenerator_H_

#include <deal.II/base/point.h>
#include <deal.II/distributed/tria.h>
#include <array>
#include <vector>
#include "../SpaceTransformations/SpaceTransformation.h"
#include "./RoundMeshGenerator.h"
#include "MeshGenerator.h"
/**
 * \class RoundMeshGenerator
 * \brief This class generates meshes, that are used to discretize a rectangular
 * Waveguide. It is derived from MeshGenerator.
 *
 * This Generator creates a mesh around a cylindrical waveguide. It should be
 * used in conjunction with a SpaceTransformation, which uses a circular shape
 * of the waveguide and an appropriate distribution of the DOFs. \author Pascal
 * Kraft \date 28.11.2016
 */
class RoundMeshGenerator : public MeshGenerator {
  Triangulation<3>::active_cell_iterator cell, endc;
  SpaceTransformation* ct;
  unsigned int Layers;
  Point<3> origin;
  std_cxx11::array<Tensor<1, 3>, 3> edges;
  std::vector<unsigned int> subs;

  const double MaxDistX;
  const double MaxDistY;

 public:
  RoundMeshGenerator(SpaceTransformation* in_ct);

  ~RoundMeshGenerator();

  /**
   * This function is intended to execute a global refinement of the mesh. This
   * means that every cell will be refined in every direction (effectively
   * multiplying the number of DOFs by 8). This version is the most expensive
   * refinement possible and should be used with caution. \param times Number of
   * refinement steps to be performed (gives us a multiplication of the number
   * of degrees of freedom by \f$8^{times}\f$.
   */
  void refine_global(Triangulation<3>* in_tria, unsigned int times);

  /**
   * This function is intended to execute an internal refinement of the mesh.
   * This means that every cell inside the waveguide will be refined in every
   * direction. This method is rather cheap and only refines where the field is
   * strong, however, the mesh outside the waveguide should not be too coarse to
   * reduce numerical errors. \param times Number of refinement steps to be
   * performed.
   */
  void refine_internal(Triangulation<3>* in_tria, unsigned int times);

  /**
   * This function is intended to execute a refinement inside and near the
   * waveguide boundary. \param times Number of refinement steps to be
   * performed.
   */
  void refine_proximity(Triangulation<3>* in_tria, unsigned int times,
                        double factor);

  /**
   * This function checks if the given coordinate is inside the waveguide or
   * not. The naming convention of physical and mathematical system find
   * application. In this version, the waveguide has been transformed and the
   * check for a tubal waveguide for example only checks if the radius of a
   * given vector is below the average of input and output radius. \params
   * position This value gives us the location to check for.
   */
  bool math_coordinate_in_waveguide(Point<3> position) const;

  /**
   * This function checks if the given coordinate is inside the waveguide or
   * not. The naming convention of physical and mathematical system find
   * application. In this version, the waveguide is bent. If we are using a
   * space transformation \f$f\f$ then this function is equal to
   * math_coordinate_in_waveguide(f(x,y,z)). \params position This value gives
   * us the location to check for.
   */
  bool phys_coordinate_in_waveguide(Point<3> position) const;

  /**
   * This function is a helper during distributed mesh generation.
   *
   */
  void set_boundary_ids(Triangulation<3>& tria) const;

  /**
   * This function takes a triangulation object and prepares it for the further
   * computations. It is intended to encapsulate all related work and is
   * explicitely not const. \param in_tria The triangulation that is supposed to
   * be prepared. All further information is derived from the parameter file and
   * not given by parameters.
   */
  void prepare_triangulation(Triangulation<3>* in_tria);
};

#endif
