#ifndef SquareMeshGenerator_h_
#define SquareMeshGenerator_h_

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <array>
#include <vector>
#include "../SpaceTransformations/SpaceTransformation.h"
#include "./SquareMeshGenerator.h"

/**
 * \class SquareMeshGenerator
 * \brief This class generates meshes, that are used to discretize a rectangular
 * Waveguide. It is derived from MeshGenerator.
 *
 * The original intention of this project was to model tubular (or cylindrical)
 * waveguides. The motivation behind this thought was the fact, that for this
 * case the modes are known analytically. In applications however modes can be
 * computed numerically and other shapes are easier to fabricate. For example
 * square or rectangular waveguides can be printed in 3D on the scales we
 * currently compute while tubular waveguides on that scale are not yet
 * feasible. \author Pascal Kraft \date 28.11.2016
 */
class SquareMeshGenerator {

public:
    SquareMeshGenerator();

    ~SquareMeshGenerator();

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
     * This function takes a triangulation object and prepares it for the further
     * computations. It is intended to encapsulate all related work and is
     * explicitely not const. \param in_tria The triangulation that is supposed to
     * be prepared. All further information is derived from the parameter file and
     * not given by parameters.
     */
    void prepare_triangulation(Triangulation<3> *in_tria);

    unsigned int getDominantComponentAndDirection(
            dealii::Point<3, double> in_dir) const;

    void set_boundary_ids(Triangulation<3> &) const;

    Triangulation<3>::active_cell_iterator cell, endc;

    void refine_triangulation_iteratively(Triangulation<3, 3> *);

    bool check_and_mark_one_cell_for_refinement(
            Triangulation<3>::active_cell_iterator);
};

#endif
