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

  void set_boundary_ids();


};

#endif SquareMeshGenerator_h_
 
