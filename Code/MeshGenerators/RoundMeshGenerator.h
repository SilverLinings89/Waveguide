#ifndef RoundMeshGenerator_H_
#define RoundMeshGenerator_H_

#include "./MeshGenerator.h"

/**
 * \class RoundMeshGenerator
 * \brief This class generates meshes, that are used to discretize a rectangular Waveguide. It is derived from MeshGenerator.
 *
 * This Generator creates a mesh around a cylindrical waveguide. It should be used in conjunction with a SpaceTransformation, which uses a circular shape of the waveguide and an appropriate distribution of the DOFs.
 * \author Pascal Kraft
 * \date 28.11.2016
 */
class RoundMeshGenerator : public MeshGenerator {

  void set_boundary_ids();



};

#endif RoundMeshGenerator_H_
