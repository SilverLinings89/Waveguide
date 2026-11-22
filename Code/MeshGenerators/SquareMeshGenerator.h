#pragma once
/**
 * @file SquareMeshGenerator.h
 * @author Pascal Kraft
 * @version 0.1
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <array>
#include <vector>
#include "./SquareMeshGenerator.h"
#include "../Core/Types.h"

/**
 * \class SquareMeshGenerator
 * \brief This class generates meshes, that are used to discretize a rectangular
 * Waveguide. 
 *
 * Important: This is legacy code. This is currently not required.
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
  /** @brief Does nothing. */
  SquareMeshGenerator();

  /** @brief Does nothing. */
  ~SquareMeshGenerator();

  /**
   * This function checks if the given coordinate is inside the waveguide or
   * not. The naming convention of physical and mathematical system find
   * application. In this version, the waveguide has been transformed and the
   * check for a tubal waveguide for example only checks if the radius of a
   * given vector is below the average of input and output radius.
   * \param position This value gives us the location to check for.
   */
  bool math_coordinate_in_waveguide(Position position) const;

  /**
   * This function checks if the given coordinate is inside the waveguide or
   * not. The naming convention of physical and mathematical system find
   * application. In this version, the waveguide is bent. If we are using a
   * space transformation \f$f\f$ then this function is equal to
   * math_coordinate_in_waveguide(f(x,y,z)). \param position This value gives
   * us the location to check for.
   */
  bool phys_coordinate_in_waveguide(Position position) const;

  /**
   * This function takes a triangulation object and prepares it for the further
   * computations. It is intended to encapsulate all related work and is
   * explicitely not const. \param in_tria The triangulation that is supposed to
   * be prepared. All further information is derived from the parameter file and
   * not given by parameters.
   */
  void prepare_triangulation(dealii::Triangulation<3> *in_tria);

  /**
   * @brief Determines which of the six coordinate directions a vector mainly points in.
   * Applied to the vector from a cell center to a face center this gives the boundary id of the face.
   *
   * @param in_dir The vector to classify.
   * @return unsigned int 0 = -x, 1 = +x, 2 = -y, 3 = +y, 4 = -z, 5 = +z, according to the component of largest absolute value and its sign.
   */
  unsigned int getDominantComponentAndDirection(Position in_dir) const;

  /**
   * @brief Sets the manifold id of the whole mesh to 0 and gives every boundary face of the passed triangulation
   * the boundary id (0 to 5) of the direction its outward normal points in.
   */
  void set_boundary_ids(dealii::Triangulation<3> &) const;

  /** @var SquareMeshGenerator::endc
   * @brief End iterator belonging to cell. Not used either.
   */
  /** @brief Cell iterator. Not used: the member functions work with local iterators of the same name. */
  dealii::Triangulation<3>::active_cell_iterator cell, endc;

  /**
   * @brief Refines the passed triangulation anisotropically until check_and_mark_one_cell_for_refinement() no longer
   * requests refinement for the last cell visited or the mesh has 100000 active cells.
   */
  void refine_triangulation_iteratively(dealii::Triangulation<3, 3> *);

  /**
   * @brief Compares the edges of the passed cell with the maximal cell size allowed at the cell center
   * (hmax_for_cell_center()) and sets a refinement flag that cuts the cell in the direction of a too long edge.
   *
   * @return true At least one edge of the cell is too long. The cell has been flagged.
   * @return false No refinement required for this cell.
   */
  bool check_and_mark_one_cell_for_refinement(dealii::Triangulation<3>::active_cell_iterator);
};
