#pragma once

/**
 * @file ShapeDescription.h
 * @author Pascal Kraft
 * @brief An object used to wrap the description of the prescribed waveguide shapes.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <string>
#include <vector>

/**
 * @brief Describes the shape of a waveguide by values at nodes along the z-axis.
 *
 * Node i has the position z[i], the transverse displacement m[i] of the waveguide axis and the slope v[i] of the
 * displacement (dm/dz). PredefinedShapeTransformation builds one Sector between two neighboring nodes and
 * interpolates m with cubic polynomials. The object is the member Parameters::sd.
 */
class ShapeDescription {
 public:
  /** @brief Creates an empty description. */
  ShapeDescription();

  /** @brief Nothing to release. */
  ~ShapeDescription();

  /**
   * @brief Reads the description from one line of Modes/test.csv.
   * The line is a comma separated list: the number of nodes n, then n values of z, n values of m and n values of v.
   * The z values are shifted such that the first node is at z = 0.
   */
  void SetByString(std::string);

  /**
   * @brief Describes a straight waveguide.
   * Adds two nodes at the lower and upper end of the global domain in the z-direction with displacement 0. It does
   * not set Sectors and v.
   */
  void SetStraight();

  /** @brief Number of nodes of the description, as read by SetByString(). */
  int Sectors;
  
  /**
   * @var m
   * @brief Transverse displacement of the waveguide axis at every node in micrometers.
   */
  /**
   * @var v
   * @brief Derivative of the displacement m with respect to z at every node.
   */
  /**
   * @var z
   * @brief Position of every node along the waveguide in micrometers.
   */
  std::vector<double> m, v, z;
};
