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
 * displacement (dm/dz, per micrometer). The nodes do not have to be equidistant. Between two neighboring nodes the
 * displacement is the cubic polynomial with the values and slopes of the two nodes (displacement(), slope()), so
 * the shape is continuously differentiable. PredefinedShapeTransformation evaluates these functions. The object is
 * the member Parameters::sd.
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
   * The z values are shifted such that the first node is at z = 0. The nodes are sorted by z (in some lines of the
   * file a node is out of order) and nodes at the same position as their predecessor are dropped; Sectors is the
   * number of remaining nodes.
   */
  void SetByString(std::string);

  /**
   * @brief Describes a straight waveguide.
   * Adds two nodes at the lower and upper end of the global domain in the z-direction with displacement 0. It does
   * not set Sectors and v.
   */
  void SetStraight();

  /**
   * @brief Displacement of the waveguide axis at a position between the nodes.
   *
   * @param z Position along the waveguide, measured like the node positions z (0 at the first node).
   * @return The value of the piecewise cubic interpolation; the displacement of the first or last node outside of
   * the range of the nodes. 0 if there are no nodes.
   */
  double displacement(double z) const;

  /**
   * @brief Derivative of displacement() with respect to z.
   *
   * @param z Position along the waveguide, measured like the node positions z (0 at the first node).
   * @return The derivative of the piecewise cubic interpolation, the slope v of a node at that node, and 0 before
   * the first and behind the last node.
   */
  double slope(double z) const;

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

 private:
  /**
   * @brief Finds the interval of nodes that contains a position.
   *
   * @param z Position along the waveguide inside of the range of the nodes.
   * @param t Receives the local coordinate in the interval, between 0 and 1.
   * @return Index i of the interval between the nodes i and i + 1.
   */
  unsigned int interval(double z, double & t) const;
};
