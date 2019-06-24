/*
 * GemoetryManager.h
 * An object of this type handles all the geometric information of the run.
 * It can be used by the Mesh-Generator to retrieve data, or by the Simulation
 * to retrieve neighboring-process information as well as surface-types
 * (boundary condition, dirichlet surface etc.)
 * An object of this type is available as a static variable.
 * Created on: Jun 19, 2019
 * Author: Pascal Kraft
 */

#ifndef CODE_HELPERS_GEOMETRY_H_
#define CODE_HELPERS_GEOMETRY_H_
#include "Parameters.h"

enum Direction {
  MinusX = 0,
  PlusX = 1,
  MinusY = 2,
  PlusY = 3,
  MinusZ = 4,
  PlusZ = 5
};

class Geometry {
 public:
  Geometry();
  virtual ~Geometry();

  void initialize(Parameters parameters);
};

#endif /* CODE_HELPERS_GEOMETRY_H_ */
