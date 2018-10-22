/*
 * FaceSurfaceComparatorZ.cpp
 *
 *  Created on: Oct 18, 2018
 *      Author: kraft
 */

#include "./FaceSurfaceComparatorZ.h"
#include <deal.II/base/exceptions.h>
#include <deal.II/distributed/tria.h>
#include <cmath>

FaceSurfaceComparatorZ::FaceSurfaceComparatorZ(double in_z, double in_tol) {
  z = in_z;
  tolerance = in_tol;
}

FaceSurfaceComparatorZ::~FaceSurfaceComparatorZ() {}
bool FaceSurfaceComparatorZ::check_face(
    const dealii::parallel::distributed::Triangulation<3, 3>::face_iterator
        face_iterator) {
  Assert(spacedim >= 3, dealii::StandardExceptions::ExcMessage(
                            "Incompatible FaceComparator. Spacedim < 3"));
  return (std::abs(face_iterator->center()[2] - z) <= tolerance);
}
