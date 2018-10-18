/*
 * FaceSurfaceComparatorZ.cpp
 *
 *  Created on: Oct 18, 2018
 *      Author: kraft
 */

#include <deal.II/base/exceptions.h>
#include <cmath>
#include "./FaceSurfaceComparatorZ.h"

template <template <int, int> class MeshType, int dim, int spacedim>
FaceSurfaceComparatorZ<MeshType, dim, spacedim>::FaceSurfaceComparatorZ(
    double in_z, double in_tol) {
  z = in_z;
  tolerance = in_tol;
}

template <template <int, int> class MeshType, int dim, int spacedim>
FaceSurfaceComparatorZ<MeshType, dim, spacedim>::~FaceSurfaceComparatorZ() {

}

template <template <int, int> class MeshType, int dim, int spacedim>
bool FaceSurfaceComparatorZ<MeshType, dim, spacedim>::check_face(
    const typename MeshType<dim, spacedim>::face_iterator face_iterator) {
  Assert(spacedim >= 3, dealii::StandardExceptions::ExcMessage(
                            "Incompatible FaceComparator. Spacedim < 3"));
  return (std::abs(face_iterator.center()[2] - z) <= tolerance);
}
