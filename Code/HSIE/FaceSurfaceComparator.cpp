/*
 * FaceSurfaceComparator.cpp
 *
 *  Created on: Oct 18, 2018
 *      Author: kraft
 */

#include "./FaceSurfaceComparator.h"

template <template <int, int> class MeshType, int dim, int spacedim>
FaceSurfaceComparator<MeshType, dim, spacedim>::FaceSurfaceComparator() {
  // TODO Auto-generated constructor stub
}

template <template <int, int> class MeshType, int dim, int spacedim>
FaceSurfaceComparator<MeshType, dim, spacedim>::~FaceSurfaceComparator() {
  // TODO Auto-generated destructor stub
}

template <template <int, int> class MeshType, int dim, int spacedim>
bool FaceSurfaceComparator<MeshType, dim, spacedim>::check_face(
    const typename MeshType<dim, spacedim>::face_iterator face_iterator) {
  return false;
}
