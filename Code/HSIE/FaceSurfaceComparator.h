/*
 * FaceSurfaceComparator.h
 *
 *  Created on: Oct 18, 2018
 *      Author: kraft
 */

#ifndef CODE_HSIE_FACESURFACECOMPARATOR_H_
#define CODE_HSIE_FACESURFACECOMPARATOR_H_

template <template <int, int> class MeshType, int dim, int spacedim>
class FaceSurfaceComparator {
 public:
  FaceSurfaceComparator();
  virtual ~FaceSurfaceComparator();
  virtual bool check_face(
      const typename MeshType<dim, spacedim>::face_iterator);
};

#endif /* CODE_HSIE_FACESURFACECOMPARATOR_H_ */
