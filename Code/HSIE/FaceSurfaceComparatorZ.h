/*
 * FaceSurfaceComparatorZ.h
 *
 *  Created on: Oct 18, 2018
 *      Author: kraft
 */

#ifndef CODE_HSIE_FACESURFACECOMPARATORZ_H_
#define CODE_HSIE_FACESURFACECOMPARATORZ_H_

#include "./FaceSurfaceComparator.h"

template <template <int, int> class MeshType, int dim, int spacedim>
class FaceSurfaceComparatorZ
    : public FaceSurfaceComparator<MeshType, dim, spacedim> {
 private:
  double z;
  double tolerance;

 public:
  FaceSurfaceComparatorZ(double in_z = 0, double in_tol = 0.0001);
  virtual ~FaceSurfaceComparatorZ();
  virtual bool check_face(
      const typename MeshType<dim, spacedim>::face_iterator);
};

#endif /* CODE_HSIE_FACESURFACECOMPARATORZ_H_ */
