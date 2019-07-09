/*
 * FaceSurfaceComparatorZ.h
 *
 *  Created on: Oct 18, 2018
 *      Author: kraft
 */

#ifndef CODE_HSIE_FACESURFACECOMPARATORZ_H_
#define CODE_HSIE_FACESURFACECOMPARATORZ_H_

#include <deal.II/distributed/tria.h>
#include "./FaceSurfaceComparator.h"

class FaceSurfaceComparatorZ : public FaceSurfaceComparator {
private:
    double z;
    double tolerance;

public:
    FaceSurfaceComparatorZ(double in_z = 0, double in_tol = 0.0001);

    virtual ~FaceSurfaceComparatorZ();

    virtual bool check_face(
            const dealii::parallel::distributed::Triangulation<3, 3>::face_iterator);
};

#endif /* CODE_HSIE_FACESURFACECOMPARATORZ_H_ */
