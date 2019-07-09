/*
 * FaceSurfaceComparator.h
 *
 *  Created on: Oct 18, 2018
 *      Author: kraft
 */

#ifndef CODE_HSIE_FACESURFACECOMPARATOR_H_
#define CODE_HSIE_FACESURFACECOMPARATOR_H_

#include <deal.II/distributed/tria.h>

class FaceSurfaceComparator {
public:
    FaceSurfaceComparator();

    virtual ~FaceSurfaceComparator();

    virtual bool check_face(
            const dealii::parallel::distributed::Triangulation<3, 3>::face_iterator);
};

#endif /* CODE_HSIE_FACESURFACECOMPARATOR_H_ */
