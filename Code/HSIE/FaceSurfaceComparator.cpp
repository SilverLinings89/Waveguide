/*
 * FaceSurfaceComparator.cpp
 *
 *  Created on: Oct 18, 2018
 *      Author: kraft
 */

#include "./FaceSurfaceComparator.h"

FaceSurfaceComparator::FaceSurfaceComparator() {
    // TODO Auto-generated constructor stub
}

FaceSurfaceComparator::~FaceSurfaceComparator() {
    // TODO Auto-generated destructor stub
}

bool FaceSurfaceComparator::check_face(
        const dealii::parallel::distributed::Triangulation<3, 3>::face_iterator
        ) {
    return false;
}
