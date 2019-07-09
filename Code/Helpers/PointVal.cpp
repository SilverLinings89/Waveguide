/*
 * PointVal.cpp
 *
 *  Created on: Dec 12, 2017
 *      Author: pascal
 */

#ifndef PointValSourceFlag
#define PointValSourceFlag

#include "PointVal.h"

PointVal::PointVal() : Ex(0.0, 0.0), Ey(0.0, 0.0), Ez(0.0, 0.0) {}

PointVal::PointVal(double rx, double ry, double rz, double ix, double iy,
                   double iz)
        : Ex(rx, ix), Ey(ry, iy), Ez(rz, iz) {}

void PointVal::set(double rx, double ry, double rz, double ix, double iy,
                   double iz) {
    Ex.real(rx);
    Ex.imag(ix);
    Ey.real(ry);
    Ey.imag(iy);
    Ez.real(rz);
    Ez.imag(iz);
}

void PointVal::rescale(double inp) {
    Ex = Ex * inp;
    Ey = Ey * inp;
    Ez = Ez * inp;
}

#endif
