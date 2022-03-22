#pragma once

/**
 * @file PointVal.h
 * @author your name (you@domain.com)
 * @brief Not currently used.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../Core/Types.h"

/**
 * @brief Old class that was used for the interpolation of input signals.
 * 
 */
class PointVal {
 public:
   ComplexNumber Ex;
   ComplexNumber Ey;
   ComplexNumber Ez;

  PointVal();

  PointVal(double, double, double, double, double, double);

  void set(double, double, double, double, double, double);

  void rescale(double);
};
