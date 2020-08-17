#pragma once

#include "../Core/Types.h"
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
