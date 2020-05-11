#ifndef PointValHeaderFlag
#define PointValHeaderFlag

#include <complex>

class PointVal {
 public:
  std::complex<double> Ex;
  std::complex<double> Ey;
  std::complex<double> Ez;

  PointVal();

  PointVal(double, double, double, double, double, double);

  void set(double, double, double, double, double, double);

  void rescale(double);
};

#endif
