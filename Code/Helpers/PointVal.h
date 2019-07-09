#ifndef PointValHeaderFlag
#define PointValHeaderFlag

#include <complex>

class PointVal {
public:
    std::complex<double> Ex, Ey, Ez;

    PointVal();

    PointVal(double, double, double, double, double, double);

    void set(double, double, double, double, double, double);

    void rescale(double);
};

#endif
