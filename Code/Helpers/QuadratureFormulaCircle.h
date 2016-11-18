#ifndef QuadratureFlag
#define QuadratureFlag

extern "C" {
	std::complex<double> gauss_product_2D_sphere(double z, int n, std::complex<double> (*f)(double,double,double), double R, double Xc, double Yc);
}

#endif
