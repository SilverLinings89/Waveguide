#ifndef HSIEPRECONDITIONERBASE_CPP
#define HSIEPRECONDITIONERBASE_CPP
#include "HSIE_Preconditioner_Base.h"
#include <complex.h>

/*
 * HSIE_Preconditioner_Base.cpp
 *
 *  Created on: Jul 11, 2018
 *      Author: kraft
 */

/**
 * Implement:
 * - a(u,v)
 * - g(j,k)
 * - A_C,T
 * - (28)
 *
 */

const unsigned int S1IntegrationPoints = 100;
const std::complex<double> k0(5, -1);

dealii::Point<3, std::complex<double>> base_fun(int i, int j,
                                                dealii::Point<2, double> x) {
  dealii::Point<3, std::complex<double>> ret =
      new dealii::Point<3, std::complex<double>>();
  ret[0] = new std::complex<double>(0, 0);
  ret[1] = new std::complex<double>(0, 0);
  ret[2] = new std::complex<double>(0, 0);
  return ret;
}

std::complex<double>* a(int in_i, int in_j) {
  dealii::Point<3, std::complex<double>> ret =
      new dealii::Point<3, std::complex<double>>();
  for (unsigned int i = 0; i < 3; i++) {
    ret[i] = new std::complex<double>(0, 0);
  }
  for (unsigned int i = 0; i < S1IntegrationPoints; i++) {
    double x = std::sin(2 * 3.14159265359 * ((double)i) /
                        ((double)S1IntegrationPoints));
    double y = std::cos(2 * 3.14159265359 * ((double)i) /
                        ((double)S1IntegrationPoints));
    dealii::Point<2, double> p1(x, y);
    dealii::Point<2, double> p2(x, -y);
    ret[0] += base_fun(in_i, 0, p1) * base_fun(in_j, 0, p2);
  }
  ret /= (double)S1IntegrationPoints;
  ret *= k0 * std::complex<double>(0, 1) / 3.14159265359;
  return ret;
}

#endif
