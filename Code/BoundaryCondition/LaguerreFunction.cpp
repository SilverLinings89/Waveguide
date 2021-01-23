//
// Created by pascal on 11.09.19.
//

#include "LaguerreFunction.h"
#include <cmath>

double LaguerreFunction::evaluate(unsigned int n, unsigned int m, double x) {
  double ret = 0;
  for (unsigned int k = 0; k <= n; k++) {
    ret += (double)binomial_coefficient(n + m, n - k) *
           (std::pow(-x, k) / (double)factorial(k));
  }
  return ret;
}

double LaguerreFunction::factorial(unsigned int in_n) {
  double n = in_n;
  if (n == 0) {
    return 1;
  } else {
    return n * factorial_internal(n - 1);
  }
}

double LaguerreFunction::factorial_internal(double n) {
  if (n == 0) {
    return 1;
  } else {
    return n * factorial(n - 1);
  }
}

unsigned int LaguerreFunction::binomial_coefficient(unsigned int n,
                                                    unsigned int k) {
  if (k > n) return 0;
  return factorial(n) / (factorial(k) * factorial(n - k));
}
