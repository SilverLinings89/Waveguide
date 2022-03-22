#pragma once

/**
 * @file LaguerreFunction.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief An implementation of Laguerre functions which is not currently being used.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

/**
 * \class LaguerreFunctions
 * 
 * These is not currently being used. It will be used in a complex scaled infinite element once that is implemented.
 * Since it is not currently used, this is not documented.
 */
class LaguerreFunction {
 private:
  static double factorial_internal(double n);

 public:
  static double evaluate(unsigned int n, unsigned int m, double x);
  static double factorial(unsigned int n);
  static unsigned int binomial_coefficient(unsigned int n, unsigned int k);
};
