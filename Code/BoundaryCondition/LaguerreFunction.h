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
 * \class LaguerreFunction
 * 
 * These is not currently being used. It will be used in a complex scaled infinite element once that is implemented.
 * All members are static, the class only serves as a namespace.
 */
class LaguerreFunction {
 private:
  /**
   * @brief Second half of the recursion of factorial(). The two functions call each other.
   *
   * @param n A non-negative integer stored as a double.
   * @return n!
   */
  static double factorial_internal(double n);

 public:
  /**
   * @brief Evaluates the generalized Laguerre polynomial \f$L_n^{(m)}(x) = \sum_{k=0}^{n} \binom{n+m}{n-k} \frac{(-x)^k}{k!}\f$.
   *
   * @param n Degree of the polynomial.
   * @param m Parameter of the generalized Laguerre polynomial. m = 0 gives the ordinary Laguerre polynomial.
   * @param x Point of evaluation.
   * @return Value of the polynomial at x.
   */
  static double evaluate(unsigned int n, unsigned int m, double x);
  /**
   * @brief Computes the factorial recursively.
   *
   * @param n The argument.
   * @return n! as a double.
   */
  static double factorial(unsigned int n);
  /**
   * @brief Computes the binomial coefficient n over k from factorials in floating point arithmetic.
   *
   * @param n Upper argument.
   * @param k Lower argument.
   * @return n! / (k! (n-k)!), or 0 if k > n.
   */
  static unsigned int binomial_coefficient(unsigned int n, unsigned int k);
};
