#pragma once

/**
 * @file PointVal.h
 * @author Pascal Kraft
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
   /** @brief x-component of the electric field at the point. */
   ComplexNumber Ex;
   /** @brief y-component of the electric field at the point. */
   ComplexNumber Ey;
   /** @brief z-component of the electric field at the point. */
   ComplexNumber Ez;

  /** @brief Sets all three components to zero. */
  PointVal();

  /**
   * @brief Sets the components from real and imaginary parts.
   * The arguments are Re(Ex), Re(Ey), Re(Ez), Im(Ex), Im(Ey), Im(Ez) in this order.
   */
  PointVal(double, double, double, double, double, double);

  /**
   * @brief Overwrites the components.
   * The arguments are Re(Ex), Re(Ey), Re(Ez), Im(Ex), Im(Ey), Im(Ez) in this order.
   */
  void set(double, double, double, double, double, double);

  /** @brief Multiplies all three components with the given real factor. */
  void rescale(double);
};
