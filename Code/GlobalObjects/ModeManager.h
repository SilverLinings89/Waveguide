#pragma once

/**
 * @file ModeManager.h
 * @author Pascal Kraft
 * @brief Not currently in use.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <deal.II/base/point.h>

/**
 * @brief Computes the analytic modes of a circular step-index waveguide (weakly guiding approximation).
 *
 * Legacy code. The global object GlobalModeManager is loaded at startup, but no other part of the code evaluates
 * the modes. A mode is described by the root u of the characteristic equation
 * \f$-u J_0(u) / J_1(u) = w K_0(w) / K_1(w)\f$ with \f$w = \sqrt{v^2 - u^2}\f$, where v is the normalized frequency.
 * The Bessel functions are evaluated by polynomial approximations.
 */
class ModeManager {
 public:
  /** @brief Marks the modes of both connectors as not prepared. */
  ModeManager();

  /** @brief Computes the roots of the characteristic equation for the input connector and stores them. Called on first use by get_input_component(). */
  void prepare_mode_in();

  /** @brief Computes the roots of the characteristic equation for the output connector and stores them. Called on first use by get_output_component(). */
  void prepare_mode_out();

  /** @brief Declared but not implemented. */
  int number_modes_in();

  /** @brief Declared but not implemented. */
  int number_modes_out();

  /**
   * @brief Evaluates one component of a mode profile of the input connector.
   *
   * The arguments are, in this order: the vector component (0, 1 or 2), the position (only the distance from the
   * z-axis is used) and the index of the mode.
   *
   * @return double For component 0 the radial profile of the mode, normalized to 1 at the radius u of the mode. 0 for the other components.
   */
  double get_input_component(int, Position, int);

  /**
   * @brief Same as get_input_component() for the output connector.
   *
   * @return double Value of the requested component of the mode profile.
   */
  double get_output_component(int, Position, int);

  /** @brief Sets both connectors to circular and their normalized frequency to twice the parameter Waveguide_value_V. */
  void load();

 private:
  /** @brief True if the input connector is circular. Always true after load(). */
  bool in_circular;
  /** @brief True if the output connector is circular. Always true after load(). */
  bool out_circular;
  /** @brief Roots u of the characteristic equation for the input connector, one per mode. */
  std::vector<double> u_in;
  /** @brief Normalized frequency of the input connector. */
  double v_in;
  /** @brief Roots u of the characteristic equation for the output connector, one per mode. */
  std::vector<double> u_out;
  /** @brief Normalized frequency of the output connector. */
  double v_out;
  /** @brief True once u_in has been computed. */
  bool in_prepared;
  /** @brief True once u_out has been computed. */
  bool out_prepared;

  /** @brief Finds all roots u in (0.1, v) of the characteristic equation for the normalized frequency v by a scan with step 0.1 followed by ten bisection steps. */
  std::vector<double> get_us(double);

  /** @brief Same search as get_us() but returns only the smallest root, or -1 if there is none. */
  double get_u0(double);

  /** @brief Left hand side of the characteristic equation, \f$-u J_0(u) / J_1(u)\f$. */
  double get_lhs(double);

  /** @brief Right hand side of the characteristic equation for the arguments u and v, \f$w K_0(w) / K_1(w)\f$ with \f$w = \sqrt{v^2 - u^2}\f$. */
  double get_rhs(double, double);

  /** @brief Bessel function of the first kind of order 0. */
  double J_0(double);

  /** @brief Bessel function of the first kind of order 1. */
  double J_1(double);

  /** @brief Bessel function of the first kind. The second argument is the order. */
  double J_n(double, int);

  /** @brief Modified Bessel function of the second kind of order 0. */
  double K_0(double);

  /** @brief Modified Bessel function of the second kind of order 1. */
  double K_1(double);

  /** @brief Modified Bessel function of the second kind. The second argument is the order. */
  double K_n(double, int);

  /** @brief Modified Bessel function of the first kind of order 0. */
  double bessi0(double);

  /** @brief Modified Bessel function of the first kind of order 1. */
  double bessi1(double);
};
