#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>
#include "../Helpers/PointVal.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "./ExactSolution.h"

/**
 * @brief The waveguide mode ExactSolution multiplied with a ramp in the z-direction.
 * The ramp is 1 for z <= tapering_min_z and 0 for z >= tapering_max_z (members of Parameters). In between it is
 * linear (SignalTaperingType::C0) or the cubic polynomial 2 t^3 - 3 t^2 + 1 with t in [0,1] (C1). The class is
 * currently not instantiated; the tapered right hand side is assembled in InnerDomain.
 */
class ExactSolutionRamped: public dealii::Function<3, ComplexNumber> {
  private:
    /** @brief The mode that is ramped. */
    ExactSolution inner_field;
    /** @brief z-coordinate at which the ramp starts to decrease from 1 (GlobalParams.tapering_min_z). */
    const double min_z;
    /** @brief z-coordinate at which the ramp reaches 0 (GlobalParams.tapering_max_z). */
    const double max_z;
    /** @brief True for the linear (C0) ramp and false for the cubic (C1) ramp. */
    const bool do_c0;

  public:
    /** @brief Reads the taper settings from GlobalParams. ExactSolution::load_data() has to be called before. */
    ExactSolutionRamped();
    /**
     * @brief The value of the ramp at a position, according to the tapering type.
     * @return double the factor in [0,1]; only the z-coordinate of the argument is used.
     */
    double get_ramping_factor_for_position(const Position &) const;
    /**
     * @brief One component of the ramped mode.
     * @param p position in micrometers.
     * @param component 0, 1 or 2 for the x-, y- or z-component.
     * @return ComplexNumber ramp(z) times ExactSolution::value().
     */
    ComplexNumber value(const Position &p, const unsigned int component) const;
    /**
     * @brief All three components of the ramped mode.
     * @param p position in micrometers.
     * @param value vector of size 3 that receives the field.
     */
    void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const;
    /**
     * @brief Curl of the ramped mode, approximated with forward finite differences of vector_value() with step 1e-4.
     * @param in_p position in micrometers.
     * @return dealii::Tensor<1, 3, ComplexNumber> the approximated curl.
     */
    dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;
    /**
     * @brief The ramped mode as a tensor.
     * @param in_p position in micrometers.
     * @return dealii::Tensor<1, 3, ComplexNumber> ramp(z) times ExactSolution::val().
     */
    dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;
    /**
     * @brief The linear ramp.
     * @param in_p position; only z is used.
     * @return double 1 - t with t = ramping_delta(in_p).
     */
    double compute_ramp_for_c0(const Position &in_p) const;
    /**
     * @brief The cubic ramp with vanishing derivative at both ends.
     * @param in_p position; only z is used.
     * @return double 2 t^3 - 3 t^2 + 1 with t = ramping_delta(in_p).
     */
    double compute_ramp_for_c1(const Position &in_p) const;
    /**
     * @brief The relative position in the ramp.
     * @param in_p position; only z is used.
     * @return double (z - min_z) / (max_z - min_z), limited to [0,1].
     */
    double ramping_delta(const Position &in_p) const;
    /**
     * @brief Derivative of the ramp with respect to z.
     * For the linear ramp the constant -1 / |max_z - min_z| is returned for every z, also outside of the ramp.
     * @param in_p position; only z is used.
     * @return double the derivative.
     */
    double get_ramping_factor_derivative_for_position(const Position &in_p) const;
};
