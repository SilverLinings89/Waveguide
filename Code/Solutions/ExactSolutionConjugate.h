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
 * @brief The complex conjugate of the waveguide mode ExactSolution.
 * Because the mode is exp(i beta z) times a profile, the conjugate is the mode that travels in the -z direction.
 * The class is currently not instantiated.
 */
class ExactSolutionConjugate: public dealii::Function<3, ComplexNumber> {
  private:
    /** @brief The mode that is conjugated. */
    ExactSolution inner_field;

  public:
    /** @brief Creates the inner mode. ExactSolution::load_data() has to be called before. */
    ExactSolutionConjugate();
    /**
     * @brief One component of the conjugated mode.
     * @param p position in micrometers.
     * @param component 0, 1 or 2 for the x-, y- or z-component.
     * @return ComplexNumber the complex conjugate of ExactSolution::value().
     */
    ComplexNumber value(const Position &p, const unsigned int component) const;
    /**
     * @brief All three components of the conjugated mode.
     * @param p position in micrometers.
     * @param value vector of size 3 that receives the field.
     */
    void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const;
    /**
     * @brief Curl of the conjugated mode.
     * @param in_p position in micrometers.
     * @return dealii::Tensor<1, 3, ComplexNumber> the complex conjugate of ExactSolution::curl().
     */
    dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;
    /**
     * @brief The conjugated mode as a tensor.
     * @param in_p position in micrometers.
     * @return dealii::Tensor<1, 3, ComplexNumber> the complex conjugate of ExactSolution::val().
     */
    dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;
};
