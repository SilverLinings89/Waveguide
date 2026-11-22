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
 * @brief The waveguide mode for the AngleWaveguideTransformation (the PML test with a tilted waveguide).
 *
 * The mode of the straight waveguide (ExactSolution) is evaluated at the transformed position
 * (x, y, z - a y) with a = GlobalParams.PML_Angle_Test and the field is multiplied with the inverse transpose of the
 * Jacobian of the global space transformation. Parameters::complete_data() selects this field for source type 3
 * together with the angle transformation.
 */
class AngledExactSolution: public dealii::Function<3, ComplexNumber> {
 private:
  /** @brief The mode of the straight waveguide (an ExactSolution created in the constructor, never deleted). */
  dealii::Function<3, ComplexNumber> * base_solution;
  /** @brief Not used. */
  BoundaryId main_boundary;
  /** @brief Not used. */
  double additional_coordinate;
  /** @brief Not used. */
  std::vector<double> u1, u2, u3;

 public:
  /** @brief Creates the base solution. ExactSolution::load_data() has to be called before. */
  AngledExactSolution();
  
  /** @brief Legacy declaration without definition. Not used. */
  std::vector<std::string> split(std::string) const;

  /**
   * @brief One component of the field.
   * @param p position in the coordinates of the mesh, in micrometers.
   * @param component 0, 1 or 2 for the x-, y- or z-component.
   * @return ComplexNumber component of J^-T E_mode(p') with p' = transform_position(p) and the Jacobian J at p'.
   */
  ComplexNumber value(const Position &p, const unsigned int component) const;

  /**
   * @brief All three components of the field, see value().
   * @param p position in the coordinates of the mesh, in micrometers.
   * @param value vector of size 3 that receives the field.
   */
  void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const;

  /**
   * @brief Curl of the field, approximated with forward finite differences of vector_value() with step 1e-4.
   * The differences are evaluated around transform_position(in_p) and vector_value() applies the transformation
   * again.
   * @param in_p position in micrometers.
   * @return dealii::Tensor<1, 3, ComplexNumber> the approximated curl.
   */
  dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;

  /**
   * @brief The mode of the straight waveguide at the transformed position, with the y-component multiplied by
   * sqrt(2). In contrast to vector_value() the Jacobian is not applied.
   * @param in_p position in micrometers.
   * @return dealii::Tensor<1, 3, ComplexNumber> the field.
   */
  dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;

  /**
   * @brief The coordinate change of the tilted waveguide.
   * @param in_p position (x, y, z).
   * @return Position (x, y, z - a y) with a = GlobalParams.PML_Angle_Test.
   */
  Position transform_position(const Position &in_p) const;
};
