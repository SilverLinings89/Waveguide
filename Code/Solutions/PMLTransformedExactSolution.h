#pragma once

#include <array>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>
#include "../Helpers/PointVal.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "./ExactSolution.h"

/**
 * @brief The source field multiplied with a real damping factor inside a PML.
 *
 * DirichletSurface uses it to compute the boundary values on the part of a Dirichlet boundary that belongs to the
 * PML domain of an adjacent absorbing surface, where the exact solution has to decay like the PML damps it. The
 * factor is only an approximation (see compute_scaling_factor()).
 */
class PMLTransformedExactSolution: public dealii::Function<3, ComplexNumber> {
 private:
  /** @brief The undamped field, GlobalParams.source_field. Not owned. */
  dealii::Function<3, ComplexNumber> * base_solution;
  /** @brief Boundary id of the Dirichlet surface the values are computed for. main_boundary / 2 is the coordinate direction used for the damping. */
  BoundaryId main_boundary;
  /** @brief Coordinate of the Dirichlet surface in its normal direction. Stored but not used. */
  double additional_coordinate;
  /** @brief Thickness of one cell layer of the PML, PML_thickness / PML_N_Layers, in micrometers. */
  double non_pml_layer_thickness;

 public:
  /**
   * @brief Creates the field for one Dirichlet surface.
   * @param in_main_id boundary id of the Dirichlet surface.
   * @param in_additional_coordinate coordinate of that surface in its normal direction.
   */
  PMLTransformedExactSolution( BoundaryId in_main_id, double in_additional_coordinate);
  
  /** @brief Legacy declaration without definition. Not used. */
  std::vector<std::string> split(std::string) const;

  /**
   * @brief One component of the damped field.
   * Only the component with index main_boundary / 2 is multiplied with the damping factor; the other components are
   * returned undamped. vector_value() damps all components.
   * @param p position in micrometers.
   * @param component 0, 1 or 2 for the x-, y- or z-component.
   * @return ComplexNumber the value of the component.
   */
  ComplexNumber value(const Position &p, const unsigned int component) const;

  /**
   * @brief All three components of the source field, multiplied with compute_scaling_factor().
   * @param p position in micrometers.
   * @param value vector of size 3 that receives the field.
   */
  void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const;

  /**
   * @brief Not implemented.
   * @return dealii::Tensor<1, 3, ComplexNumber> always zero.
   */
  dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;

  /**
   * @brief The damped field as a tensor; the same values as vector_value().
   * @param in_p position in micrometers.
   * @return dealii::Tensor<1, 3, ComplexNumber> the field.
   */
  dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;
  
  /**
   * @brief Relative depth of a position in the PML for the three coordinate directions.
   * For every direction the value is 0 if the coordinate is inside the range of the local (subdomain) geometry.
   * Outside it is (distance to the range -/+ one layer thickness) / (PML_thickness - one layer thickness); the layer
   * thickness is added below the range and subtracted above it. Below the z-range the x-coordinate of the position
   * is used.
   * @param in_p position in micrometers.
   * @return std::array<double, 3> the values for x, y and z.
   */
  std::array<double, 3> fraction_of_pml_direction(const Position & in_p) const;

  /**
   * @brief The damping factor exp(-sigma * distance).
   * With d = fraction_of_pml_direction(in_p)[main_boundary / 2] the distance is d * (PML_thickness - one layer
   * thickness). sigma is PML_Sigma_Max if PML_skaling_order is 0 and d^(order + 1) / order otherwise. This is an
   * approximation of the damping of the PML and not the exact integral of its profile.
   * @param in_p position in micrometers.
   * @return double the factor.
   */
  double compute_scaling_factor(const Position & in_p) const;
};
