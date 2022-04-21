#pragma once 
/**
 * @file WaveguideTransformation.h
 * @author Pascal Kraft
 * @brief Contains the implementation of the Waveguide Transformation.
 * @version 0.1
 * @date 2022-04-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <math.h>
#include <vector>

#include "../Core/InnerDomain.h"
#include "../Optimization/ShapeFunction.h"
#include "../Core/Sector.h"
#include "SpaceTransformation.h"

enum ResponsibleComponent {
  VerticalDisplacementComponent, WaveguideHeightComponent, WaveguideWidthComponent
};

/**
 * \class WaveguideTransformation
 * \brief In this case we regard a rectangular waveguide and the effects on the
 * material tensor by the space transformation and the boundary condition PML
 * may overlap.
 * 
 * The waveguide transformation is a variable y-shift of the coordinate system and uses a shape-function to describe the shape.
 * 
 * For the non-documented members see the documentation in the base class SpaceTransformation.
 */

class WaveguideTransformation : public SpaceTransformation {
  dealii::Tensor<2,3,double> I;
  ShapeFunction waveguide_width, waveguide_height, vertical_shift;

 public:
  WaveguideTransformation();

  virtual ~WaveguideTransformation();

  Position math_to_phys(Position coord) const override;

  Position phys_to_math(Position coord) const override;

  dealii::Tensor<2, 3, ComplexNumber> get_Tensor(Position &coordinate) override;

  dealii::Tensor<2, 3, double> get_Space_Transformation_Tensor(Position &coordinate) override;

  Tensor<2,3,double> get_J(Position &) override;

  Tensor<2,3,double> get_J_inverse(Position &) override;

  /**
   * At the beginning (before the first solution of a system) only the boundary
   * conditions for the shape of the waveguide are known. Therefore the values
   * for the degrees of freedom need to be estimated. This function sets all
   * variables to appropiate values and estimates an appropriate shape based on
   * averages and a polynomial interpolation of the boundary conditions on the
   * shape.
   */
  void estimate_and_initialize() override;

  /**
   * This is a getter for the values of degrees of freedom. A getter-setter
   * interface was introduced since the values are estimated automatically
   * during the optimization and non-physical systems should be excluded from
   * the domain of possible cases. \param dof The index of the degree of freedom
   * to be retrieved from the structure of the modelled waveguide. \return This
   * function returns the value of the requested degree of freedom. Should this
   * dof not exist, 0 will be returnd.
   */
  double get_dof(int dof) const override;

  /**
   * This is a getter for the values of degrees of freedom. A getter-setter
   * interface was introduced since the values are estimated automatically
   * during the optimization and non-physical systems should be excluded from
   * the domain of possible cases. \param dof The index of the degree of freedom
   * to be retrieved from the structure of the modelled waveguide. \return This
   * function returns the value of the requested degree of freedom. Should this
   * dof not exist, 0 will be returnd.
   */
  double get_free_dof(int dof) const override;

  /**
   * This function sets the value of the dof provided to the given value. It is
   * important to consider, that some dofs are non-writable (i.e. the values of
   * the degrees of freedom on the boundary, like the radius of the
   * input-connector cannot be changed). \param dof The index of the parameter
   * to be changed. \param value The value, the dof should be set to.
   */
  void set_free_dof(int dof, double value) override;

  /**
   * Other objects can use this function to retrieve an array of the current
   * values of the degrees of freedom of the functional we are optimizing. This
   * also includes restrained degrees of freedom and other functions can be used
   * to determine this property. This has to be done because in different cases
   * the number of restrained degrees of freedom can vary and we want no logic
   * about this in other functions.
   */
  Vector<double> get_dof_values() const override;

  /**
   * This function returns the number of unrestrained degrees of freedom of the
   * current optimization run.
   */
  unsigned int n_free_dofs() const override;

  /**
   * This function returns the total number of DOFs including restrained ones.
   * This is the lenght of the array returned by Dofs().
   */
  unsigned int n_dofs() const override;

  /**
   * Console output of the current Waveguide Structure.
   */
  void Print() const override;

  std::pair<ResponsibleComponent, unsigned int> map_free_dof_index(unsigned int) const;

  std::pair<ResponsibleComponent, unsigned int> map_dof_index(unsigned int) const;
};
