#pragma once 

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <math.h>
#include <vector>

#include "../Core/InnerDomain.h"
#include "../Core/Sector.h"
#include "SpaceTransformation.h"

/**
 * \class AngleWaveguideTransformation
 * \brief 
 * 
 * \author Pascal Kraft \date 28.11.2016
 */

class AngleWaveguideTransformation : public SpaceTransformation {
  bool is_constant = true;
  dealii::Tensor<2, 3, double> J_perm;
  dealii::Tensor<2, 3, double> J_inv_perm;
  double det;
  bool is_J_prepared = false;
  bool is_J_inv_prepared = false;
  bool is_det_prepared = false;

 public:
  AngleWaveguideTransformation();

  virtual ~AngleWaveguideTransformation();

  Position math_to_phys(Position coord) const;

  Position phys_to_math(Position coord) const;

  dealii::Tensor<2, 3, double> get_J(Position &coordinate) override;

  dealii::Tensor<2, 3, double> get_J_inverse(Position &coordinate) override;

  double get_det(Position coord) override;

  dealii::Tensor<2, 3, ComplexNumber> get_Tensor(Position &coordinate) override;

  dealii::Tensor<2, 3, double> get_Space_Transformation_Tensor(Position &coordinate) override;

  /**
   * At the beginning (before the first solution of a system) only the boundary
   * conditions for the shape of the waveguide are known. Therefore the values
   * for the degrees of freedom need to be estimated. This function sets all
   * variables to appropiate values and estimates an appropriate shape based on
   * averages and a polynomial interpolation of the boundary conditions on the
   * shape.
   */
  void estimate_and_initialize();

  
  /**
   * Other objects can use this function to retrieve an array of the current
   * values of the degrees of freedom of the functional we are optimizing. This
   * also includes restrained degrees of freedom and other functions can be used
   * to determine this property. This has to be done because in different cases
   * the number of restrained degrees of freedom can vary and we want no logic
   * about this in other functions.
   */
  Vector<double> get_dof_values() const;

  /**
   * This function returns the number of unrestrained degrees of freedom of the
   * current optimization run.
   */
  unsigned int n_free_dofs() const;

  /**
   * This function returns the total number of DOFs including restrained ones.
   * This is the lenght of the array returned by Dofs().
   */
  unsigned int n_dofs() const;

   /**
   * Console output of the current Waveguide Structure.
   */
  void Print() const;

};
