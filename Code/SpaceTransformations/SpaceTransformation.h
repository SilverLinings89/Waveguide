#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <math.h>
#include <vector>
#include "../Core/Enums.h"
#include "../Core/Types.h"

using namespace dealii;


template <unsigned int Dofs_Per_Sector>
class Sector;

/**
 * \class SpaceTransformation
 * \brief The SpaceTransformation class encapsulates the coordinate
 * transformation used in the simulation.
 *
 * Two important decisions have to be made in the computation: Which shape
 * should be used for the waveguide? This can either be rectangular or tubular.
 * Should the coordinate-transformation always be equal to identity in any
 * domain where PML is applied? (yes or no). However, the space transformation
 * is the only information required to compute the Tensor \f$g\f$ which is a \f$3\times3\f$
 * matrix whilch (multiplied by the material value of the untransfomred
 * coordinate either inside or outside the waveguide) gives us the value of
 * \f$\epsilon\f$ and \f$\mu\f$. From this class we derive several different
 * classes which then specify the interface specified in this class. 
 * \author Pascal Kraft 
 * \date 17.12.2015
 */

class SpaceTransformation {
 public:

  bool apply_math_to_phys = true;

  SpaceTransformation();

  virtual Position math_to_phys(Position coord) const = 0;

  virtual Position phys_to_math(Position coord) const = 0;

  virtual double get_det(Position ) {
    return 1.0;
  }

  virtual Tensor<2,3,double> get_J(Position &) {
    Tensor<2,3,double> ret;
    ret[0][0] = 1;
    ret[1][1] = 1;
    ret[2][2] = 1;
    return ret;
  }

  virtual Tensor<2,3,double> get_J_inverse(Position &) {
    Tensor<2,3,double> ret;
    ret[0][0] = 1;
    ret[1][1] = 1;
    ret[2][2] = 1;
    return ret;
  }

  virtual Tensor<2, 3, ComplexNumber> get_Tensor(Position &) = 0;

  virtual Tensor<2, 3, double> get_Space_Transformation_Tensor(Position &) = 0;

  virtual Tensor<2, 3, ComplexNumber> get_Tensor_for_step(Position &coordinate, unsigned int dof, double step_width);

  void switch_application_mode(bool apply_math_to_physical);

  /**
   * At the beginning (before the first solution of a system) only the boundary
   * conditions for the shape of the waveguide are known. Therefore the values
   * for the degrees of freedom need to be estimated. This function sets all
   * variables to appropiate values and estimates an appropriate shape based on
   * averages and a polynomial interpolation of the boundary conditions on the
   * shape.
   */
  virtual void estimate_and_initialize() = 0;

  /**
   * This is a getter for the values of degrees of freedom. A getter-setter
   * interface was introduced since the values are estimated automatically
   * during the optimization and non-physical systems should be excluded from
   * the domain of possible cases. \param dof The index of the degree of freedom
   * to be retrieved from the structure of the modelled waveguide. \return This
   * function returns the value of the requested degree of freedom. Should this
   * dof not exist, 0 will be returned.
   */
  virtual double get_dof(int) const {
    return 0;
  };

  /**
   * This is a getter for the values of degrees of freedom. A getter-setter
   * interface was introduced since the values are estimated automatically
   * during the optimization and non-physical systems should be excluded from
   * the domain of possible cases. \param dof The index of the degree of freedom
   * to be retrieved from the structure of the modelled waveguide. \return This
   * function returns the value of the requested degree of freedom. Should this
   * dof not exist, 0 will be returnd.
   */
  virtual double get_free_dof(int) const { return 0.0; };

  /**
   * This function sets the value of the dof provided to the given value. It is
   * important to consider, that some dofs are non-writable (i.e. the values of
   * the degrees of freedom on the boundary, like the radius of the
   * input-connector cannot be changed). \param dof The index of the parameter
   * to be changed. \param value The value, the dof should be set to.
   */
  virtual void set_free_dof(int , double ){return;};

  /**
   * Using this method unifies the usage of coordinates. This function takes a
   * global \f$z\f$ coordinate (in the computational domain) and returns both a
   * Sector-Index and an internal \f$z\f$ coordinate indicating which sector
   * this coordinate belongs to and how far along in the sector it is located.
   * \param double in_z global system \f$z\f$ coordinate for the transformation.
   */
  virtual std::pair<int, double> Z_to_Sector_and_local_z(double in_z) const;

  /**
   * Other objects can use this function to retrieve an array of the current
   * values of the degrees of freedom of the functional we are optimizing. This
   * also includes restrained degrees of freedom and other functions can be used
   * to determine this property. This has to be done because in different cases
   * the number of restrained degrees of freedom can vary and we want no logic
   * about this in other functions.
   */
  virtual Vector<double> get_dof_values() const {
    Vector<double> ret;
    return ret;
  };

  /**
   * This function returns the number of unrestrained degrees of freedom of the
   * current optimization run.
   */
  virtual unsigned int n_free_dofs() const {
    return 0;
  }

  /**
   * This function returns the total number of DOFs including restrained ones.
   * This is the lenght of the array returned by Dofs().
   */
  virtual unsigned int n_dofs() const  {
    return 0;
  }

  /**
   * Console output of the current Waveguide Structure.
   */
  virtual void Print() const = 0;

  Position operator()(Position ) const;

};
