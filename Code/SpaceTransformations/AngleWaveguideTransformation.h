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
 public:
  AngleWaveguideTransformation();

  virtual ~AngleWaveguideTransformation();

  Position math_to_phys(Position coord) const;

  Position phys_to_math(Position coord) const;

  dealii::Tensor<2, 3, ComplexNumber> get_Tensor(Position &coordinate) const;

  dealii::Tensor<2, 3, double> get_Space_Transformation_Tensor(Position &coordinate) const;

  /**
   * This member contains all the Sectors who, as a sum, form the complete
   * Waveguide. These Sectors are a partition of the simulated domain.
   */
  std::vector<Sector<2>> case_sectors;

  /**
   * Since the computational domain is split into subdomains (called sectors),
   * it is important to keep track of the amount of subdomains. This member
   * stores the number of Sectors the computational domain has been split into.
   */
  const int sectors;

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
   * This member calculates the value of Q1 for a provided \f$z\f$-coordinate.
   * This value is used in the transformation of the solution-vector in
   * transformed coordinates (solution of the system-matrix) to real coordinates
   * (physical field). \param z The value of Q1 is independent of \f$x\f$ and
   * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
   * function.
   */
  double get_Q1(double z) const;

  /**
   * This member calculates the value of Q2 for a provided \f$z\f$-coordinate.
   * This value is used in the transformation of the solution-vector in
   * transformed coordinates (solution of the system-matrix) to real coordinates
   * (physical field). \param z The value of Q2 is independent of \f$x\f$ and
   * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
   * function.
   */
  double get_Q2(double z) const;

  /**
   * This member calculates the value of Q3 for a provided \f$z\f$-coordinate.
   * This value is used in the transformation of the solution-vector in
   * transformed coordinates (solution of the system-matrix) to real coordinates
   * (physical field). \param z The value of Q3 is independent of \f$x\f$ and
   * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
   * function.
   */
  double get_Q3(double z) const;

  /**
   * This is a getter for the values of degrees of freedom. A getter-setter
   * interface was introduced since the values are estimated automatically
   * during the optimization and non-physical systems should be excluded from
   * the domain of possible cases. \param dof The index of the degree of freedom
   * to be retrieved from the structure of the modelled waveguide. \return This
   * function returns the value of the requested degree of freedom. Should this
   * dof not exist, 0 will be returnd.
   */
  double get_dof(int dof) const;

  /**
   * This function sets the value of the dof provided to the given value. It is
   * important to consider, that some dofs are non-writable (i.e. the values of
   * the degrees of freedom on the boundary, like the radius of the
   * input-connector cannot be changed). \param dof The index of the parameter
   * to be changed. \param value The value, the dof should be set to.
   */
  void set_dof(int dof, double value);

  /**
   * This is a getter for the values of degrees of freedom. A getter-setter
   * interface was introduced since the values are estimated automatically
   * during the optimization and non-physical systems should be excluded from
   * the domain of possible cases. \param dof The index of the degree of freedom
   * to be retrieved from the structure of the modelled waveguide. \return This
   * function returns the value of the requested degree of freedom. Should this
   * dof not exist, 0 will be returnd.
   */
  double get_free_dof(int dof) const;

  /**
   * This function sets the value of the dof provided to the given value. It is
   * important to consider, that some dofs are non-writable (i.e. the values of
   * the degrees of freedom on the boundary, like the radius of the
   * input-connector cannot be changed). \param dof The index of the parameter
   * to be changed. \param value The value, the dof should be set to.
   */
  void set_free_dof(int dof, double value);

  /**
   * Returns the complete length of the computational domain.
   */
  double System_Length() const;

  /**
   * Returns the length of one sector
   */
  double Sector_Length() const;

  /**
   * Returns the length of one layer
   */
  double Layer_Length() const;

  /**
   * Returns the radius for a system-coordinate;
   */
  double get_r(double in_z) const;

  /**
   * Returns the shift for a system-coordinate;
   */
  double get_m(double in_z) const;

  /**
   * Returns the tilt for a system-coordinate;
   */
  double get_v(double in_z) const;

  /**
   * This vector of values saves the initial configuration
   */
  Vector<double> InitialDofs;

  /**
   * Other objects can use this function to retrieve an array of the current
   * values of the degrees of freedom of the functional we are optimizing. This
   * also includes restrained degrees of freedom and other functions can be used
   * to determine this property. This has to be done because in different cases
   * the number of restrained degrees of freedom can vary and we want no logic
   * about this in other functions.
   */
  Vector<double> Dofs() const;

  /**
   * This function returns the number of unrestrained degrees of freedom of the
   * current optimization run.
   */
  unsigned int NFreeDofs() const;

  /**
   * This function returns the total number of DOFs including restrained ones.
   * This is the lenght of the array returned by Dofs().
   */
  unsigned int NDofs() const;

  /**
   * Since Dofs() also returns restrained degrees of freedom, this function can
   * be applied to determine if a degree of freedom is indeed free or
   * restrained. "restrained" means that for example the DOF represents the
   * radius at one of the connectors (input or output) and therefore we forbid
   * the optimization scheme to vary this value.
   */
  bool IsDofFree(int) const;

  /**
   * Console output of the current Waveguide Structure.
   */
  void Print() const;

  ComplexNumber evaluate_for_z_with_sum(double, double, InnerDomain *);

  ComplexNumber evaluate_for_z(double z_in, InnerDomain *);

  Tensor<2, 3, double> transformation_tensor;
};