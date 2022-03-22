#pragma once

/**
 * @file Sector.h
 * @author your name (you@domain.com)
 * @brief Contains the header of the Sector class.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <deal.II/base/tensor.h>

/**
 * \class Sector
 * \brief Sectors are used, to split the computational domain into chunks, whose
 * degrees of freedom are likely coupled.
 *
 * The interfaces between Sectors lie in the xy-plane and they are ordered by
 * their z-value. \author Pascal Kraft \date 17.12.2015
 */
template <unsigned int Dofs_Per_Sector>
class Sector {
 public:
  /**
   * Constructor of the Sector class, that takes all important properties as an
   * input property. \param in_left stores if the sector is at the left end. It
   * is used to initialize the according variable. \param in_right stores if the
   * sector is at the right end. It is used to initialize the according
   * variable. \param in_z_0 stores the z-coordinate of the left surface-plain.
   * It is used to initialize the according variable. \param in_z_1 stores the
   * z-coordinate of the right surface-plain. It is used to initialize the
   * according variable.
   */
  Sector(bool in_left, bool in_right, double in_z_0, double in_z_1);

  ~Sector();

  /**
   * This value describes, if this Sector is at the left (small z) end of the
   * computational domain.
   */
  const bool left;
  /**
   * This value describes, if this Sector is at the right (large z) end of the
   * computational domain.
   */
  const bool right;
  /**
   * This value is true, if either left or right are true.
   */
  const bool boundary;

  const double z_0;

  /**
   * The objects created from this class are supposed to hand back the material
   * properties which include the space-transformation Tensors. For this to be
   * possible, the Sector has to be able to transform from global coordinates to
   * coordinates that are scaled inside the Sector. For this purpose, the z_0
   * and z_1 variables store the z-coordinate of both, the left and right
   * surface.
   */
  const double z_1;

  /**
   * This method gets called from the WaveguideStructure object used in the
   * simulation. This is where the Waveguide object gets the material Tensors to
   * build the system-matrix. This method returns a complex-values Matrix
   * containing the system-tensors \f$\boldsymbol{\mu^{-1}} \f$ and
   * \f$\boldsymbol{\epsilon} \f$. \param in_x x-coordinate of the point, for
   * which the Tensor should be calculated. \param in_y y-coordinate of the
   * point, for which the Tensor should be calculated. \param in_z z-coordinate
   * of the point, for which the Tensor should be calculated.
   */
  dealii::Tensor<2, 3, double> TransformationTensorInternal(double in_x, double in_y, double in_z) const;

  /**
   * This function is used during the optimization-operation to update the
   * properties of the space-transformation. However, to ensure, that the
   * boundary-conditions remain intact, this function cannot edit the left
   * degrees of freedom if left is true and it cannot edit the right degrees of
   * freedom if right is true
   */
  void set_properties(double m_0, double m_1, double r_0, double r_1);

  void set_properties(double m_0, double m_1, double r_0, double r_1, double v_0, double v_1);

  /**
   * This function is the same as set_properties with the difference of being
   * able to change the values of the input- and output boundary.
   */
  void set_properties_force(double m_0, double m_1, double r_0, double r_1);

  void set_properties_force(double m_0, double m_1, double r_0, double r_1, double v_0, double v_1);

  /**
   * The values of Q1, Q2 and Q3 are needed to compute the solution in real
   * coordinates from the one in trnsformed coordinates. This function returnes
   * Q1 for a given position and the current transformation.
   */
  double getQ1(double) const;

  /**
   * The values of Q1, Q2 and Q3 are needed to compute the solution in real
   * coordinates from the one in transformed coordinates. This function returnes
   * Q2 for a given position and the current transformation.
   */
  double getQ2(double) const;

  /**
   * The values of Q1, Q2 and Q3 are needed to compute the solution in real
   * coordinates from the one in transformed coordinates. This function returnes
   * Q3 for a given position and the current transformation.
   */
  double getQ3(double) const;

  /**
   * This function returns the number of the lowest degree of freedom associated
   * with this Sector. Keep in mind, that the degrees of freedom associated with
   * edges on the lower (small \f$ z \f$) interface are not included since this
   * functionality is supposed to help in the block-structure generation and
   * those dofs are part of the neighboring block.
   */
  unsigned int getLowestDof() const;

  /**
   * This function returns the number of dofs which are part of this sector. The
   * same remarks as for getLowestDof() apply.
   */
  unsigned int getNDofs() const;

  /**
   * In order to set appropriate boundary conditions it makes sense to
   * determine, which degrees are associated with an edge which is part of an
   * interface to another sector. Due to the reordering of dofs this is
   * especially easy since the dofs on the interface are those in the interval
   * \f[[\,\operatorname{LowestDof} + \operatorname{NDofs} -
   * \operatorname{NInternalBoundaryDofs}\, , \,  \operatorname{LowestDof} +
   * \operatorname{NDofs}\,]\f]
   */
  unsigned int getNInternalBoundaryDofs() const;

  /**
   * This function can be used to query the number of cells in a Sector /
   * subdomain. In this case there are no problems with interface-dofs. Every
   * cell belongs to exactly one sector (the problem arises from the fact, that
   * one edge can (and most of the time will) belong to more then one cell).
   */
  unsigned int getNActiveCells() const;

  /**
   * Setter for the value that the getter should return. Called after
   * Dof-reordering.
   */
  void setLowestDof(unsigned int);

  /**
   * Setter for the value that the getter should return. Called after
   * Dof-reordering.
   */
  void setNDofs(unsigned int);

  /**
   * Setter for the value that the getter should return. Called after
   * Dof-reordering.
   */
  void setNInternalBoundaryDofs(unsigned int);

  /**
   * Setter for the value that the getter should return. Called after
   * Dof-reordering.
   */
  void setNActiveCells(unsigned int);

  /**
   * This function returns the value of a specified dof at a given internal
   * position. \param i index of the dof. This class has a template argument
   * specifying the number of dofs per sector. This argument has to be less or
   * equal. \param z this is a relative value for interpolation with \f$z \in
   * [0,1]\f$. If \f$z = 0\f$ the values for the lower end of the sector are
   * returned. If \f$z=1\f$ the values for the upper end of the sector are
   * returned. In between the values are interpolated according to the rules for
   * the specific dof.
   */
  double get_dof(unsigned int i, double z) const;

  /**
   * Get an interpolation of the radius for a coordinate z
   * \param double z is the \f$z \in [0,1]\f$ coordinate for the interpolation.
   */
  double get_r(double z) const;

  /**
   * Get an interpolation of the tilt for a coordinate z
   * \param double z is the \f$z \in [0,1]\f$ coordinate for the interpolation.
   */
  double get_v(double z) const;

  /**
   * Get an interpolation of the shift for a coordinate z
   * \param double z is the \f$z \in [0,1]\f$ coordinate for the interpolation.
   */
  double get_m(double z) const;

  // private:

  unsigned int LowestDof, NDofs, NInternalBoundaryDofs, NActiveCells;

  // Stores the values of the left degrees of freedom. These are the values of
  // the dofs on the surface of the sector towards z--> -\infty.
  std::vector<double> dofs_l;

  // Stores the values of the right degrees of freedom. These are the values of
  // the dofs on the surface of the sector towards z--> \infty.
  std::vector<double> dofs_r;

  //  This stores the indices for derivative dpendence. For example: If the
  //  third dof is the dof type 'm' (in former nomenclature) and the second is
  //  of type 'v' so v should be the derivative of 'm' then the derivative[3]=2.
  //  This is used in the function get_dof.
  std::vector<unsigned int> derivative;

  // this stores for any dof if it has zero_derivative (at sector interfaces) or
  // not.
  std::vector<bool> zero_derivative;
};
