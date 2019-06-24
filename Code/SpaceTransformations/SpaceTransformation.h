#ifndef SPACETRANSFORMATION_H_
#define SPACETRANSFORMATION_H_

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <math.h>
#include <vector>

using namespace dealii;

enum Evaluation_Domain { CIRCLE_CLOSE, CIRCLE_MAX, RECTANGLE_INNER };

enum Evaluation_Metric { FUNDAMENTAL_MODE_EXCITATION, POYNTING_TYPE_ENERGY };

class NumericProblem;
/**
 * \class SpaceTransformation
 * \brief The SpaceTransformation class encapsulates the coordinate
 * transformation used in the simulation.
 *
 * Two important decisions have to be made in the computation: Which shape
 * should be used for the waveguide? This can either be rectangular or tubular.
 * Should the coordinate-transformation always be equal to identity in any
 * domain where PML is applied? (yes or no). However, the space transformation
 * is the only information required to compute the Tensor \f$g\f$ which is a 3x3
 * matrix whilch (multiplied by the material value of the untransfomred
 * coordinate either inside or outside the waveguide) gives us the value of
 * \f$\epsilon\f$ and \f$\mu\f$. From this class we derive several different
 * classes which then specify the interface specified in this class. \author
 * Pascal Kraft \date 17.12.2015
 */
class SpaceTransformation {
 public:
  bool homogenized = false;

  const unsigned int dofs_per_layer;

  const unsigned int boundary_dofs_in;

  const unsigned int boundary_dofs_out;

  SpaceTransformation(int, int);

  virtual Point<3> math_to_phys(Point<3> coord) const = 0;

  virtual Point<3> phys_to_math(Point<3> coord) const = 0;

  bool is_identity(Point<3> coord) const;

  virtual Tensor<2, 3, std::complex<double>> get_Tensor(
      Point<3> &coordinate) const = 0;

  virtual Tensor<2, 3, std::complex<double>> get_Preconditioner_Tensor(
      Point<3> &coordinate, int block) const = 0;

  virtual Tensor<2, 3, double> get_Space_Transformation_Tensor(
      Point<3> &coordinate) const = 0;

  virtual Tensor<2, 3, double> get_Space_Transformation_Tensor_Homogenized(
      Point<3> &coordinate) const = 0;

  virtual Tensor<2, 3, std::complex<double>> Apply_PML_To_Tensor(
      Point<3> &coordinate, Tensor<2, 3, double> Tensor_input) const = 0;

  virtual Tensor<2, 3, std::complex<double>>
      Apply_PML_To_Tensor_For_Preconditioner(Point<3> &coordinate,
                                             Tensor<2, 3, double> Tensor_input,
                                             int block) const = 0;

  virtual Tensor<2, 3, std::complex<double>> get_Tensor_for_step(
      Point<3> &coordinate, unsigned int dof, double step_width);

  /**
   * This function is used to determine, if a system-coordinate belongs to a
   * PML-region for the PML that limits the computational domain along the
   * x-axis. Since there are 3 blocks of PML-type material, there are 3
   * functions. \param position Stores the position in which to test for
   * presence of a PML-Material.
   */
  virtual bool PML_in_X(Point<3> &position) const = 0;
  /**
   * This function is used to determine, if a system-coordinate belongs to a
   * PML-region for the PML that limits the computational domain along the
   * y-axis. Since there are 3 blocks of PML-type material, there are 3
   * functions. \param position Stores the position in which to test for
   * presence of a PML-Material.
   */
  virtual bool PML_in_Y(Point<3> &position) const = 0;
  /**
   * This function is used to determine, if a system-coordinate belongs to a
   * PML-region for the PML that limits the computational domain along the
   * z-axis. Since there are 3 blocks of PML-type material, there are 3
   * functions. \param position Stores the position in which to test for
   * presence of a PML-Material.
   */
  virtual bool PML_in_Z(Point<3> &position) const = 0;

  /**
   * This function fulfills the same purpose as those with similar names but it
   * is supposed to be used together with Preconditioner_PML_in_Z instead of the
   * versions without "Preconditioner".
   */
  virtual double Preconditioner_PML_Z_Distance(Point<3> &p,
                                               unsigned int block) const = 0;

  /**
   * This function calculates for a given point, its distance to a PML-boundary
   * limiting the computational domain. This function is used merely to make
   * code more readable. There is a function for every one of the dimensions
   * since the normal vectors of PML-regions in this implementation are the
   * coordinate-axis. This value is set to zero outside the PML and positive
   * inside both PML-domains (only one for the z-direction). \param position
   * Stores the position from which to calculate the distance to the
   * PML-surface.
   */
  virtual double PML_X_Distance(Point<3> &position) const = 0;
  /**
   * This function calculates for a given point, its distance to a PML-boundary
   * limiting the computational domain. This function is used merely to make
   * code more readable. There is a function for every one of the dimensions
   * since the normal vectors of PML-regions in this implementation are the
   * coordinate-axis. This value is set to zero outside the PML and positive
   * inside both PML-domains (only one for the z-direction). \param position
   * Stores the position from which to calculate the distance to the
   * PML-surface.
   */

  virtual double PML_Y_Distance(Point<3> &position) const = 0;
  /**
   * This function calculates for a given point, its distance to a PML-boundary
   * limiting the computational domain. This function is used merely to make
   * code more readable. There is a function for every one of the dimensions
   * since the normal vectors of PML-regions in this implementation are the
   * coordinate-axis. This value is set to zero outside the PML and positive
   * inside both PML-domains (only one for the z-direction). \param position
   * Stores the position from which to calculate the distance to the
   * PML-surface.
   */
  virtual double PML_Z_Distance(Point<3> &position) const = 0;

  /**
   * The material-property \f$\epsilon_r\f$ has a different value inside and
   * outside of the waveguides core. This variable stores its value inside the
   * core.
   */
  const double epsilon_K;
  /**
   *  The material-property \f$\epsilon_r\f$ has a different value inside and
   * outside of the waveguides core. This variable stores its value outside the
   * core.
   */
  const double epsilon_M;
  /**
   * Since the computational domain is split into subdomains (called sectors),
   * it is important to keep track of the amount of subdomains. This member
   * stores the number of Sectors the computational domain has been split into.
   */
  const int sectors;

  /**
   * This value is initialized with the value Delta from the input-file.
   */
  const double deltaY;

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
   * This member calculates the value of Q1 for a provided \f$z\f$-coordinate.
   * This value is used in the transformation of the solution-vector in
   * transformed coordinates (solution of the system-matrix) to real coordinates
   * (physical field). \param z The value of Q1 is independent of \f$x\f$ and
   * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
   * function.
   */
  virtual double get_Q1(double z) const = 0;

  /**
   * This member calculates the value of Q2 for a provided \f$z\f$-coordinate.
   * This value is used in the transformation of the solution-vector in
   * transformed coordinates (solution of the system-matrix) to real coordinates
   * (physical field). \param z The value of Q2 is independent of \f$x\f$ and
   * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
   * function.
   */
  virtual double get_Q2(double z) const = 0;

  /**
   * This member calculates the value of Q3 for a provided \f$z\f$-coordinate.
   * This value is used in the transformation of the solution-vector in
   * transformed coordinates (solution of the system-matrix) to real coordinates
   * (physical field). \param z The value of Q3 is independent of \f$x\f$ and
   * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
   * function.
   */
  virtual double get_Q3(double z) const = 0;

  /**
   * This is a getter for the values of degrees of freedom. A getter-setter
   * interface was introduced since the values are estimated automatically
   * during the optimization and non-physical systems should be excluded from
   * the domain of possible cases. \param dof The index of the degree of freedom
   * to be retrieved from the structure of the modelled waveguide. \return This
   * function returns the value of the requested degree of freedom. Should this
   * dof not exist, 0 will be returned.
   */
  virtual double get_dof(int dof) const = 0;

  /**
   * This function sets the value of the dof provided to the given value. It is
   * important to consider, that some dofs are non-writable (i.e. the values of
   * the degrees of freedom on the boundary, like the radius of the
   * input-connector cannot be changed). \param dof The index of the parameter
   * to be changed. \param value The value, the dof should be set to.
   */
  virtual void set_dof(int dof, double value) = 0;

  virtual std::pair<double, double> dof_support(unsigned int index) const;

  bool point_in_dof_support(Point<3> location, unsigned int dof_index) const;

  /**
   * This is a getter for the values of degrees of freedom. A getter-setter
   * interface was introduced since the values are estimated automatically
   * during the optimization and non-physical systems should be excluded from
   * the domain of possible cases. \param dof The index of the degree of freedom
   * to be retrieved from the structure of the modelled waveguide. \return This
   * function returns the value of the requested degree of freedom. Should this
   * dof not exist, 0 will be returnd.
   */
  virtual double get_free_dof(int dof) const = 0;

  /**
   * This function sets the value of the dof provided to the given value. It is
   * important to consider, that some dofs are non-writable (i.e. the values of
   * the degrees of freedom on the boundary, like the radius of the
   * input-connector cannot be changed). \param dof The index of the parameter
   * to be changed. \param value The value, the dof should be set to.
   */
  virtual void set_free_dof(int dof, double value) = 0;

  /**
   * Using this method unifies the usage of coordinates. This function takes a
   * global \f$z\f$ coordinate (in the computational domain) and returns both a
   * Sector-Index and an internal \f$z\f$ coordinate indicating which sector
   * this coordinate belongs to and how far along in the sector it is located.
   * \param double in_z global system \f$z\f$ coordinate for the transformation.
   */
  std::pair<int, double> Z_to_Sector_and_local_z(double in_z) const;

  /**
   * Returns the length of one sector
   */
  double Sector_Length() const;

  /**
   * Returns the radius for a system-coordinate;
   */
  virtual double get_r(double in_z) const = 0;

  /**
   * Returns the shift for a system-coordinate;
   */
  virtual double get_m(double in_z) const = 0;

  /**
   * Returns the tilt for a system-coordinate;
   */
  virtual double get_v(double in_z) const = 0;

  /**
   * This Method writes a comprehensive description of the current structure to
   * the console.
   */
  // virtual void WriteConfigurationToConsole()=0;

  int Z_to_Layer(double) const;

  /**
   * This vector of values saves the initial configuration
   */
  Vector<double> InitialDofs;

  /**
   * This vector of values saves the initial configuration
   */
  double InitialQuality;

  /**
   * Other objects can use this function to retrieve an array of the current
   * values of the degrees of freedom of the functional we are optimizing. This
   * also includes restrained degrees of freedom and other functions can be used
   * to determine this property. This has to be done because in different cases
   * the number of restrained degrees of freedom can vary and we want no logic
   * about this in other functions.
   */
  virtual Vector<double> Dofs() const = 0;

  /**
   * This function returns the number of unrestrained degrees of freedom of the
   * current optimization run.
   */
  virtual unsigned int NFreeDofs() const = 0;

  /**
   * This function returns the total number of DOFs including restrained ones.
   * This is the lenght of the array returned by Dofs().
   */
  virtual unsigned int NDofs() const = 0;

  /**
   * Since Dofs() also returns restrained degrees of freedom, this function can
   * be applied to determine if a degree of freedom is indeed free or
   * restrained. "restrained" means that for example the DOF represents the
   * radius at one of the connectors (input or output) and therefore we forbid
   * the optimization scheme to vary this value.
   */
  virtual bool IsDofFree(int) const = 0;

  /**
   * Console output of the current Waveguide Structure.
   */
  virtual void Print() const = 0;

  std::complex<double> evaluate_for_z_with_sum(double, Evaluation_Domain,
                                               Evaluation_Metric, NumericProblem *);

  std::complex<double> gauss_product_2D_sphere(double z, int n, double R,
                                               double Xc, double Yc,
                                               NumericProblem *in_w,
                                               Evaluation_Metric in_m);

  std::complex<double> integrate_Waveguide_Core_2D(double z, int n,
                                                   NumericProblem *in_w,
                                                   Evaluation_Metric in_m);

  const int rank;
};

#endif
