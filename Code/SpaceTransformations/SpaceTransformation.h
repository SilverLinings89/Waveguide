#ifndef SPACETRANSFORMATION_H_
#define SPACETRANSFORMATION_H_


#include <math.h>
#include <vector>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>


using namespace dealii;

class SpaceTransformation {

  const unsigned int dofs_per_layer;

  const unsigned int boundary_dofs_in;

  const unsigned int boundary_dofs_out;

  SpaceTransformation();

  Point<3> math_to_phys(Point<3> coord);

  Point<3> phys_to_math(Point<3> coord);

  bool is_identity(Point<3> coord);

  Tensor<2,3, std::complex<double>> get_epsilon(Point<3> coordinate);

  Tensor<2,3, std::complex<double>> get_mu(Point<3> coordinate);

  Tensor<2,3, double> get_transformation_tensor(Point<3> coordinate);

  /**
   * This member contains all the Sectors who, as a sum, form the complete Waveguide. These Sectors are a partition of the simulated domain.
   */
  std::vector<Sector> case_sectors;
  /**
   * The material-property \f$\epsilon_r\f$ has a different value inside and outside of the waveguides core. This variable stores its value inside the core.
   */
  const double epsilon_K;
  /**
   *  The material-property \f$\epsilon_r\f$ has a different value inside and outside of the waveguides core. This variable stores its value outside the core.
   */
  const double epsilon_M;
  /**
   * Since the computational domain is split into subdomains (called sectors), it is important to keep track of the amount of subdomains. This member stores the number of Sectors the computational domain has been split into.
   */
  const int sectors;

  /**
   * This value is initialized with the value Delta from the input-file.
   */
  const double deltaY;

  /**
   * At the beginning (before the first solution of a system) only the boundary conditions for the shape of the waveguide are known. Therefore the values for the degrees of freedom need to be estimated. This function sets all variables to appropiate values and estimates an appropriate shape based on averages and a polynomial interpolation of the boundary conditions on the shape.
   */
  void  estimate_and_initialize();

  /**
   * This member calculates the value of Q1 for a provided \f$z\f$-coordinate. This value is used in the transformation of the solution-vector in transformed coordinates (solution of the system-matrix) to real coordinates (physical field).
   * \param z The value of Q1 is independent of \f$x\f$ and \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the function.
   */
  double  get_Q1 ( double z);

  /**
   * This member calculates the value of Q2 for a provided \f$z\f$-coordinate. This value is used in the transformation of the solution-vector in transformed coordinates (solution of the system-matrix) to real coordinates (physical field).
   * \param z The value of Q2 is independent of \f$x\f$ and \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the function.
   */
  double  get_Q2 ( double z);

  /**
   * This member calculates the value of Q3 for a provided \f$z\f$-coordinate. This value is used in the transformation of the solution-vector in transformed coordinates (solution of the system-matrix) to real coordinates (physical field).
   * \param z The value of Q3 is independent of \f$x\f$ and \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the function.
   */
  double  get_Q3 ( double z);


  /**
   * This is a getter for the values of degrees of freedom. A getter-setter interface was introduced since the values are estimated automatically during the optimization and non-physical systems should be excluded from the domain of possible cases.
   * \param dof The index of the degree of freedom to be retrieved from the structure of the modelled waveguide.
   * \return This function returns the value of the requested degree of freedom. Should this dof not exist, 0 will be returnd.
   */
  double  get_dof (int dof, bool free);

  /**
   * This function sets the value of the dof provided to the given value. It is important to consider, that some dofs are non-writable (i.e. the values of the degrees of freedom on the boundary, like the radius of the input-connector cannot be changed).
   * \param dof The index of the parameter to be changed.
   * \param value The value, the dof should be set to.
   */
  void  set_dof (int dof , double value, bool free );

  /**
   * Using this method unifies the usage of coordinates. This function takes a global \f$z\f$ coordinate (in the computational domain) and returns both a Sector-Index and an internal \f$z\f$ coordinate indicating which sector this coordinate belongs to and how far along in the sector it is located.
   * \param double in_z global system \f$z\f$ coordinate for the transformation.
   */
  std::pair<int, double> Z_to_Sector_and_local_z(double in_z);

  /**
   * Returns the complete length of the computational domain.
   */
  double System_Length();

  /**
   * Returns the length of one sector
   */
  double Sector_Length();

  /**
   * Returns the length of one layer
   */
  double Layer_Length();

  /**
   * Returns the radius for a system-coordinate;
   */
  double get_r(double in_z);

  /**
   * Returns the shift for a system-coordinate;
   */
  double get_m(double in_z);

  /**
   * Returns the tilt for a system-coordinate;
   */
  double get_v(double in_z);

  /**
   * This Method writes a comprehensive description of the current structure to the console.
   */
  void WriteConfigurationToConsole();

  int Z_to_Layer(double);

  /**
   * This vector of values saves the initial configuration
   */
  Vector<double> InitialDofs;

  /**
   * This vector of values saves the initial configuration
   */
  double InitialQuality;

  Vector<double> Dofs();

  unsigned int NFreeDofs();

  unsigned int NDofs();

  bool IsDofFree(int );

  void Print();


};

#endif SPACETRANSFORMATION_H_
 
