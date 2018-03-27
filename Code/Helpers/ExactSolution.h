
#ifndef ExactSolutionFlag_H_
#define ExactSolutionFlag_H_
#include <vector>
#include <string>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include "PointVal.h"

using namespace dealii;

/**
 * \class ExactSolution
 * \brief This class is derived from the Function class and can be used to estimate the L2-error for a straight waveguide. In the case of a completely cylindrical waveguide, an analytic solution is known (the modes of the input-signal themselves) and this class offers a representation of this analytical solution. If the waveguide has any other shape, this solution does not lose its value completely - it can still be used as a starting-vector for iterative solvers.
 *
 * The structure of this class is defined by the properties of the Function-class meaning that we have two functions:
 *  -#  virtual double value (const Point<dim> &p, const unsigned int component ) calculates the value for a single component of the vector-valued return-value.
 *  -#  virtual void vector_value (const Point<dim> &p,	Vector<double> &value) puts these individual components into the parameter value, which is a reference to a vector, handed over to store the result.
 *
 * \author Pascal Kraft
 * \date 23.11.2015
 */

class ExactSolution : public Function<3, double>
{
private:
  bool is_rectangular, is_dual;
  std::vector<float> mesh_points;
  PointVal** vals;

public:
  ExactSolution (bool in_rectangular = false, bool in_dual = false);

  /**
   * This function calculates one single component of the solution vector. To calculate this, we do the following: We know the input on the boundary of the computational domain for \f$z = z_{in}\f$. So for a given position \f$ p = (x,y,z)\f$ we calculate \f[ f_c(x,y,z) = \sum_{j=0}^N \left( a_j \, \boldsymbol{\phi_j}(x,y,z_{in}) \right) \cdot \boldsymbol{e_c} \, \mathrm{e}^{i \omega (z-z_{in})}.\f]
   * Here, \f$\boldsymbol{\phi_j}\f$ is the j-th mode of the waveguide which is induced with the intensity \f$ a_j\f$. \f$\boldsymbol{e_c}\f$ is the c-th unit-vector with \f$c\f$ being the index of the component we want to compute.
   * \param p This value contains the position for which we want to calculate the exact solution.
   * \param component This integer holds the index of the component we want to compute. Keep in mind that these are not coordinates in the physical sense. The components 0 to 2 are the real parts of the solution-vector and the components 3-5 are the imaginary parts.
   */
  double value (const Point<3> &p, const unsigned int component ) const;

  /**
   * This function is the one that gets called from external contexts and calls the value-function to calculate the individual components. The real solution looks as follows:
   * \f[ f(x,y,z) = \begin{pmatrix} \operatorname{value}(x,y,z,0) \\ \operatorname{value}(x,y,z,1)\\\operatorname{value}(x,y,z,2) \end{pmatrix} + i \begin{pmatrix}\operatorname{value}(x,y,z,3)\\\operatorname{value}(x,y,z,4)\\\operatorname{value}(x,y,z,5)\end{pmatrix}.\f]
   */


  void vector_value (const Point<3> &p,	Vector<double> &value) const;
  std::vector<std::string> split(std::string) const;
};

#endif
