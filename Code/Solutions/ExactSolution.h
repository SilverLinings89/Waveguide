#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>
#include "../Helpers/PointVal.h"
#include "../GlobalObjects/GlobalObjects.h"
#include <deal.II/base/function_lib.h>

/**
 * \class ExactSolution
 * \brief This class is derived from the Function class and can be used to
 * estimate the L2-error for a straight waveguide. In the case of a completely
 * cylindrical waveguide, an analytic solution is known (the modes of the
 * input-signal themselves) and this class offers a representation of this
 * analytical solution. If the waveguide has any other shape, this solution does
 * not lose its value completely - it can still be used as a starting-vector for
 * iterative solvers.
 *
 * The structure of this class is defined by the properties of the
 * Function-class meaning that we have two functions:
 *  -#  virtual double value (const Point<dim> &p, const unsigned int component
 * ) calculates the value for a single component of the vector-valued
 * return-value.
 *  -#  virtual void vector_value (const Point<dim> &p,	Vector<double> &value)
 * puts these individual components into the parameter value, which is a
 * reference to a vector, handed over to store the result.
 *
 * \author Pascal Kraft
 * \date 23.11.2015
 */

class ExactSolution: public dealii::Function<3, ComplexNumber> {
 private:
  /** @brief Not used. */
  std::vector<float> mesh_points;
  /** @brief Not used. */
  PointVal **vals;
  /** @brief The imaginary unit i. */
  const ComplexNumber imaginary_unit;

 public:
  /** @brief Bilinear interpolant of the profile of E_x over the cross section, a function of (x, y). */
  dealii::Functions::InterpolatedUniformGridData<2> component_x;
  /** @brief Bilinear interpolant of the profile of E_y over the cross section, a function of (x, y). */
  dealii::Functions::InterpolatedUniformGridData<2> component_y;
  /**
   * @brief Bilinear interpolant of the stored profile of the longitudinal component over the cross section.
   * The stored values are real; E_z is -i times this function (see the remark in ExactSolution.cpp).
   */
  dealii::Functions::InterpolatedUniformGridData<2> component_z;
  /** @brief Values of the profile of E_x on the data grid, index [i_x][i_y]. Normalized, see load_data(). */
  static dealii::Table<2,double> data_table_x;
  /** @brief Values of the profile of E_y on the data grid, index [i_x][i_y]. Normalized, see load_data(). */
  static dealii::Table<2,double> data_table_y;
  /** @brief Stored values of the longitudinal component on the data grid, index [i_x][i_y]. Normalized, see load_data(). */
  static dealii::Table<2,double> data_table_z;
  /** @brief Extent of the data grid in micrometers: ranges[0] is (x_min, x_max) and ranges[1] is (y_min, y_max). */
  static std::array<std::pair<double, double>, 2> ranges;
  /** @brief Number of intervals of the uniform data grid in the x-direction (index 0) and the y-direction (index 1). */
  static std::array<unsigned int, 2> n_intervals;
  /** @brief Relative permittivity of the cross-section on the data grid, index [i_x][i_y]. Empty if the file has no such column. */
  static dealii::Table<2,double> data_table_epsilon;
  /** @brief True if the loaded file contains the permittivity of the cross-section (sixth column). */
  static bool has_permittivity;
  /** @brief Propagation constant from the header of the loaded file ("# beta = ..."), 0 if the file has no header. */
  static double file_beta;
  /** @brief Vacuum wavelength from the header of the loaded file ("# wavelength = ..."), 0 if the file has no header. */
  static double file_wavelength;
  /** @brief Description of the structure from the header of the loaded file ("# structure = ..."), empty without header. */
  static std::string file_structure;
  /** @brief Name of the mode from the header of the loaded file ("# mode = ..."), empty without header. */
  static std::string file_mode;
  /** @brief Smallest and largest value of the permittivity column. */
  static std::pair<double, double> permittivity_range;
  /** @brief Creates the interpolants from the static tables. load_data() has to be called before the first object is created. */
  ExactSolution();

  /**
   * @brief Reads the mode profile from a text file into the static tables.
   * Every line contains x, y and the three real values of the profile (E_x, E_y, longitudinal component), separated by
   * single spaces, and optionally the relative permittivity at the point as a sixth value. Lines that start with '#'
   * are a header with entries "# key = value" (beta, wavelength, structure, mode), see modesolver/modefile.py. The points form a uniform grid and are ordered such that x runs fastest. All values are divided by
   * the largest Euclidean norm of the three values of a point. Throws std::runtime_error if the file cannot be opened.
   *
   * @param fname path of the file (Modes/output_mode.dat in the repository).
   */
  static void load_data(std::string fname);

  /**
   * @brief Finds a mode file: an absolute path or a path that exists relative to the working directory is returned
   * unchanged, every other path is taken relative to the root of the repository.
   * @param fname the value of the case parameter "mode file".
   * @return std::string the path to open.
   */
  static std::string resolve_path(const std::string & fname);

  /**
   * @brief The permittivity of the cross-section stored in the mode file, taken from the nearest grid point.
   * Outside of the grid the value of the nearest point of its boundary is returned, which continues the
   * cross-section into the absorbing layers.
   * @param x x-coordinate in micrometers.
   * @param y y-coordinate in micrometers.
   * @return double relative permittivity; has_permittivity has to be true.
   */
  static double permittivity_at(double x, double y);

  /**
   * @brief One component of the mode E(x, y, z) = profile(x, y) exp(i beta (z - z_min)), where z_min is the lower end
   * of the global domain.
   * @param p position in micrometers. It has to be inside the data grid in x and y.
   * @param component 0, 1 or 2 for the x-, y- or z-component.
   * @return ComplexNumber the value of the component.
   */
  ComplexNumber value(const Position &p, const unsigned int component) const override;

  /**
   * @brief All three components of the mode, see value().
   * @param p position in micrometers.
   * @param value vector of size 3 that receives the field.
   */
  void vector_value(const Position &p, dealii::Vector<ComplexNumber> &value) const override;

  /**
   * @brief Curl of the mode.
   * The derivative in the z-direction is known analytically (factor i beta). The in-plane derivatives are the gradients
   * of the bilinear interpolants, which are piecewise constant in the direction of differentiation.
   * @param in_p position in micrometers.
   * @return dealii::Tensor<1, 3, ComplexNumber> the curl.
   */
  dealii::Tensor<1, 3, ComplexNumber> curl(const Position &in_p) const;

  /**
   * @brief The mode as a tensor; the same values as vector_value().
   * @param in_p position in micrometers.
   * @return dealii::Tensor<1, 3, ComplexNumber> the field.
   */
  dealii::Tensor<1, 3, ComplexNumber> val(const Position &in_p) const;
  
  /**
   * @brief The phase factor of the propagation in the z-direction.
   * @param in_p position in micrometers; only z is used.
   * @return ComplexNumber exp(i beta (z - z_min)) with the lower end z_min of the global domain.
   */
  ComplexNumber compute_phase_for_position(const Position &in_p) const;

  /**
   * @brief The propagation constant of the mode, computed from the effective permittivity and the wavelength.
   */
  static double beta();

  /**
   * @brief Drops the z-coordinate.
   * @param in_p position (x, y, z).
   * @return Position2D (x, y).
   */
  Position2D get_2D_position_from_3d(const Position & in_p) const;

  /**
   * @brief Values and derivatives of the profile that describe the source term of the mode. Not called anywhere.
   * With f the profile of E_x and h the profile of E_z (including the factor -i) it returns f, df/dx, d2f/dxdy,
   * d2f/dy2, h, dh/dx, dh/dy, d2h/dx2, d2h/dy2 and beta. The derivatives are central finite differences with the
   * spacing of the data grid.
   * @param in_p position (x, y) in micrometers.
   * @return J_derivative_terms the values.
   */
  J_derivative_terms get_derivative_terms(const Position2D & in_p) const;
};
