#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include "ExactSolution.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/array_view.h>
#include <string>
#include <vector>

#include "../Core/InnerDomain.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Core/Types.h"
#include "../Helpers/PointVal.h"
#include "../Helpers/staticfunctions.h"

dealii::Table<2,double> ExactSolution::data_table_x;
dealii::Table<2,double> ExactSolution::data_table_y;
dealii::Table<2,double> ExactSolution::data_table_z;
std::array<std::pair<double, double>, 2> ExactSolution::ranges;
std::array<unsigned int, 2> ExactSolution::n_intervals;
dealii::Table<2,double> ExactSolution::data_table_epsilon;
bool ExactSolution::has_permittivity = false;
double ExactSolution::file_beta = 0;
double ExactSolution::file_wavelength = 0;
std::string ExactSolution::file_structure;
std::string ExactSolution::file_mode;
std::pair<double, double> ExactSolution::permittivity_range;
const ComplexNumber imaginary_unit(0,1);

// The mode profiles are stored as three real numbers (E_x, E_y, E_z) per point. They were computed for the
// convention exp(i (omega t - beta z)), in which the longitudinal component of a mode propagating in the +z
// direction is E_z = +i * (stored value). This code uses exp(i (beta z - omega t)) (see compute_phase_for_position
// and the sign of the PML), i.e. the complex conjugate. The longitudinal component therefore has to be imported as
// E_z = -i * (stored value). Only with this sign the imported field is divergence free
// (d/dx E_x + d/dy E_y + i beta E_z = 0) and therefore a mode travelling in the +z direction.
const ComplexNumber z_component_factor(0,-1);

double ExactSolution::beta() {
  return 2.0 * GlobalParams.Pi * std::sqrt(GlobalParams.Epsilon_R_effective) / GlobalParams.Lambda;
}

Position2D ExactSolution::get_2D_position_from_3d(const Position & in_p) const {
  Position2D p2d;
  p2d[0] = in_p[0];
  p2d[1] = in_p[1];
  return p2d;
}

ComplexNumber ExactSolution::value(const Position &in_p, const unsigned int component) const {
  Position2D p2d = get_2D_position_from_3d(in_p);
  ComplexNumber ret_val(0.0, 0.0);
  switch(component) {
    case 0:
      ret_val = component_x.value(p2d,0);
      break;
    case 1:
      ret_val = component_y.value(p2d,0);
      break;
    case 2:
      ret_val = z_component_factor * component_z.value(p2d,0);
      break;
    default:
      std::cout << "Error in call to ExactSolution::value. Invalid component requested." << std::endl;
  }
  return ret_val * compute_phase_for_position(in_p);
}

ComplexNumber ExactSolution::compute_phase_for_position(const Position &in_p) const {
  const double z =  in_p[2] - Geometry.global_z_range.first;
  return std::exp(beta() * z * imaginary_unit);
}

void ExactSolution::vector_value(const Position &in_p, Vector<ComplexNumber> &values) const {
  Position p = in_p;
  ComplexNumber phase = compute_phase_for_position(p);
  Position2D p2d = get_2D_position_from_3d(in_p);
  values[0] = component_x.value(p2d,0) * phase;
  values[1] = component_y.value(p2d,0) * phase;
  values[2] = z_component_factor * component_z.value(p2d,0) * phase;
  return;
}

Tensor<1, 3, ComplexNumber> ExactSolution::curl(const Position &in_p) const {
  // The field has the form E = (f(x,y), g(x,y), h(x,y)) * exp(i beta z), so the derivative in the z-direction is
  // known analytically and only the in-plane derivatives of the tabulated profile are required.
  const Position2D p2d = get_2D_position_from_3d(in_p);
  const ComplexNumber phase = compute_phase_for_position(in_p);
  const ComplexNumber i_beta = imaginary_unit * beta();
  const ComplexNumber f = component_x.value(p2d, 0);
  const ComplexNumber g = component_y.value(p2d, 0);
  const dealii::Tensor<1, 2> grad_f = component_x.gradient(p2d, 0);
  const dealii::Tensor<1, 2> grad_g = component_y.gradient(p2d, 0);
  const dealii::Tensor<1, 2> grad_h_table = component_z.gradient(p2d, 0);
  const ComplexNumber dh_dx = z_component_factor * grad_h_table[0];
  const ComplexNumber dh_dy = z_component_factor * grad_h_table[1];
  Tensor<1, 3, ComplexNumber> ret;
  ret[0] = (dh_dy - i_beta * g) * phase;
  ret[1] = (i_beta * f - dh_dx) * phase;
  ret[2] = (grad_g[0] - grad_f[1]) * phase;
  return ret;
}

Tensor<1, 3, ComplexNumber> ExactSolution::val(const Position &in_p) const {
  dealii::Tensor<1, 3, ComplexNumber> ret;
  dealii::Vector<ComplexNumber> temp;
  temp.reinit(3, false);
  vector_value(in_p, temp);
  ret[0] = temp[0];
  ret[1] = temp[1];
  ret[2] = temp[2];
  return ret;
}

ExactSolution::ExactSolution() : Function<3, ComplexNumber>(3),
  imaginary_unit(0,1),
  component_x(ranges, n_intervals, data_table_x),
  component_y(ranges, n_intervals, data_table_y),
  component_z(ranges, n_intervals, data_table_z)
     {
}

void ExactSolution::load_data(std::string fname) {
  std::ifstream input(fname);
  if(!input.is_open()) {
    throw std::runtime_error("ExactSolution::load_data: The mode profile '" + fname + "' could not be opened. Set the environment variable WAVEGUIDE_ROOT to the root folder of the repository.");
  }
  std::vector<double> x_vals, y_vals;
  std::string line;
  double lx = -1000;
  double ly = -1000;
  double norm = -1;
  std::vector<std::string> entries;
  file_beta = 0;
  file_wavelength = 0;
  file_structure = "";
  file_mode = "";
  has_permittivity = false;
  // Removes spaces, tabs and line ends at both ends.
  auto trim = [](const std::string & text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if(first == std::string::npos) return std::string();
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
  };
  // A line of the header: "# key = value". Other lines that start with '#' are comments.
  auto is_header_or_empty = [&](const std::string & in_line, bool store) {
    const std::string text = trim(in_line);
    if(text.size() > 0 && text[0] != '#') return false;
    const size_t separator = text.find('=');
    if(store && separator != std::string::npos) {
      const std::string key = trim(text.substr(1, separator - 1));
      const std::string value = trim(text.substr(separator + 1));
      if(key == "beta") file_beta = std::stod(value);
      if(key == "wavelength") file_wavelength = std::stod(value);
      if(key == "structure") file_structure = value;
      if(key == "mode") file_mode = value;
    }
    return true;
  };
  while (std::getline(input, line)) {
    if(is_header_or_empty(line, true)) continue;
    entries = split(line, " ");
    if(entries.size() > 5) has_permittivity = true;
    double x = stod(entries[0]);
    double y = stod(entries[1]);
    if(x > lx + FLOATING_PRECISION) {
      x_vals.push_back(x);
      lx = x;
    }
    if(y > ly+ FLOATING_PRECISION) {
      y_vals.push_back(y);
      ly = y;
    }
    double loc_norm = std::sqrt(std::pow(stod(entries[2]), 2) + std::pow(stod(entries[3]), 2) + std::pow(stod(entries[4]), 2));
    if(loc_norm > norm) {
      norm = loc_norm;
    }
  }
  std::pair<double, double> x_range, y_range;
  x_range.first = x_vals[0];
  x_range.second = x_vals[x_vals.size() - 1];
  y_range.first = y_vals[0];
  y_range.second = y_vals[y_vals.size() - 1];
  ExactSolution::ranges[0] = x_range;
  ExactSolution::ranges[1] = y_range;
  ExactSolution::n_intervals[0] = x_vals.size() - 1;
  ExactSolution::n_intervals[1] = y_vals.size() - 1;
  data_table_x.reinit(x_vals.size(), y_vals.size());
  data_table_y.reinit(x_vals.size(), y_vals.size());
  data_table_z.reinit(x_vals.size(), y_vals.size());
  data_table_epsilon.reinit(has_permittivity ? x_vals.size() : 0, has_permittivity ? y_vals.size() : 0);
  permittivity_range = std::pair<double, double>(1e300, -1e300);
  input.close();
  input.open(fname);
  unsigned int i = 0;
  unsigned int j = 0;
  while (std::getline(input, line)) {
    if(is_header_or_empty(line, false)) continue;
    entries = split(line, " ");
    data_table_x[i][j] = stod(entries[2]) / norm;
    data_table_y[i][j] = stod(entries[3]) / norm;
    data_table_z[i][j] = stod(entries[4]) / norm;
    if(has_permittivity) {
      data_table_epsilon[i][j] = stod(entries[5]);
      permittivity_range.first = std::min(permittivity_range.first, data_table_epsilon[i][j]);
      permittivity_range.second = std::max(permittivity_range.second, data_table_epsilon[i][j]);
    }
    i++;
    if(i == x_vals.size()) {
      i = 0;
      j++;
    }
  }
  
}

std::string ExactSolution::resolve_path(const std::string & fname) {
  if(fname.size() > 0 && fname[0] == '/') return fname;
  std::ifstream in_working_directory(fname);
  if(in_working_directory.is_open()) return fname;
  return repository_path(fname);
}

double ExactSolution::permittivity_at(double x, double y) {
  const double hx = (ranges[0].second - ranges[0].first) / n_intervals[0];
  const double hy = (ranges[1].second - ranges[1].first) / n_intervals[1];
  const long i = std::lround((x - ranges[0].first) / hx);
  const long j = std::lround((y - ranges[1].first) / hy);
  const unsigned int index_x = (unsigned int) std::min(std::max(i, 0L), (long) n_intervals[0]);
  const unsigned int index_y = (unsigned int) std::min(std::max(j, 0L), (long) n_intervals[1]);
  return data_table_epsilon[index_x][index_y];
}

J_derivative_terms ExactSolution::get_derivative_terms(const Position2D & in_p) const {
  J_derivative_terms ret;
  // The profile is interpolated bilinearly between the data points. Finite differences therefore have to use the
  // spacing of the data grid as their step width - a smaller step would only measure the kinks of the interpolant.
  const double hx = (ranges[0].second - ranges[0].first) / n_intervals[0];
  const double hy = (ranges[1].second - ranges[1].first) / n_intervals[1];
  auto shifted = [&](const double dx, const double dy) {
    Position2D p = in_p;
    p[0] += dx;
    p[1] += dy;
    return p;
  };
  const Position2D plus_x = shifted(hx, 0), minus_x = shifted(-hx, 0);
  const Position2D plus_y = shifted(0, hy), minus_y = shifted(0, -hy);
  const Position2D plus_plus = shifted(hx, hy), plus_minus = shifted(hx, -hy);
  const Position2D minus_plus = shifted(-hx, hy), minus_minus = shifted(-hx, -hy);

  ret.f = component_x.value(in_p);
  ret.d_f_dx = (component_x.value(plus_x) - component_x.value(minus_x)) / (2 * hx);
  ret.d_f_dxy = (component_x.value(plus_plus) - component_x.value(plus_minus) - component_x.value(minus_plus) + component_x.value(minus_minus)) / (4 * hx * hy);
  ret.d_f_dyy = (component_x.value(plus_y) - 2 * component_x.value(in_p) + component_x.value(minus_y)) / (hy * hy);

  const double central_value_z = component_z.value(in_p);
  ret.h = z_component_factor * central_value_z;
  ret.d_h_dx = z_component_factor * (component_z.value(plus_x) - component_z.value(minus_x)) / (2 * hx);
  ret.d_h_dy = z_component_factor * (component_z.value(plus_y) - component_z.value(minus_y)) / (2 * hy);
  ret.d_h_dxx = z_component_factor * (component_z.value(plus_x) - 2 * central_value_z + component_z.value(minus_x)) / (hx * hx);
  ret.d_h_dyy = z_component_factor * (component_z.value(plus_y) - 2 * central_value_z + component_z.value(minus_y)) / (hy * hy);
  ret.beta = beta();
  return ret;
}
