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
const ComplexNumber imaginary_unit(0,1);

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
      ret_val = imaginary_unit * component_z.value(p2d,0);
      break;
    default:
      std::cout << "Error in call to ExactSolution::value. Invalid component requested." << std::endl;
  }
  return ret_val * compute_phase_for_position(in_p);
}

ComplexNumber ExactSolution::compute_phase_for_position(const Position &in_p) const {
  const double Lambda_eff = (GlobalParams.Lambda / std::sqrt(GlobalParams.Epsilon_R_effective));
  const double z =  in_p[2] - Geometry.global_z_range.first;
  const double phi = 2.0 * GlobalParams.Pi * z / Lambda_eff;
  return std::exp(phi * imaginary_unit);
}

void ExactSolution::vector_value(const Position &in_p, Vector<ComplexNumber> &values) const {
  Position p = in_p;
  ComplexNumber phase = compute_phase_for_position(p);
  Position2D p2d = get_2D_position_from_3d(in_p);
  values[0] = component_x.value(p2d,0) * phase;
  values[1] = component_y.value(p2d,0) * phase;
  values[2] = imaginary_unit * component_z.value(p2d,0) * phase;
  return;
}

Tensor<1, 3, ComplexNumber> ExactSolution::curl(const Position &in_p) const {
  const double h = 0.01;
  Tensor<1, 3, ComplexNumber> ret;
  Vector<ComplexNumber> dxF;
  Vector<ComplexNumber> dyF;
  Vector<ComplexNumber> dzF;
  Vector<ComplexNumber> val;
  dxF.reinit(6, false);
  dyF.reinit(6, false);
  dzF.reinit(6, false);
  val.reinit(6, false);
  this->vector_value(in_p, val);
  Position deltap = in_p;
  deltap[0] = deltap[0] + h;
  this->vector_value(deltap, dxF);
  deltap = in_p;
  deltap[1] = deltap[1] + h;
  this->vector_value(deltap, dyF);
  deltap = in_p;
  deltap[2] = deltap[2] + h;
  this->vector_value(deltap, dzF);
  for (int i = 0; i < 3; i++) {
    dxF[i] = (dxF[i] - val[i]) / h;
    dyF[i] = (dyF[i] - val[i]) / h;
    dzF[i] = (dzF[i] - val[i]) / h;
  }
  ret[0] = dyF[2] - dzF[1];
  ret[1] = dzF[0] - dxF[2];
  ret[2] = dxF[1] - dyF[0];
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
  std::vector<double> x_vals, y_vals;
  std::string line;
  double lx = -1000;
  double ly = -1000;
  double norm = -1;
  std::vector<std::string> entries;
  while (std::getline(input, line)) {
    entries = split(line, " ");
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
  input.close();
  input.open(fname);
  unsigned int i = 0;
  unsigned int j = 0;
  while (std::getline(input, line)) {
    entries = split(line, " ");
    data_table_x[i][j] = stod(entries[2]) / norm;
    data_table_y[i][j] = stod(entries[3]) / norm;
    data_table_z[i][j] = stod(entries[4]) / norm;
    i++;
    if(i == x_vals.size()) {
      i = 0;
      j++;
    }
  }
  
}

J_derivative_terms ExactSolution::get_derivative_terms(const Position2D & in_p) const {
  J_derivative_terms ret;
  const double h = 0.01;
  const double Lambda_eff = (GlobalParams.Lambda / std::sqrt(GlobalParams.Epsilon_R_effective));
  Position2D plus_plus = in_p;
  plus_plus[0] += h;
  plus_plus[1] += h;
  Position2D plus_minus = in_p;
  plus_plus[0] += h;
  plus_plus[1] -= h;
  Position2D minus_plus = in_p;
  plus_plus[0] -= h;
  plus_plus[1] += h;
  Position2D minus_minus = in_p;
  plus_plus[0] -= h;
  plus_plus[1] -= h;
  Position2D minus_y = in_p;
  plus_plus[1] -= h;
  Position2D plus_y = in_p;
  plus_plus[1] += h;
  Position2D minus_x = in_p;
  plus_plus[0] -= h;
  Position2D plus_x = in_p;
  plus_plus[0] += h;

  double central_value_z = component_z.value(in_p);
  ret.d_f_dxy = (component_x.value(plus_plus) - component_x.value(plus_minus) - component_x.value(minus_plus) + component_x.value(minus_minus)) / (4*h*h);
  ret.d_f_dyy = (component_x.value(plus_y) - 2*component_x.value(in_p) + component_x.value(minus_y)) / (h*h);
  ret.d_f_dx = component_x.gradient(in_p)[0];

  ret.d_h_dxx = imaginary_unit * (component_z.value(plus_x) - 2*central_value_z + component_z.value(minus_x)) / (h*h);
  ret.d_h_dyy = imaginary_unit * (component_z.value(plus_y) - 2*central_value_z + component_z.value(minus_y)) / (h*h);
  ret.d_h_dx = imaginary_unit * component_z.gradient(in_p)[0];
  ret.d_h_dy = imaginary_unit * component_z.gradient(in_p)[1];

  ret.f = component_x.value(in_p);
  ret.h = imaginary_unit * component_z.value(in_p);
  ret.beta = 2.0 * GlobalParams.Pi / Lambda_eff;
  return ret;
}
