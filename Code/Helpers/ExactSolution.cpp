
#ifndef ExactSolutionFlag_CPP
#define ExactSolutionFlag_CPP

#include "ExactSolution.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <complex>
#include <string>
#include <vector>

#include "../Core/NumericProblem.h"
#include "../Core/GlobalObjects.h"
#include "PointVal.h"

double ExactSolution::value(const Point<3> &in_p,
                            const unsigned int component) const {
  Point<3, double> p = in_p;
  if (is_dual) p[2] = -in_p[2];

  if (is_rectangular) {
    std::complex<double> ret_val(0.0, 0.0);
    const double delta = abs(mesh_points[0] - mesh_points[1]);
    const int mesh_number = mesh_points.size();
    if (!(abs(p[1]) >= mesh_points[0] || abs(p[0]) >= mesh_points[0])) {
      int ix = 0;
      int iy = 0;
      while (mesh_points[ix] > p[0] && ix < mesh_number) ix++;
      while (mesh_points[iy] > p[1] && iy < mesh_number) iy++;
      if (ix == 0 || iy == 0 || ix == mesh_number || iy == mesh_number) {
        return 0.0;
      } else {
        double dx = (p[0] - mesh_points[ix]) / delta;
        double dy = (p[1] - mesh_points[iy]) / delta;
        double m1m1 = dx * dy;
        double m1p1 = dx * (1.0 - dy);
        double p1p1 = (1.0 - dx) * (1.0 - dy);
        double p1m1 = (1.0 - dx) * dy;
        switch (component % 3) {
          case 0:
            ret_val.real(p1p1 * vals[ix][iy].Ex.real() +
                         p1m1 * vals[ix][iy - 1].Ex.real() +
                         m1m1 * vals[ix - 1][iy - 1].Ex.real() +
                         m1p1 * vals[ix - 1][iy].Ex.real());
            ret_val.imag(p1p1 * vals[ix][iy].Ex.imag() +
                         p1m1 * vals[ix][iy - 1].Ex.imag() +
                         m1m1 * vals[ix - 1][iy - 1].Ex.imag() +
                         m1p1 * vals[ix - 1][iy].Ex.imag());
            // ret_val *= -1.0;
            break;
          case 1:
            ret_val.real(p1p1 * vals[ix][iy].Ey.real() +
                         p1m1 * vals[ix][iy - 1].Ey.real() +
                         m1m1 * vals[ix - 1][iy - 1].Ey.real() +
                         m1p1 * vals[ix - 1][iy].Ey.real());
            ret_val.imag(p1p1 * vals[ix][iy].Ey.imag() +
                         p1m1 * vals[ix][iy - 1].Ey.imag() +
                         m1m1 * vals[ix - 1][iy - 1].Ey.imag() +
                         m1p1 * vals[ix - 1][iy].Ey.imag());
            break;
          case 2:
            ret_val.real(p1p1 * vals[ix][iy].Ez.real() +
                         p1m1 * vals[ix][iy - 1].Ez.real() +
                         m1m1 * vals[ix - 1][iy - 1].Ez.real() +
                         m1p1 * vals[ix - 1][iy].Ez.real());
            ret_val.imag(p1p1 * vals[ix][iy].Ez.imag() +
                         p1m1 * vals[ix][iy - 1].Ez.imag() +
                         m1m1 * vals[ix - 1][iy - 1].Ez.imag() +
                         m1p1 * vals[ix - 1][iy].Ez.imag());
            break;
          default:
            ret_val.real(0.0);
            ret_val.imag(0.0);
            break;
        }
      }
      double n;
      if (abs(p[0]) <= GlobalParams.Width_of_waveguide &&
          abs(p[1]) <= GlobalParams.Height_of_waveguide) {
        n = std::sqrt(GlobalParams.Epsilon_R_in_waveguide);
      } else {
        n = std::sqrt(GlobalParams.Epsilon_R_outside_waveguide);
      }
      double k = n * 2 * GlobalParams.Pi / GlobalParams.M_W_Lambda;
      std::complex<double> phase(0.0,
          (p[2] - Geometry.global_z_range.first) * k);
      ret_val *= std::exp(phase);
      if (component > 2) {
        return ret_val.imag();
      } else {
        return ret_val.real();
      }
    } else {
      return 0.0;
    }
  } else {
    return GlobalModeManager.get_input_component(component, p, 0);
  }
}

void ExactSolution::vector_value(const Point<3> &in_p,
                                 Vector<double> &values) const {
  Point<3, double> p = in_p;
  if (is_dual) p[2] = -in_p[2];

  if (is_rectangular) {
    const double delta = abs(mesh_points[0] - mesh_points[1]);
    const int mesh_number = mesh_points.size();
    if (!(abs(p[1]) >= mesh_points[0] || abs(p[0]) >= mesh_points[0])) {
      int ix = 0;
      int iy = 0;
      while (mesh_points[ix] > p[0] && ix < mesh_number) ix++;
      while (mesh_points[iy] > p[1] && iy < mesh_number) iy++;
      if (ix == 0 || iy == 0 || ix == mesh_number || iy == mesh_number) {
        for (unsigned int i = 0; i < values.size(); i++) {
          values[i] = 0.0;
        }
        return;
      } else {
        double dx = (p[0] - mesh_points[ix]) / delta;
        double dy = (p[1] - mesh_points[iy]) / delta;
        double m1m1 = dx * dy;
        double m1p1 = dx * (1.0 - dy);
        double p1p1 = (1.0 - dx) * (1.0 - dy);
        double p1m1 = (1.0 - dx) * dy;
        values[0] = p1p1 * vals[ix][iy].Ex.real() +
                    p1m1 * vals[ix][iy - 1].Ex.real() +
                    m1m1 * vals[ix - 1][iy - 1].Ex.real() +
                    m1p1 * vals[ix - 1][iy].Ex.real();
        values[1] = p1p1 * vals[ix][iy].Ey.real() +
                    p1m1 * vals[ix][iy - 1].Ey.real() +
                    m1m1 * vals[ix - 1][iy - 1].Ey.real() +
                    m1p1 * vals[ix - 1][iy].Ey.real();
        values[2] = p1p1 * vals[ix][iy].Ez.real() +
                    p1m1 * vals[ix][iy - 1].Ez.real() +
                    m1m1 * vals[ix - 1][iy - 1].Ez.real() +
                    m1p1 * vals[ix - 1][iy].Ez.real();
        values[3] = p1p1 * vals[ix][iy].Ex.imag() +
                    p1m1 * vals[ix][iy - 1].Ex.imag() +
                    m1m1 * vals[ix - 1][iy - 1].Ex.imag() +
                    m1p1 * vals[ix - 1][iy].Ex.imag();
        values[4] = p1p1 * vals[ix][iy].Ey.imag() +
                    p1m1 * vals[ix][iy - 1].Ey.imag() +
                    m1m1 * vals[ix - 1][iy - 1].Ey.imag() +
                    m1p1 * vals[ix - 1][iy].Ey.imag();
        values[5] = p1p1 * vals[ix][iy].Ez.imag() +
                    p1m1 * vals[ix][iy - 1].Ez.imag() +
                    m1m1 * vals[ix - 1][iy - 1].Ez.imag() +
                    m1p1 * vals[ix - 1][iy].Ez.imag();
        double n;
        if (abs(p[0]) <= GlobalParams.Width_of_waveguide &&
            abs(p[1]) <= GlobalParams.Height_of_waveguide) {
          n = std::sqrt(GlobalParams.Epsilon_R_in_waveguide);
        } else {
          n = std::sqrt(GlobalParams.Epsilon_R_outside_waveguide);
        }
        double k = n * 2 * GlobalParams.Pi / GlobalParams.M_W_Lambda;
        std::complex<double> phase(
            0.0, -(p[2] + GlobalParams.Geometry_Size_Z / 2.0) * k);
        phase = std::exp(phase);
        for (unsigned int komp = 0; komp < 3; komp++) {
          std::complex<double> entr(values[0 + komp], values[3 + komp]);
          entr *= phase;
          values[0 + komp] = entr.real();
          values[3 + komp] = entr.imag();
        }
        return;
      }
    } else {
      for (unsigned int i = 0; i < values.size(); i++) {
        values[i] = 0.0;
      }
      return;
    }
  } else {
    for (unsigned int c = 0; c < 6; ++c)
      values[c] = GlobalModeManager.get_input_component(c, p, 0);
  }
}

Tensor<1, 3, std::complex<double>> ExactSolution::curl(
    const Point<3> &in_p) const {
  const double h = 0.0001;
  Tensor<1, 3, std::complex<double>> ret;
  if (is_rectangular) {
    Vector<double> dxF;
    Vector<double> dyF;
    Vector<double> dzF;
    Vector<double> val;
    dxF.reinit(6, false);
    dyF.reinit(6, false);
    dzF.reinit(6, false);
    val.reinit(6, false);
    this->vector_value(in_p, val);
    Point<3> deltap = in_p;
    deltap[0] = deltap[0] + h;
    this->vector_value(deltap, dxF);
    deltap = in_p;
    deltap[1] = deltap[1] + h;
    this->vector_value(deltap, dyF);
    deltap = in_p;
    deltap[1] = deltap[1] + h;
    this->vector_value(deltap, dzF);
    for (int i = 0; i < 6; i++) {
      dxF[i] = (dxF[i] - val[i]) / h;
      dyF[i] = (dyF[i] - val[i]) / h;
      dzF[i] = (dzF[i] - val[i]) / h;
    }
    ret[0].real(dyF[2] - dzF[1]);
    ret[0].imag(dyF[5] - dzF[4]);
    ret[1].real(dzF[0] - dxF[2]);
    ret[1].imag(dzF[3] - dxF[5]);
    ret[2].real(dxF[1] - dyF[0]);
    ret[2].imag(dxF[4] - dyF[3]);
  }
  return ret;
}

Tensor<1, 3, std::complex<double>> ExactSolution::val(
    const Point<3> &in_p) const {
  Tensor<1, 3, std::complex<double>> ret;
  Vector<double> temp;
  temp.reinit(6, false);
  vector_value(in_p, temp);
  ret[0].real(temp[0]);
  ret[0].imag(temp[3]);
  ret[1].real(temp[1]);
  ret[1].imag(temp[4]);
  ret[2].real(temp[2]);
  ret[2].imag(temp[5]);
  return ret;
}

std::vector<std::string> ExactSolution::split(std::string str) const {
  std::vector<std::string> ret;
  std::istringstream iss(str);
  std::string token;
  while (std::getline(iss, token, '\t')) ret.push_back(token);
  return ret;
}

double scientific_string_to_double(std::string inp) {
  std::istringstream os(inp);
  double d = 0.0;
  os >> d;
  return d;
}

ExactSolution::ExactSolution(bool in_rectangular, bool in_dual)
    : Function<3>(6) {
  is_dual = in_dual;
  is_rectangular = in_rectangular;
  if (is_rectangular) {
    deallog << "Preparing exact solution for rectangular waveguide."
            << std::endl;
  } else {
    deallog << "Preparing exact solution for circular waveguide." << std::endl;
  }
  if (is_rectangular) {
    std::ifstream input("Modes/mode_1550nm.dat");
    std::string line;
    double l_val = 3.0;
    int cnt_a = 0;
    while (std::getline(input, line)) {
      std::vector<std::string> ls = split(line);
      std::istringstream iss(ls[2]);
      double x = 0.0;
      iss >> x;
      if (x < l_val) {
        mesh_points.push_back(x);
        l_val = x;
      }
      cnt_a++;
    }
    deallog << cnt_a << " - " << mesh_points.size() << std::endl;
    unsigned int cnt = mesh_points.size();
    vals = new PointVal *[cnt];
    for (unsigned int i = 0; i < cnt; i++) {
      vals[i] = new PointVal[cnt];
    }
    deallog << cnt << std::endl;
    std::ifstream input2("Modes/mode_1550nm.dat");
    std::string line2;
    double max = 0.0;
    for (unsigned int i = 0; i < cnt; ++i) {
      for (unsigned int j = 0; j < cnt; ++j) {
        getline(input2, line2);
        std::vector<std::string> ls = split(line2);
        double d1 = scientific_string_to_double(ls[4]);
        double d2 = scientific_string_to_double(ls[5]);
        double d3 = scientific_string_to_double(ls[3]);
        double d4 = scientific_string_to_double(ls[7]);
        double d5 = scientific_string_to_double(ls[8]);
        double d6 = scientific_string_to_double(ls[6]);
        if (d1 > max) max = d1;
        if (d2 > max) max = d2;
        if (d3 > max) max = d3;
        if (d4 > max) max = d4;
        if (d5 > max) max = d5;
        if (d6 > max) max = d6;
        vals[i][j].set(d1, d2, d3, d4, d5, d6);
      }
    }
    for (unsigned int i = 0; i < cnt; ++i) {
      for (unsigned int j = 0; j < cnt; ++j) {
        vals[j][i].rescale(1.0 / max);
      }
    }

    deallog << " Mesh constant: " << abs(mesh_points[0] - mesh_points[1])
            << std::endl;
  }
  deallog << "Done Preparing exact solution." << std::endl;
}

#endif
