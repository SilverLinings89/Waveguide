#include "ExactSolution.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>

#include "../Core/InnerDomain.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Core/Types.h"
#include "../Helpers/PointVal.h"

ComplexNumber ExactSolution::value(const Position &in_p, const unsigned int component) const {
  Position p = in_p;
  if (is_dual) p[2] = -in_p[2];

  if (is_rectangular) {
    ComplexNumber ret_val(0.0, 0.0);
    const double delta = abs(mesh_points[0] - mesh_points[1]);
    const int mesh_number = mesh_points.size();
    const double Lambda_eff = (GlobalParams.Lambda / std::sqrt(GlobalParams.Epsilon_R_effective));
    const ComplexNumber imag = {0,1};
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
            ret_val += p1p1 * vals[ix][iy].Ex;
            ret_val += p1m1 * vals[ix][iy-1].Ex;
            ret_val += m1p1 * vals[ix-1][iy].Ex;
            ret_val += m1m1 * vals[ix-1][iy-1].Ex;
            break;
          case 1:
            ret_val += p1p1 * vals[ix][iy].Ey;
            ret_val += p1m1 * vals[ix][iy-1].Ey;
      
            ret_val += m1p1 * vals[ix-1][iy].Ey;
            ret_val += m1m1 * vals[ix-1][iy-1].Ey;
            break;
          case 2:
            ret_val += p1p1 * vals[ix][iy].Ez;
            ret_val += p1m1 * vals[ix][iy-1].Ez;
            ret_val += m1p1 * vals[ix-1][iy].Ez;
            ret_val += m1m1 * vals[ix-1][iy-1].Ez;
            break;
        }
      }
      // double n = std::sqrt(GlobalParams.Epsilon_R_in_waveguide);
      // double lambda = GlobalParams.Lambda / n;
      const double z =  - p[2] + Geometry.global_z_range.first;
      const double phi = 2.0 * GlobalParams.Pi * z / Lambda_eff;
      const ComplexNumber phase = std::exp(phi * imag);
      ret_val *= phase;
      return ret_val;
    } else {
      return 0.0;
    }
  } else {
    return GlobalModeManager.get_input_component(component, p, 0);
  }
}

void ExactSolution::vector_value(const Position &in_p, Vector<ComplexNumber> &values) const {
  Position p = in_p;
  const double Lambda_eff = (GlobalParams.Lambda / std::sqrt(GlobalParams.Epsilon_R_effective));
  const ComplexNumber imag = {0,1};
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
        values[0] = p1p1 * vals[ix]   [iy].Ex +
                    p1m1 * vals[ix]   [iy-1].Ex +
                    m1m1 * vals[ix-1] [iy-1].Ex +
                    m1p1 * vals[ix-1] [iy].Ex;
        values[1] = p1p1 * vals[ix]   [iy].Ey +
                    p1m1 * vals[ix]   [iy-1].Ey +
                    m1m1 * vals[ix-1] [iy-1].Ey +
                    m1p1 * vals[ix-1] [iy].Ey;
        values[2] = p1p1 * vals[ix]   [iy].Ez +
                    p1m1 * vals[ix]   [iy-1].Ez +
                    m1m1 * vals[ix-1] [iy-1].Ez +
                    m1p1 * vals[ix-1] [iy].Ez;
        const double z =  - p[2] + Geometry.global_z_range.first;
        const double phi = 2.0 * GlobalParams.Pi * z / Lambda_eff;
        const ComplexNumber phase = std::exp(phi * imag);
        for (unsigned int komp = 0; komp < 3; komp++) {
          values[komp] *= phase;
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
    for (unsigned int c = 0; c < 3; ++c)
      values[c] = GlobalModeManager.get_input_component(c, p, 0);
  }
}

Tensor<1, 3, ComplexNumber> ExactSolution::curl(
    const Position &in_p) const {
  const double h = 0.0001;
  Tensor<1, 3, ComplexNumber> ret;
  if (is_rectangular) {
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
  }
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
    : Function<3, ComplexNumber>(3) {
  is_dual = in_dual;
  is_rectangular = in_rectangular;
  if (is_rectangular) {
    std::ifstream input("../Modes/mode_1550nm.dat");
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
    unsigned int cnt = mesh_points.size();
    vals = new PointVal *[cnt];
    for (unsigned int i = 0; i < cnt; i++) {
      vals[i] = new PointVal[cnt];
    }
    std::ifstream input2("../Modes/mode_1550nm.dat");
    std::string line2;
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
        vals[i][j].set(d1, d2, d3, d4, d5, d6);
      }
    }
    double max = 0.0;
    for (unsigned int i = 0; i < cnt; ++i) {
      for (unsigned int j = 0; j < cnt; ++j) {
        double norm = std::sqrt( std::abs(vals[j][i].Ex)*std::abs(vals[j][i].Ex) + std::abs(vals[j][i].Ey)*std::abs(vals[j][i].Ey) + std::abs(vals[j][i].Ez)*std::abs(vals[j][i].Ez));
        if(norm > max) max = norm;
      }
    }
    for (unsigned int i = 0; i < cnt; ++i) {
      for (unsigned int j = 0; j < cnt; ++j) {
        vals[j][i].rescale(1.0 / max);
      }
    }
  }
}
