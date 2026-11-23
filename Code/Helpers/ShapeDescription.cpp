#include "ShapeDescription.h"
#include <stdio.h>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "../GlobalObjects/GlobalObjects.h"

ShapeDescription::ShapeDescription() {}

ShapeDescription::~ShapeDescription() {}

void ShapeDescription::SetStraight() {
  z.push_back(GlobalParams.global_z_shift);
  z.push_back(GlobalParams.Geometry_Size_Z + GlobalParams.global_z_shift);
  m.push_back(0);
  m.push_back(0);
}

void ShapeDescription::SetByString(std::string str) {
  std::istringstream iss(str);
  std::string token;
  std::getline(iss, token, ',');
  Sectors = std::stoi(token);
  double z_0 = 0.0;
  int i = 0;
  for (i = 0; i < Sectors; i++) {
    std::getline(iss, token, ',');
    if(i == 0) {
      z_0 = std::stod(token);
      z.push_back(0);
    } else {
      z.push_back(std::stod(token) - z_0);
    }
  }
  for (i = 0; i < Sectors; i++) {
    std::getline(iss, token, ',');
    m.push_back(std::stod(token));
  }
  for (i = 0; i < Sectors; i++) {
    std::getline(iss, token, ',');
    v.push_back(std::stod(token));
  }

  // Bring the nodes into ascending order of z and drop nodes at the position of their predecessor.
  std::vector<unsigned int> order(Sectors);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](unsigned int a, unsigned int b) { return z[a] < z[b]; });
  std::vector<double> sorted_z, sorted_m, sorted_v;
  for (unsigned int index : order) {
    if (!sorted_z.empty() && z[index] <= sorted_z.back()) continue;
    sorted_z.push_back(z[index]);
    sorted_m.push_back(m[index]);
    sorted_v.push_back(v[index]);
  }
  z = sorted_z;
  m = sorted_m;
  v = sorted_v;
  Sectors = z.size();
}

unsigned int ShapeDescription::interval(double in_z, double & t) const {
  // First node behind in_z; the interval ends there.
  unsigned int upper = std::upper_bound(z.begin(), z.end(), in_z) - z.begin();
  upper = std::min<unsigned int>(std::max<unsigned int>(upper, 1), z.size() - 1);
  t = (in_z - z[upper - 1]) / (z[upper] - z[upper - 1]);
  return upper - 1;
}

double ShapeDescription::displacement(double in_z) const {
  if (m.empty()) return 0.0;
  if (m.size() == 1 || in_z <= z.front()) return m.front();
  if (in_z >= z.back()) return m.back();
  double t = 0;
  const unsigned int i = interval(in_z, t);
  const double h = z[i + 1] - z[i];
  // Cubic Hermite polynomial. The slopes are derivatives with respect to z, the local coordinate t is z / h.
  const double slope_0 = (v.size() == m.size()) ? v[i] * h : 0.0;
  const double slope_1 = (v.size() == m.size()) ? v[i + 1] * h : 0.0;
  return (2 * (m[i] - m[i + 1]) + slope_0 + slope_1) * t * t * t + (3 * (m[i + 1] - m[i]) - 2 * slope_0 - slope_1) * t * t + slope_0 * t + m[i];
}

double ShapeDescription::slope(double in_z) const {
  if (m.size() < 2 || in_z < z.front() || in_z > z.back()) return 0.0;
  const bool has_slopes = v.size() == m.size();
  if (in_z == z.front()) return has_slopes ? v.front() : 0.0;
  if (in_z == z.back()) return has_slopes ? v.back() : 0.0;
  double t = 0;
  const unsigned int i = interval(in_z, t);
  const double h = z[i + 1] - z[i];
  const double slope_0 = has_slopes ? v[i] * h : 0.0;
  const double slope_1 = has_slopes ? v[i + 1] * h : 0.0;
  return (3 * (2 * (m[i] - m[i + 1]) + slope_0 + slope_1) * t * t + 2 * (3 * (m[i + 1] - m[i]) - 2 * slope_0 - slope_1) * t + slope_0) / h;
}
