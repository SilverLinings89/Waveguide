#ifndef SpaceTransformation_CPP
#define SpaceTransformation_CPP

#include "SpaceTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <complex>
#include "../Core/Sector.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"

std::pair<int, double> SpaceTransformation::Z_to_Sector_and_local_z(
    double in_z) const {
  std::pair<int, double> ret;
  ret.first = 0;
  ret.second = 0.0;
  if (in_z <= -GlobalParams.M_R_ZLength / 2.0) {
    ret.first = 0;
    ret.second = 0.0;
  } else if (abs(in_z) < GlobalParams.M_R_ZLength / 2.0) {
    ret.first = floor((in_z + GlobalParams.M_R_ZLength / 2.0) /
                      (GlobalParams.SectorThickness));
    ret.second = (in_z + GlobalParams.M_R_ZLength / 2.0 -
                  (ret.first * GlobalParams.SectorThickness)) /
                 (GlobalParams.SectorThickness);
  } else if (in_z >= GlobalParams.M_R_ZLength / 2.0) {
    ret.first = sectors - 1;
    ret.second = 1.0;
  }
  return ret;
}

SpaceTransformation::SpaceTransformation(int in_dofs_per_layer, int in_rank)
    : dofs_per_layer(in_dofs_per_layer),
      boundary_dofs_in(in_dofs_per_layer),
      boundary_dofs_out(in_dofs_per_layer),
      epsilon_K(GlobalParams.M_W_epsilonin),
      epsilon_M(GlobalParams.M_W_epsilonout),
      sectors(GlobalParams.M_W_Sectors),
      deltaY(GlobalParams.M_W_Delta),
      rank(in_rank) {
  InitialQuality = 0;
}

double SpaceTransformation::Sector_Length() const {
  return GlobalParams.SectorThickness;
}

int SpaceTransformation::Z_to_Layer(double in_z) const {
  double temp = (in_z - GlobalParams.Minimum_Z) / GlobalParams.LayerThickness;
  int flr = floor(temp);
  if (flr == 0) {
    return 0;
  } else {
    if (temp - flr < 0.000001) {
      return flr - 1;
    } else {
      return flr;
    }
  }
}

bool SpaceTransformation::is_identity(Point<3, double> coord) const {
  double sum = 0.0;
  Point<3, double> temp = math_to_phys(coord);
  for (unsigned int i = 0; i < 3; i++) {
    sum += std::abs(temp[i] - coord[i]);
  }
  return sum < 0.0001;
}

std::pair<double, double> SpaceTransformation::dof_support(
    unsigned int index) const {
  std::pair<double, double> ret;
  ret.first = 0.0;
  ret.second = 0.0;
  int boundary = index / dofs_per_layer;
  ret.first =
      -GlobalParams.M_R_ZLength / 2.0 + (boundary - 1) * Sector_Length();
  ret.second = ret.first + 2 * Sector_Length();
  return ret;
}

bool SpaceTransformation::point_in_dof_support(Point<3> location,
                                               unsigned int dof_index) const {
  std::pair<double, double> temp = dof_support(dof_index);
  if (std::abs(location[2]) > GlobalParams.M_R_ZLength / 2.0) {
    return false;
  } else {
    return (temp.first <= location[2] && temp.second >= location[2]);
  }
}

Tensor<2, 3, std::complex<double>> SpaceTransformation::get_Tensor_for_step(
    Point<3> &coordinate, unsigned int dof, double step_width) {
  double old_value = get_dof(dof);
  Tensor<2, 3, double> trafo = get_Space_Transformation_Tensor(coordinate);

  Tensor<2, 3, std::complex<double>> original =
      Apply_PML_To_Tensor(coordinate, trafo);
  set_dof(dof, old_value + step_width);
  trafo = get_Space_Transformation_Tensor(coordinate);
  Tensor<2, 3, std::complex<double>> ret =
      Apply_PML_To_Tensor(coordinate, trafo);
  set_dof(dof, old_value);
  return ret - original;
}


#endif
