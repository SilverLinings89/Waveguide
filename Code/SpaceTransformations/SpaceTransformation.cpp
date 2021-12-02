#include "SpaceTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <complex>
#include "../Core/Sector.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "../GlobalObjects/GlobalObjects.h"

std::pair<int, double> SpaceTransformation::Z_to_Sector_and_local_z(
    double in_z) const {
  std::pair<int, double> ret;
  ret.first = 0;
  ret.second = 0.0;
  if (in_z <= Geometry.global_z_range.first) {
    ret.first = 0;
    ret.second = 0.0;
  } else if (in_z < Geometry.global_z_range.second
      && in_z > Geometry.global_z_range.first) {
    ret.first = floor(
        (in_z + Geometry.global_z_range.first)
            /
                      (GlobalParams.Sector_thickness));
    ret.second = (in_z + Geometry.global_z_range.first
        -
                  (ret.first * GlobalParams.Sector_thickness)) /
                 (GlobalParams.Sector_thickness);
  } else if (in_z >= Geometry.global_z_range.second) {
    ret.first = sectors - 1;
    ret.second = 1.0;
  }
  return ret;
}

SpaceTransformation::SpaceTransformation(int in_dofs_per_layer)
    : dofs_per_layer(in_dofs_per_layer),
      boundary_dofs_in(in_dofs_per_layer),
      boundary_dofs_out(in_dofs_per_layer),
      epsilon_K(GlobalParams.Epsilon_R_in_waveguide),
      epsilon_M(GlobalParams.Epsilon_R_outside_waveguide),
      sectors(GlobalParams.Number_of_sectors),
      deltaY(GlobalParams.Vertical_displacement_of_waveguide) {
  InitialQuality = 0;
}

double SpaceTransformation::Sector_Length() const {
  return GlobalParams.Sector_thickness;
}

bool SpaceTransformation::is_identity(Position coord) const {
  double sum = 0.0;
  Position temp = math_to_phys(coord);
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
      -GlobalParams.Geometry_Size_Z / 2.0 + (boundary - 1) * Sector_Length();
  ret.second = ret.first + 2 * Sector_Length();
  return ret;
}

bool SpaceTransformation::point_in_dof_support(Position location,
                                               unsigned int dof_index) const {
  std::pair<double, double> temp = dof_support(dof_index);
  if (std::abs(location[2]) > GlobalParams.Geometry_Size_Z / 2.0) {
    return false;
  } else {
    return (temp.first <= location[2] && temp.second >= location[2]);
  }
}

Tensor<2, 3, ComplexNumber> SpaceTransformation::get_Tensor_for_step(
    Position &coordinate, unsigned int dof, double step_width) {
  double old_value = get_dof(dof);
  Tensor<2, 3, double> trafo1 = get_Space_Transformation_Tensor(coordinate);

  set_dof(dof, old_value + step_width);
  Tensor<2, 3, double> trafo2 = get_Space_Transformation_Tensor(coordinate);

  set_dof(dof, old_value);
  return trafo2 - trafo1;
}
