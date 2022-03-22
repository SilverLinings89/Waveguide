#include "SpaceTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <complex>
#include "../Core/Sector.h"
#include "../GlobalObjects/GlobalObjects.h"

std::pair<int, double> SpaceTransformation::Z_to_Sector_and_local_z(double in_z) const {
  std::pair<int, double> ret;
  ret.first = 0;
  ret.second = 0.0;
  if (in_z <= Geometry.global_z_range.first) {
    ret.first = 0;
    ret.second = 0.0;
  } else if (in_z < Geometry.global_z_range.second && in_z > Geometry.global_z_range.first) {
    ret.first = floor( (in_z + Geometry.global_z_range.first) / (GlobalParams.Sector_thickness));
    ret.second = (in_z + Geometry.global_z_range.first -  (ret.first * GlobalParams.Sector_thickness)) / (GlobalParams.Sector_thickness);
  } else if (in_z >= Geometry.global_z_range.second) {
    ret.first = GlobalParams.Number_of_sectors - 1;
    ret.second = 1.0;
  }

  if (ret.second < 0 || ret.second > 1){
    std::cout << "Global ranges: " << Geometry.global_z_range.first << " to " << Geometry.global_z_range.second << std::endl;
    std::cout << "Details " << GlobalParams.Sector_thickness << ", " << floor( (in_z + Geometry.global_z_range.first) / (GlobalParams.Sector_thickness)) << " and " << (in_z + Geometry.global_z_range.first) / (GlobalParams.Sector_thickness) << std::endl;
    std::cout << "In an erroneous call: ret.first: " << ret.first << " ret.second: " << ret.second << " and in_z: " << in_z << " located in sector " << ret.first << " and " << GlobalParams.Sector_thickness << std::endl; 
  }
  return ret;
}

SpaceTransformation::SpaceTransformation() { }

Tensor<2, 3, ComplexNumber> SpaceTransformation::get_Tensor_for_step(Position &coordinate, unsigned int dof, double step_width) {
  double old_value = get_dof(dof);
  Tensor<2, 3, double> trafo1 = get_Space_Transformation_Tensor(coordinate);

  set_free_dof(dof, old_value + step_width);
  Tensor<2, 3, double> trafo2 = get_Space_Transformation_Tensor(coordinate);

  set_free_dof(dof, old_value);
  return trafo2 - trafo1;
}

Position SpaceTransformation::operator()(Position in_p) const {
  return math_to_phys(in_p);
}

void SpaceTransformation::switch_application_mode(bool appl_math_to_phys) {
  apply_math_to_phys = appl_math_to_phys;
}

