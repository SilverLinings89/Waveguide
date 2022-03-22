#include "PredefinedShapeTransformation.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "../Core/Sector.h"
#include "../Helpers/staticfunctions.h"
#include "SpaceTransformation.h"

using namespace dealii;

PredefinedShapeTransformation::PredefinedShapeTransformation()
    : SpaceTransformation() {
  for(unsigned int i = 0; i < 3; i++) {
    for(unsigned int j = 0; j < 3; j++) {
      if(i == j) {
        I[i][j] = 1;
      } else {
        I[i][j] = 0;
      }
    }
  }
}

PredefinedShapeTransformation::~PredefinedShapeTransformation() {}

Position PredefinedShapeTransformation::math_to_phys(Position coord) const {
  Position ret;
  std::pair<int, double> sec = Z_to_Sector_and_local_z(coord[2]);
  double m = case_sectors[sec.first].get_m(sec.second);
  ret[0] = coord[0];
  ret[1] = coord[1] + m;
  ret[2] = coord[2];
  return ret;
}

Position PredefinedShapeTransformation::phys_to_math(Position coord) const {
  Position ret;
  std::pair<int, double> sec = Z_to_Sector_and_local_z(coord[2]);
  double m = case_sectors[sec.first].get_m(sec.second);
  ret[0] = coord[0];
  ret[1] = coord[1] - m;
  ret[2] = coord[2];
  return ret;
}

Tensor<2, 3, ComplexNumber>
PredefinedShapeTransformation::get_Tensor(Position &position) {
  return get_Space_Transformation_Tensor(position);
}


void PredefinedShapeTransformation::estimate_and_initialize() {
  print_info("PredefinedShapeTransformation::estimate_and_initialize", "Start");
  Sector<2> the_first(true, false, GlobalParams.sd.z[0], GlobalParams.sd.z[1]);
  the_first.set_properties_force(GlobalParams.sd.m[0], GlobalParams.sd.m[1],
                                  GlobalParams.sd.v[0], GlobalParams.sd.v[1]);
  case_sectors.push_back(the_first);
  for (int i = 1; i < GlobalParams.sd.Sectors - 2; i++) {
      Sector<2> intermediate(false, false, GlobalParams.sd.z[i], GlobalParams.sd.z[i + 1]);
      intermediate.set_properties_force(
          GlobalParams.sd.m[i], GlobalParams.sd.m[i + 1], GlobalParams.sd.v[i],
          GlobalParams.sd.v[i + 1]);
      case_sectors.push_back(intermediate);
  }
  Sector<2> the_last(false, true,
                      GlobalParams.sd.z[GlobalParams.sd.Sectors - 2],
                      GlobalParams.sd.z[GlobalParams.sd.Sectors - 1]);
  the_last.set_properties_force(
      GlobalParams.sd.m[GlobalParams.sd.Sectors - 2],
      GlobalParams.sd.m[GlobalParams.sd.Sectors - 1],
      GlobalParams.sd.v[GlobalParams.sd.Sectors - 2],
      GlobalParams.sd.v[GlobalParams.sd.Sectors - 1]);
  case_sectors.push_back(the_last);
  if(GlobalParams.MPI_Rank == 0) {
    for (unsigned int i = 0; i < case_sectors.size(); i++) {
      std::string msg_lower = "Layer at z: " + std::to_string(case_sectors[i].z_0) + "(m: " + std::to_string(case_sectors[i].get_m(0.0)) + " v: " + std::to_string(case_sectors[i].get_v(0.0)) + ")";
      print_info("PredefinedShapeTransformation::estimate_and_initialize", msg_lower);
    }
    std::string msg_last = "Layer at z: " + std::to_string(case_sectors[case_sectors.size()-1].z_1) + "(m: " + std::to_string(case_sectors[case_sectors.size()-1].get_m(1.0)) + " v: " + std::to_string(case_sectors[case_sectors.size()-1].get_v(1.0)) + ")";
    
  }
  print_info("PredefinedShapeTransformation::estimate_and_initialize", "End");
}

double PredefinedShapeTransformation::get_m(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].get_m(two.second);
}

double PredefinedShapeTransformation::get_v(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].get_v(two.second);
}

void PredefinedShapeTransformation::Print() const {
  std::cout << "Printing is not yet implemented." << std::endl;
}

Tensor<2, 3, double>
PredefinedShapeTransformation::get_Space_Transformation_Tensor(Position &position) {
  Tensor<2, 3, double> J_loc = get_J(position);
  Tensor<2, 3, double> ret;
  ret[0][0] = 1;
  ret[1][1] = 1;
  ret[2][2] = 1;
  return (J_loc * ret * transpose(J_loc)) / determinant(J_loc);
}

Tensor<2,3,double> PredefinedShapeTransformation::get_J(Position &in_p) {
  Tensor<2,3,double> ret = I;
  ret[1][2] = - get_v(in_p[2]);
  return ret;
}

Tensor<2,3,double> PredefinedShapeTransformation::get_J_inverse(Position &in_p) {
  Tensor<2,3,double> ret = get_J(in_p);
  return invert(ret);
}