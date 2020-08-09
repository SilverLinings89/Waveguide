#ifndef HOMOGENOUS_TRANSFORMATION_RECTANGULAR_CPP
#define HOMOGENOUS_TRANSFORMATION_RECTANGULAR_CPP

#include "HomogenousTransformationRectangular.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <complex>
#include "../Core/Sector.h"
#include "../Core/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"

using namespace dealii;

HomogenousTransformationRectangular::HomogenousTransformationRectangular(
    int in_rank)
    : SpaceTransformation(2, in_rank),
      epsilon_K(GlobalParams.Epsilon_R_in_waveguide),
      epsilon_M(GlobalParams.Epsilon_R_outside_waveguide),
      sectors(GlobalParams.M_W_Sectors),
      deltaY(GlobalParams.M_W_Delta) {
  homogenized = true;
}

HomogenousTransformationRectangular::~HomogenousTransformationRectangular() {}

Point<3, double> HomogenousTransformationRectangular::math_to_phys(
    Point<3, double> coord) const {
  Point<3, double> ret;
  if (coord[2] < GlobalParams.Geometry_Size_Z / (-2.0)) {
    ret[0] = (2 * GlobalParams.Width_of_waveguide) * coord[0] /
             (GlobalParams.Width_of_waveguide + GlobalParams.Width_of_waveguide);
    ret[1] = (2 * GlobalParams.Height_of_waveguide) * coord[1] /
             (GlobalParams.Height_of_waveguide + GlobalParams.Height_of_waveguide);
    ret[2] = coord[2];
  } else if (coord[2] >= GlobalParams.Geometry_Size_Z / (-2.0) &&
             coord[2] < GlobalParams.Geometry_Size_Z / (2.0)) {
    std::pair<int, double> sec = Z_to_Sector_and_local_z(coord[2]);
    double m = case_sectors[sec.first].get_m(sec.second);
    ret[0] = coord[0];
    ret[1] = coord[1] + m;
    ret[2] = coord[2];
  } else {
    ret[0] = (2 * GlobalParams.Width_of_waveguide) * coord[0] /
             (GlobalParams.Width_of_waveguide + GlobalParams.Width_of_waveguide);
    ret[1] = (2 * GlobalParams.Height_of_waveguide) * coord[1] /
             (GlobalParams.Height_of_waveguide + GlobalParams.Height_of_waveguide);
    ret[2] = coord[2];
  }
  return ret;
}

Tensor<2, 3, double> HomogenousTransformationRectangular::get_Space_Transformation_Tensor_Homogenized(
    Point<3, double> &position) const {
  std::pair<int, double> sector_z = Z_to_Sector_and_local_z(position[2]);

  Tensor<2, 3, double> transformation =
      case_sectors[sector_z.first].TransformationTensorInternal(position[0],
          position[1], sector_z.second);

  double dist = position[0] * position[0] + position[1] * position[1];

  dist = sqrt(dist);
  double v1 = Geometry.global_x_range.second;
  double v2 = Geometry.global_y_range.second;
  double maxdist = std::min(v1, v2);
  double mindist = (GlobalParams.Width_of_waveguide + GlobalParams.Width_of_waveguide) / 2.0;
  double sig = sigma(dist, mindist, maxdist);
  double factor = InterpolationPolynomialZeroDerivative(sig, 1, 0);
  transformation *= factor;
  for (int i = 0; i < 3; i++) {
    transformation[i][i] += 1 - factor;
  }

  return transformation;
}

Point<3, double> HomogenousTransformationRectangular::phys_to_math(
    Point<3, double> coord) const {
  Point<3, double> ret;
  if (coord[2] < GlobalParams.Geometry_Size_Z / (-2.0)) {
    ret[0] = (GlobalParams.Width_of_waveguide + GlobalParams.Width_of_waveguide) * coord[0] /
             (2 * GlobalParams.Width_of_waveguide);
    ret[1] = (GlobalParams.Height_of_waveguide + GlobalParams.Height_of_waveguide) * coord[1] /
             (2 * GlobalParams.Height_of_waveguide);
    ret[2] = coord[2];
  } else if (coord[2] >= GlobalParams.Geometry_Size_Z / (-2.0) &&
             coord[2] < GlobalParams.Geometry_Size_Z / (2.0)) {
    std::pair<int, double> sec = Z_to_Sector_and_local_z(coord[2]);
    double m = case_sectors[sec.first].get_m(sec.second);
    ret[0] = coord[0];
    ret[1] = coord[1] - m;
    ret[2] = coord[2];
  } else {
    ret[0] = (GlobalParams.Width_of_waveguide + GlobalParams.Width_of_waveguide) * coord[0] /
             (2 * GlobalParams.Width_of_waveguide);
    ret[1] = (GlobalParams.Height_of_waveguide + GlobalParams.Height_of_waveguide) * coord[1] /
             (2 * GlobalParams.Height_of_waveguide);
    ret[2] = coord[2];
  }
  return ret;
}

Tensor<2, 3, std::complex<double>>
HomogenousTransformationRectangular::get_Tensor(
    Point<3, double> &position) const {
  return get_Space_Transformation_Tensor_Homogenized(position);
}

Tensor<2, 3, double>
HomogenousTransformationRectangular::get_Space_Transformation_Tensor(
    Point<3, double> &position) const {
  std::pair<int, double> sector_z = Z_to_Sector_and_local_z(position[2]);

  Tensor<2, 3, double> transformation =
      case_sectors[sector_z.first].TransformationTensorInternal(
          position[0], position[1], sector_z.second);

  return transformation;
}

double HomogenousTransformationRectangular::get_dof(int dof) const {
  if (dof < (int)NDofs() && dof >= 0) {
    int sector = floor(dof / 2);
    if (sector == sectors) {
      return case_sectors[sector - 1].dofs_r[dof % 2];
    } else {
      return case_sectors[sector].dofs_l[dof % 2];
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in "
                 "HomogenousTransformationRectangular::get_dof!"
              << std::endl;
    return 0.0;
  }
}

double HomogenousTransformationRectangular::get_free_dof(int in_dof) const {
  int dof = in_dof + 2;
  if (dof < (int)NDofs() - 2 && dof >= 0) {
    int sector = floor(dof / 2);
    if (sector == sectors) {
      return case_sectors[sector - 1].dofs_r[dof % 2];
    } else {
      return case_sectors[sector].dofs_l[dof % 2];
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in "
                 "HomogenousTransformationRectangular::get_free_dof!"
              << std::endl;
    return 0.0;
  }
}

void HomogenousTransformationRectangular::set_dof(int dof, double in_val) {
  if (dof < (int)NDofs() && dof >= 0) {
    int sector = floor(dof / 2);
    if (sector == sectors) {
      case_sectors[sector - 1].dofs_r[dof % 2] = in_val;
    } else if (sector == 0) {
      case_sectors[0].dofs_l[dof % 2] = in_val;
    } else {
      case_sectors[sector].dofs_l[dof % 2] = in_val;
      case_sectors[sector - 1].dofs_r[dof % 2] = in_val;
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in "
                 "HomogenousTransformationRectangular::set_dof!"
              << std::endl;
  }
}

void HomogenousTransformationRectangular::set_free_dof(int in_dof,
                                                       double in_val) {
  int dof = in_dof + 2;
  if (dof < (int)NDofs() - 2 && dof >= 0) {
    int sector = floor(dof / 2);
    if (sector == sectors) {
      case_sectors[sector - 1].dofs_r[dof % 2] = in_val;
    } else if (sector == 0) {
      case_sectors[0].dofs_l[dof % 2] = in_val;
    } else {
      case_sectors[sector].dofs_l[dof % 2] = in_val;
      case_sectors[sector - 1].dofs_r[dof % 2] = in_val;
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in "
                 "HomogenousTransformationRectangular::set_free_dof!"
              << std::endl;
  }
}

double HomogenousTransformationRectangular::Sector_Length() const {
  return GlobalParams.SectorThickness;
}

void HomogenousTransformationRectangular::estimate_and_initialize() {
  if (GlobalParams.M_PC_Use) {
    Sector<2> the_first(true, false, GlobalParams.sd.z[0],
                        GlobalParams.sd.z[1]);
    the_first.set_properties_force(GlobalParams.sd.m[0], GlobalParams.sd.m[1],
                                   GlobalParams.sd.v[0], GlobalParams.sd.v[1]);
    case_sectors.push_back(the_first);
    for (int i = 1; i < GlobalParams.sd.Sectors - 2; i++) {
      Sector<2> intermediate(false, false, GlobalParams.sd.z[i],
                             GlobalParams.sd.z[i + 1]);
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
    for (unsigned int i = 0; i < case_sectors.size(); i++) {
      deallog << "From z: " << case_sectors[i].z_0
              << "(m: " << case_sectors[i].get_m(0.0)
              << " v: " << case_sectors[i].get_v(0.0) << ")" << std::endl;
      deallog << "  To z: " << case_sectors[i].z_1
              << "(m: " << case_sectors[i].get_m(1.0)
              << " v: " << case_sectors[i].get_v(1.0) << ")" << std::endl;
    }
  } else {
    case_sectors.reserve(sectors);
    double m_0 = GlobalParams.M_W_Delta / 2.0;
    double m_1 = -GlobalParams.M_W_Delta / 2.0;
    if (sectors == 1) {
      Sector<2> temp12(true, true, -GlobalParams.Geometry_Size_Z / 2.0,
                       GlobalParams.Geometry_Size_Z / 2.0);
      case_sectors.push_back(temp12);
      case_sectors[0].set_properties_force(
          GlobalParams.M_W_Delta / 2.0, -GlobalParams.M_W_Delta / 2.0,
          GlobalParams.Width_of_waveguide, GlobalParams.Width_of_waveguide, 0, 0);
    } else {
      double length = Sector_Length();
      Sector<2> temp(true, false, -GlobalParams.Geometry_Size_Z / (2.0),
                     -GlobalParams.Geometry_Size_Z / 2.0 + length);
      case_sectors.push_back(temp);
      for (int i = 1; i < sectors; i++) {
        Sector<2> temp2(false, false,
                        -GlobalParams.Geometry_Size_Z / (2.0) + length * (1.0 * i),
                        -GlobalParams.Geometry_Size_Z / (2.0) + length * (i + 1.0));
        case_sectors.push_back(temp2);
      }

      double length_rel = 1.0 / ((double)(sectors));
      case_sectors[0].set_properties_force(
          m_0, InterpolationPolynomialZeroDerivative(length_rel, m_0, m_1), 0,
          InterpolationPolynomialDerivative(length_rel, m_0, m_1, 0, 0));
      for (int i = 1; i < sectors; i++) {
        double z_l = i * length_rel;
        double z_r = (i + 1) * length_rel;
        case_sectors[i].set_properties_force(
            InterpolationPolynomialZeroDerivative(z_l, m_0, m_1),
            InterpolationPolynomialZeroDerivative(z_r, m_0, m_1),
            InterpolationPolynomialDerivative(z_l, m_0, m_1, 0, 0),
            InterpolationPolynomialDerivative(z_r, m_0, m_1, 0, 0));
      }
    }
  }
}

double HomogenousTransformationRectangular::get_r(double) const {
  // std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  std::cout << "Asking for Radius of rectangular Waveguide." << std::endl;
  return 0;
}

double HomogenousTransformationRectangular::get_m(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].get_m(two.second);
}

double HomogenousTransformationRectangular::get_v(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].get_v(two.second);
}

double HomogenousTransformationRectangular::get_Q1(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].getQ1(two.second);
}

double HomogenousTransformationRectangular::get_Q2(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].getQ2(two.second);
}

double HomogenousTransformationRectangular::get_Q3(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].getQ3(two.second);
}

Vector<double> HomogenousTransformationRectangular::Dofs() const {
  Vector<double> ret;
  const int total = NDofs();
  ret.reinit(total);
  for (int i = 0; i < total; i++) {
    ret[i] = get_dof(i);
  }
  return ret;
}

unsigned int HomogenousTransformationRectangular::NFreeDofs() const {
  return NDofs() - 4;
}

bool HomogenousTransformationRectangular::IsDofFree(int index) const {
  return index > 1 && index < (int)NDofs() - 1;
}

void HomogenousTransformationRectangular::Print() const {
  std::cout << "Printing is not yet implemented." << std::endl;
}

unsigned int HomogenousTransformationRectangular::NDofs() const {
  return sectors * 2 + 2;
}

#endif
