#include "InhomogenousTransformationRectangular.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "../Core/Sector.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "../Helpers/staticfunctions.h"
#include "SpaceTransformation.h"

using namespace dealii;

InhomogenousTransformationRectangular::InhomogenousTransformationRectangular()
    : SpaceTransformation(3),
      sectors(GlobalParams.Number_of_sectors) {
  homogenized = false;
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

InhomogenousTransformationRectangular::~InhomogenousTransformationRectangular() {}

Position InhomogenousTransformationRectangular::math_to_phys(Position coord) const {
  Position ret;
  std::pair<int, double> sec = Z_to_Sector_and_local_z(coord[2]);
  double m = case_sectors[sec.first].get_m(sec.second);
  ret[0] = coord[0];
  ret[1] = coord[1] + m;
  ret[2] = coord[2];
  return ret;
}

Position InhomogenousTransformationRectangular::phys_to_math(Position coord) const {
  Position ret;
  std::pair<int, double> sec = Z_to_Sector_and_local_z(coord[2]);
  double m = case_sectors[sec.first].get_m(sec.second);
  ret[0] = coord[0];
  ret[1] = coord[1] - m;
  ret[2] = coord[2];
  return ret;
}

Tensor<2, 3, ComplexNumber>
InhomogenousTransformationRectangular::get_Tensor(Position &position) {
  return get_Space_Transformation_Tensor(position);
}

double InhomogenousTransformationRectangular::get_dof(int dof) const {
  if (dof < (int)NDofs() && dof >= 0) {
    int sector = floor(dof / 2);
    if (sector == sectors) {
      return case_sectors[sector - 1].dofs_r[dof % 2];
    } else {
      return case_sectors[sector].dofs_l[dof % 2];
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in "
                 "InhomogenousTransformationRectangular::get_dof!"
              << std::endl;
    return 0.0;
  }
}

double InhomogenousTransformationRectangular::get_free_dof(int in_dof) const {
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
                 "InhomogenousTransformationRectangular::get_free_dof!"
              << std::endl;
    return 0.0;
  }
}

void InhomogenousTransformationRectangular::set_dof(int dof, double in_val) {
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
                 "InhomogenousTransformationRectangular::set_dof!"
              << std::endl;
  }
}

void InhomogenousTransformationRectangular::set_free_dof(int in_dof,
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
                 "InhomogenousTransformationRectangular::set_free_dof!"
              << std::endl;
  }
}

void InhomogenousTransformationRectangular::estimate_and_initialize() {
  if (GlobalParams.Use_Predefined_Shape) {
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
    double m_0 = 0;
    double m_1 = GlobalParams.Vertical_displacement_of_waveguide;
    double z_start = 0;
    double z_end = GlobalParams.Geometry_Size_Z;
    double v_0 = 0;
    double v_1 = 0;
    const double length = GlobalParams.Sector_thickness;
    if (sectors == 1) {
      Sector<2> temp12(true, true, z_start, z_end);
      case_sectors.push_back(temp12);
      case_sectors[0].set_properties_force(m_0, m_1, GlobalParams.Height_of_waveguide, GlobalParams.Height_of_waveguide, v_0, v_1);
    } else {
      
      Sector<2> temp(true, false, z_start, z_start + length);
      case_sectors.push_back(temp);
      for (int i = 1; i < sectors; i++) {
        Sector<2> temp2(false, false,z_start + length * (1.0 * i), z_start + length * (i + 1.0));
        case_sectors.push_back(temp2);
      }

      double length_rel = 1.0 / ((double)(sectors));
      case_sectors[0].set_properties_force(m_0, InterpolationPolynomialZeroDerivative(length_rel, m_0, m_1), 0, InterpolationPolynomialDerivative(length_rel, m_0, m_1, 0, 0));
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

double InhomogenousTransformationRectangular::get_r(double) const {
  std::cout << "Asking for Radius of rectangular Waveguide." << std::endl;
  return 0;
}

double InhomogenousTransformationRectangular::get_m(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].get_m(two.second);
}

double InhomogenousTransformationRectangular::get_v(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].get_v(two.second);
}

Vector<double> InhomogenousTransformationRectangular::Dofs() const {
  Vector<double> ret;
  const int total = NDofs();
  ret.reinit(total);
  for (int i = 0; i < total; i++) {
    ret[i] = get_dof(i);
  }
  return ret;
}

unsigned int InhomogenousTransformationRectangular::NFreeDofs() const {
  return NDofs() - 4;
}

bool InhomogenousTransformationRectangular::IsDofFree(int index) const {
  return index > 1 && index < (int)NDofs() - 1;
}

void InhomogenousTransformationRectangular::Print() const {
  std::cout << "Printing is not yet implemented." << std::endl;
}

unsigned int InhomogenousTransformationRectangular::NDofs() const {
  return sectors * 2 + 2;
}

Tensor<2, 3, double>
InhomogenousTransformationRectangular::get_Space_Transformation_Tensor(Position &position) {
  Tensor<2, 3, double> J_loc = get_J(position);
  Tensor<2, 3, double> ret;
  ret[0][0] = 1;
  ret[1][1] = 1;
  ret[2][2] = 1;
  return (J_loc * ret * transpose(J_loc)) / determinant(J_loc);
}

Tensor<2,3,double> InhomogenousTransformationRectangular::get_J(Position &in_p) {
  Tensor<2,3,double> ret = I;
  ret[1][2] = get_v(in_p[2]);
  return ret;
}

Tensor<2,3,double> InhomogenousTransformationRectangular::get_J_inverse(Position &in_p) {
  Tensor<2,3,double> ret = get_J(in_p);
  return invert(ret);
}