#ifndef HOMOGENOUS_TRANSFORMATION_RECTANGULAR_CPP
#define HOMOGENOUS_TRANSFORMATION_RECTANGULAR_CPP

#include "HomogenousTransformationRectangular.h"
#include "../Helpers/staticfunctions.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"

using namespace dealii;

HomogenousTransformationRectangular::HomogenousTransformationRectangular ():
    SpaceTransformation(3),
  XMinus( -(GlobalParams.M_R_XLength *0.5 - GlobalParams.M_BC_XMinus)),
  XPlus( GlobalParams.M_R_XLength *0.5 - GlobalParams.M_BC_XPlus),
  YMinus( -(GlobalParams.M_R_YLength *0.5 - GlobalParams.M_BC_YMinus)),
  YPlus( GlobalParams.M_R_YLength *0.5 - GlobalParams.M_BC_YPlus),
  ZMinus( - GlobalParams.M_R_ZLength *0.5 ),
  ZPlus( GlobalParams.M_R_ZLength *0.5 ),
  epsilon_K(GlobalParams.M_W_epsilonin),
  epsilon_M(GlobalParams.M_W_epsilonout),
  sectors(GlobalParams.M_W_Sectors),
  deltaY(GlobalParams.M_W_Delta)
{


}

HomogenousTransformationRectangular::~HomogenousTransformationRectangular() {

}

Point<3> HomogenousTransformationRectangular::math_to_phys(Point<3> coord) const {
  Point<3> ret;
  if(coord[2] < GlobalParams.M_R_ZLength/(-2.0)) {
    ret[0] = (2*GlobalParams.M_C_Dim1In) * coord[0] / (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out);
    ret[1] = (2*GlobalParams.M_C_Dim1In) * coord[1] / (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out);
    ret[2] = coord[2];
  } else if(coord[2] >= GlobalParams.M_R_ZLength/(-2.0) && coord[2] < GlobalParams.M_R_ZLength/(2.0)) {
   // TODO: Use sectors here.
  } else {
    ret[0] = (2*GlobalParams.M_C_Dim1Out) * coord[0] / (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out);
    ret[1] = (2*GlobalParams.M_C_Dim1Out) * coord[1] / (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out);
    ret[2] = coord[2];
  }
  return ret;
}

Point<3> HomogenousTransformationRectangular::phys_to_math(Point<3> coord) const {
  Point<3> ret;
  if(coord[2] < GlobalParams.M_R_ZLength/(-2.0)) {
    ret[0] = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) * coord[0] / (2*GlobalParams.M_C_Dim1In);
    ret[1] = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) * coord[1] / (2*GlobalParams.M_C_Dim1In);
    ret[2] = coord[2];
  } else if(coord[2] >= GlobalParams.M_R_ZLength/(-2.0) && coord[2] < GlobalParams.M_R_ZLength/(2.0)) {
   // TODO: Use sectors here.
  } else {
    ret[0] = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) * coord[0] / (2*GlobalParams.M_C_Dim1In);
    ret[1] = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) * coord[1] / (2*GlobalParams.M_C_Dim1In);
    ret[2] = coord[2];
  }
  return ret;
}

bool HomogenousTransformationRectangular::PML_in_X(Point<3> &p) const {
  return p(0) < XMinus ||p(0) > XPlus;
}

bool HomogenousTransformationRectangular::PML_in_Y(Point<3> &p) const {
  return p(1) < YMinus ||p(1) > YPlus;
}

bool HomogenousTransformationRectangular::PML_in_Z(Point<3> &p)  const{
  return p(2) < ZMinus ||p(2) > ZPlus;
}

bool HomogenousTransformationRectangular::Preconditioner_PML_in_Z(Point<3> &, unsigned int block) const {
  if( (int)block == GlobalParams.NumberProcesses-2) return false;
  if ( (int)block == (int)GlobalParams.MPI_Rank-1){
    return true;
  } else {
    return false;
  }
}

double HomogenousTransformationRectangular::Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block ) const{
  double width = GlobalParams.LayerThickness * 1.0;

  return p(2) +GlobalParams.M_R_ZLength/2.0 - ((double)block +1)*width;

}

double HomogenousTransformationRectangular::PML_X_Distance(Point<3> &p) const{
  if(p(0) >0){
    return p(0) - XPlus ;
  } else {
    return -p(0) - XMinus;
  }
}

double HomogenousTransformationRectangular::PML_Y_Distance(Point<3> &p) const{
  if(p(1) >0){
    return p(1) - YMinus;
  } else {
    return -p(1) - YPlus;
  }
}

double HomogenousTransformationRectangular::PML_Z_Distance(Point<3> &p) const{
  if(p(2) < 0) {
    return - (p(2) + (GlobalParams.M_R_ZLength / 2.0));
  } else {
    return p(2) - (GlobalParams.M_R_ZLength / 2.0);
  }
}

Tensor<2,3, std::complex<double>> HomogenousTransformationRectangular::get_Tensor(Point<3> & position) const {
  std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);
  Tensor<2,3, std::complex<double>> ret;

  double omegaepsilon0 = GlobalParams.C_omega;
  // * ((System_Coordinate_in_Waveguide(position))?GlobalParams.M_W_epsilonin : GlobalParams.M_W_epsilonout);
  std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);
  if(PML_in_X(position)){
    double r,d, sigmax;
    r = PML_X_Distance(position);
    if(position[0] < 0){
      d = GlobalParams.M_BC_XMinus;
    } else {
      d = GlobalParams.M_BC_XPlus;
    }
    sigmax = pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_SigmaXMax;
    sx.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaXMax);
    sx.imag( sigmax / ( omegaepsilon0));
    S1 /= sx;
    S2 *= sx;
    S3 *= sx;
  }
  if(PML_in_Y(position)){
    double r,d, sigmay;
    r = PML_Y_Distance(position);
    if(position[1] < 0){
      d = GlobalParams.M_BC_YMinus;
    } else {
      d = GlobalParams.M_BC_YPlus;
    }
    sigmay = pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_SigmaYMax;
    sy.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaYMax);
    sy.imag( sigmay / ( omegaepsilon0));
    S1 *= sy;
    S2 /= sy;
    S3 *= sy;
  }
  if(PML_in_Z(position)){
    double r,d, sigmaz;
    r = PML_Z_Distance(position);
    d = (GlobalParams.M_R_ZLength / (GlobalParams.NumberProcesses - GlobalParams.M_BC_Zplus)) * GlobalParams.M_BC_Zplus ;
    sigmaz = pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_SigmaZMax;
    sz.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_SigmaZMax);
    sz.imag( sigmaz / omegaepsilon0 );
    S1 *= sz;
    S2 *= sz;
    S3 /= sz;
  }

  ret[0][0] = S1;
  ret[1][1] = S2;
  ret[2][2] = S3;

  Tensor<2,3, std::complex<double>> ret2;

  std::pair<int, double> sector_z = Z_to_Sector_and_local_z(position[2]);

  Tensor<2,3, double> transformation = case_sectors[sector_z.first].TransformationTensorInternal(position[0], position[1], sector_z.second);
  double dist = position[0] * position[0] + position[1]*position[1];
  dist = sqrt(dist);
  double v1 = GlobalParams.M_R_XLength/2.0 - std::min(GlobalParams.M_BC_XMinus, GlobalParams.M_BC_XPlus);
  double v2 = GlobalParams.M_R_YLength/2.0 - std::min(GlobalParams.M_BC_YMinus, GlobalParams.M_BC_YPlus);
  double maxdist = std::min(v1, v2);
  double mindist = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0;
  double sig = sigma(dist, mindist, maxdist);
  double factor = InterpolationPolynomialZeroDerivative(sig, 1,0);
  transformation *= factor;
  for(int i = 0; i < 3; i++) {
    transformation[i][i] += 1-factor;
  }
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      ret2[i][j] = transformation[i][j]* std::complex<double>(1.0, 0.0);
    }
  }

  Tensor<2,3, std::complex<double>> ret3;

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      ret3[i][j] = std::complex<double>(0.0, 0.0);
      for(int k = 0; k < 3; k++) {
        ret3[i][j] += ret[i][k] * ret2[k][j];
      }
    }
  }


  return ret3;
}

Tensor<2,3, std::complex<double>> HomogenousTransformationRectangular::get_Preconditioner_Tensor(Point<3> & position, int block) const {
  std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);
  Tensor<2,3, std::complex<double>> ret;

  Tensor<2,3, std::complex<double>> MaterialTensor;

  std::pair<int, double> sector_z = Z_to_Sector_and_local_z(position[2]);

  Tensor<2,3, double> transformation = case_sectors[sector_z.first].TransformationTensorInternal(position[0], position[1], sector_z.second);

  // Tensor<2,3, double> transformation = structure->TransformationTensor(position[0], position[1], position[2]);
  double dist = position[0] * position[0] + position[1]*position[1];
  dist = sqrt(dist);
  double v1 = GlobalParams.M_R_XLength/2.0 - std::min(GlobalParams.M_BC_XMinus, GlobalParams.M_BC_XPlus);
  double v2 = GlobalParams.M_R_YLength/2.0 - std::min(GlobalParams.M_BC_YMinus, GlobalParams.M_BC_YPlus);
  double maxdist = std::min(v1, v2);
  double mindist = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0;
  double sig = sigma(dist, mindist, maxdist);
  double factor = InterpolationPolynomialZeroDerivative(sig, 1,0);
  transformation *= factor;
  for(int i = 0; i < 3; i++) {
    transformation[i][i] += 1-factor;
  }
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      MaterialTensor[i][j] = transformation[i][j]* std::complex<double>(1.0, 0.0);
    }
  }

  double omegaepsilon0 = GlobalParams.C_omega;
  std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0),sz_p(0.0,0.0);
  if(PML_in_X(position)){
    double r,d, sigmax;
    r = PML_X_Distance(position);
    if(position[0] < 0){
      d = GlobalParams.M_BC_XMinus;
    } else {
      d = GlobalParams.M_BC_XPlus;
    }
    sigmax = pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_SigmaXMax;
    sx.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaXMax);
    sx.imag( sigmax / ( omegaepsilon0));
  }
  if(PML_in_Y(position)){
    double r,d, sigmay;
    r = PML_Y_Distance(position);
    if(position[1] < 0){
      d = GlobalParams.M_BC_YMinus;
    } else {
      d = GlobalParams.M_BC_YPlus;
    }
    sigmay = pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_SigmaYMax;
    sy.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaYMax);
    sy.imag( sigmay / ( omegaepsilon0));
  }
  if(Preconditioner_PML_in_Z(position, block)){
    double r,d, sigmaz;
    r = Preconditioner_PML_Z_Distance(position, block);
    d = GlobalParams.LayerThickness;
    sigmaz = pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_SigmaZMax;
    sz_p.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaZMax);
    sz_p.imag( sigmaz / omegaepsilon0 );
  }

  if(PML_in_Z(position)){
    double r,d, sigmaz;
    r = PML_Z_Distance(position);
    d = GlobalParams.M_BC_Zplus * GlobalParams.LayerThickness;
    sigmaz = pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_SigmaZMax;
    sz.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaZMax);
    sz.imag( sigmaz / omegaepsilon0 );
  }

  sz += sz_p;

  MaterialTensor[0][0] *= sy*sz/sx;
  MaterialTensor[0][1] *= sz;
  MaterialTensor[0][2] *= sy;

  MaterialTensor[1][0] *= sz;
  MaterialTensor[1][1] *= sx*sz/sy;
  MaterialTensor[1][2] *= sx;

  MaterialTensor[2][0] *= sy;
  MaterialTensor[2][1] *= sx;
  MaterialTensor[2][2] *= sx*sy/sz;

  return MaterialTensor;
}

std::complex<double> HomogenousTransformationRectangular::gauss_product_2D_sphere(double z, int n, double R, double Xc, double Yc, Waveguide * in_w)
{
  double* r = NULL;
  double* t = NULL;
  double* q = NULL;
  double* A = NULL;
  double  B;
  double x, y;
  std::complex<double> s(0.0, 0.0);

  int i,j;

  /* Load appropriate predefined table */
  for (i = 0; i<GSPHERESIZE;i++)
  {
    if(n==gsphere[i].n)
    {
      r = gsphere[i].r;
      t = gsphere[i].t;
      q = gsphere[i].q;
      A = gsphere[i].A;
      B = gsphere[i].B;
      break;
    }
  }

  if (NULL==r) return -1.0;

  for (i=0;i<n;i++)
  {
    for (j=0;j<n;j++)
    {
      x = r[j]*q[i];
      y = r[j]*t[i];
      s += A[j]*in_w->evaluate_for_Position(R*x-Xc,R*y-Yc,z);
    }
  }

  s *= R*R*B;

  return s;
}

std::complex<double> HomogenousTransformationRectangular::evaluate_for_z(double in_z, Waveguide * in_w) {
  double r = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0;

  std::complex<double> res = gauss_product_2D_sphere(in_z,10,r,0,0, in_w);
  return sqrt(std::norm(res));
}

double HomogenousTransformationRectangular::get_dof(int dof) const {
  if(dof < (int)NDofs() && dof >= 0) {
    int sector = floor(dof/3);
    if(sector == sectors) {
      return case_sectors[sector-1].dofs_r[dof%3];
    } else {
      return case_sectors[sector].dofs_l[dof%3];
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in HomogenousTransformationCircular::get_dof!" <<std::endl;
    return 0.0;
  }
}

double HomogenousTransformationRectangular::get_free_dof(int in_dof) const {
  int dof = in_dof +3 ;
  if(dof < (int)NDofs()-3 && dof >= 0) {
    int sector = floor(dof/3);
    if(sector == sectors) {
      return case_sectors[sector-1].dofs_r[dof%3];
    } else {
      return case_sectors[sector].dofs_l[dof%3];
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in HomogenousTransformationCircular::get_free_dof!" <<std::endl;
    return 0.0;
  }
}

void HomogenousTransformationRectangular::set_dof(int dof, double in_val) {
  if(dof < (int)NDofs() && dof >= 0) {
    int sector = floor(dof/3);
    if(sector == sectors) {
      case_sectors[sector-1].dofs_r[dof%3] = in_val;
    } else if (sector == 0) {
      case_sectors[0].dofs_l[dof%3] = in_val;
    } else {
      case_sectors[sector].dofs_l[dof%3] = in_val;
      case_sectors[sector-1].dofs_r[dof%3] = in_val;
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in HomogenousTransformationCircular::set_dof!" <<std::endl;
  }
}

void HomogenousTransformationRectangular::set_free_dof(int in_dof, double in_val) {
  int dof = in_dof + 3;
  if(dof < (int)NDofs() -3 && dof >= 0) {
    int sector = floor(dof/3);
    if(sector == sectors) {
      case_sectors[sector-1].dofs_r[dof%3] = in_val;
    } else if (sector == 0) {
      case_sectors[0].dofs_l[dof%3] = in_val;
    } else {
      case_sectors[sector].dofs_l[dof%3] = in_val;
      case_sectors[sector-1].dofs_r[dof%3] = in_val;
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in HomogenousTransformationCircular::set_free_dof!" <<std::endl;
  }
}

double HomogenousTransformationRectangular::Sector_Length()  const{
  return GlobalParams.M_R_ZLength / (double)GlobalParams.M_W_Sectors;
}

void HomogenousTransformationRectangular::estimate_and_initialize() {
    case_sectors.reserve(sectors);
    double m_0 = GlobalParams.M_W_Delta/2.0;
    double m_1 = -GlobalParams.M_W_Delta/2.0;
    double r_0 = GlobalParams.M_C_Dim1In;
    double r_1 = GlobalParams.M_C_Dim1Out;
    if(sectors == 1) {
      Sector<4> temp12(true, true, -GlobalParams.M_R_ZLength/2, GlobalParams.M_R_ZLength/2 );
      case_sectors.push_back(temp12);
      case_sectors[0].set_properties_force(GlobalParams.M_W_Delta/2.0,-GlobalParams.M_W_Delta/2.0, GlobalParams.M_C_Dim1In, GlobalParams.M_C_Dim1Out, 0, 0);
    } else {
      double length = Sector_Length();
      Sector<4> temp(true, false, -GlobalParams.M_R_ZLength/(2.0), -GlobalParams.M_R_ZLength/2.0 + length );
      case_sectors.push_back(temp);
      for(int  i = 1; i < sectors; i++) {
        Sector<4> temp2( false, false, -GlobalParams.M_R_ZLength/(2.0) + length*(1.0 *i), -GlobalParams.M_R_ZLength/(2.0) + length*(i + 1.0) );
        case_sectors.push_back(temp2);
      }

      double length_rel = 1.0/((double)(sectors));
      case_sectors[0].set_properties_force(
          m_0,
          InterpolationPolynomialZeroDerivative(length_rel, m_0, m_1),
          r_0,
          InterpolationPolynomialZeroDerivative(length_rel, r_0, r_1),
          0,
          InterpolationPolynomialDerivative(length_rel, m_0, m_1, 0, 0)
      );
      for(int  i = 1; i < sectors ; i++) {
        double z_l = i*length_rel;
        double z_r = (i+1)*length_rel;
        case_sectors[i].set_properties_force(
            InterpolationPolynomialZeroDerivative(z_l, m_0, m_1),
            InterpolationPolynomialZeroDerivative(z_r, m_0, m_1),
            InterpolationPolynomialZeroDerivative(z_l, r_0, r_1),
            InterpolationPolynomialZeroDerivative(z_r, r_0, r_1),
            InterpolationPolynomialDerivative(z_l, m_0, m_1, 0, 0),
            InterpolationPolynomialDerivative(z_r, m_0, m_1, 0, 0)
        );
      }
    }

    // for (unsigned int i = 0;  i < NFreeDofs(); ++ i) {
    //  InitialDofs[i] = this->get_dof(i, true);
    //}

}

double HomogenousTransformationRectangular::get_r(double z_in) const {
  std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
  return case_sectors[two.first].get_r(two.second);
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
  for(int i= 0; i < total; i++ ){
    ret[i] = get_dof(i);
  }
  return ret;
}

unsigned int HomogenousTransformationRectangular::NFreeDofs() const {
  return NDofs() - 6;
}

bool HomogenousTransformationRectangular::IsDofFree(int index) const {
  return index > 2 && index < (int)NDofs()-3;
}

void HomogenousTransformationRectangular::Print ()  const{
  std::cout << "Printing is not yet implemented." << std::endl;
}

unsigned int HomogenousTransformationRectangular::NDofs()  const{
  return sectors * 3 + 3;
}
#endif
