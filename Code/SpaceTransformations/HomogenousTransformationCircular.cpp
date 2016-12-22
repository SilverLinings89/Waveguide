#ifndef HomogenousTransformationCircular_CPP
#define HomogenousTransformationCircular_CPP

#include "../Helpers/staticfunctions.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "HomogenousTransformationCircular.h"

using namespace dealii;

HomogenousTransformationCircular::HomogenousTransformationCircular ():
  XMinus( -(GlobalParams.M_R_XLength *0.5 - GlobalParams.M_BC_XMinus)),
  XPlus( GlobalParams.M_R_XLength *0.5 - GlobalParams.M_BC_XPlus),
  YMinus( -(GlobalParams.M_R_YLength *0.5 - GlobalParams.M_BC_YMinus)),
  YPlus( GlobalParams.M_R_YLength *0.5 - GlobalParams.M_BC_YPlus),
  ZMinus( - GlobalParams.M_R_ZLength *0.5 ),
  ZPlus( GlobalParams.M_R_ZLength *0.5 ),
  deltaY(GlobalParams.M_W_Delta),
  epsilon_K(GlobalParams.M_W_epsilonin),
  epsilon_M(GlobalParams.M_W_epsilonout),
  sectors(GlobalParams.M_W_Sectors)
{


}

bool HomogenousTransformationCircular::PML_in_X(Point<3> &p) {
  return p(0) < XMinus ||p(0) > XPlus;
}

bool HomogenousTransformationCircular::PML_in_Y(Point<3> &p) {
  return p(1) < YMinus ||p(1) > YPlus;
}

bool HomogenousTransformationCircular::PML_in_Z(Point<3> &p) {
  return p(2) < ZMinus ||p(2) > ZPlus;
}

bool HomogenousTransformationCircular::Preconditioner_PML_in_Z(Point<3> &p, unsigned int block) {
  double l = structure->Layer_Length();
  double width = l * 1.0;
  if( block == GlobalParams.NumberProcesses-2) return false;
  if ( block == GlobalParams.MPI_Rank-1){
    return true;
  } else {
    return false;
  }
}

double HomogenousTransformationCircular::Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block ){
  double l = structure->Layer_Length();
  double width = l * 1.0;

  return p(2) +GlobalParams.M_R_ZLength/2.0 - ((double)block +1)*l;

}

double HomogenousTransformationCircular::PML_X_Distance(Point<3> &p){
  if(p(0) >0){
    return p(0) - XPlus ;
  } else {
    return -p(0) - XMinus;
  }
}

double HomogenousTransformationCircular::PML_Y_Distance(Point<3> &p){
  if(p(1) >0){
    return p(1) - YMinus;
  } else {
    return -p(1) - YPlus;
  }
}

double HomogenousTransformationCircular::PML_Z_Distance(Point<3> &p){
  if(p(3) < 0) {
    return - (p(2) + (GlobalParams.M_R_ZLength / 2.0));
  } else {
    return p(2) - (GlobalParams.M_R_ZLength / 2.0);
  }
}

Tensor<2,3, std::complex<double>> HomogenousTransformationCircular::get_Tensor(Point<3> & position) {
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
  Tensor<2,3, double> transformation = structure->TransformationTensor(position[0], position[1], position[2]);
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

Tensor<2,3, std::complex<double>> HomogenousTransformationCircular::get_Preconditioner_Tensor(Point<3> & position, int block) {
  std::complex<double> S1(1.0, 0.0),S2(1.0,0.0), S3(1.0,0.0);
  Tensor<2,3, std::complex<double>> ret;

  Tensor<2,3, std::complex<double>> MaterialTensor;
  Tensor<2,3, double> transformation = structure->TransformationTensor(position[0], position[1], position[2]);
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

std::complex<double> HomogenousTransformationCircular::gauss_product_2D_sphere(double z, int n, double R, double Xc, double Yc, Waveguide * in_w)
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

std::complex<double> HomogenousTransformationCircular::evaluate_for_z(double in_z, Waveguide * in_w) {
  double r = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/2.0;

  std::complex<double> res = gauss_product_2D_sphere(in_z,10,r,0,0, in_w);
  return sqrt(std::norm(res));
}

#endif
