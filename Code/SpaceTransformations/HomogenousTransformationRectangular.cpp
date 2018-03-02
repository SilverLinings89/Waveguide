#ifndef HOMOGENOUS_TRANSFORMATION_RECTANGULAR_CPP
#define HOMOGENOUS_TRANSFORMATION_RECTANGULAR_CPP

#include "HomogenousTransformationRectangular.h"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include "SpaceTransformation.h"
#include "../Helpers/QuadratureFormulaCircle.cpp"
#include "../Helpers/staticfunctions.h"
#include "../Core/Sector.h"
using namespace dealii;

HomogenousTransformationRectangular::HomogenousTransformationRectangular (int in_rank):
    SpaceTransformation(2, in_rank),
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
  homogenized = true;

}

HomogenousTransformationRectangular::~HomogenousTransformationRectangular() {

}

Point<3, double> HomogenousTransformationRectangular::math_to_phys(Point<3, double> coord) const {
  Point<3, double> ret;
  if(coord[2] < GlobalParams.M_R_ZLength/(-2.0)) {
    ret[0] = (2*GlobalParams.M_C_Dim1In) * coord[0] / (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out);
    ret[1] = (2*GlobalParams.M_C_Dim2In) * coord[1] / (GlobalParams.M_C_Dim2In + GlobalParams.M_C_Dim2Out);
    ret[2] = coord[2];
  } else if(coord[2] >= GlobalParams.M_R_ZLength/(-2.0) && coord[2] < GlobalParams.M_R_ZLength/(2.0)) {
  	std::pair<int, double> sec = Z_to_Sector_and_local_z(coord[2]);
		double m = case_sectors[sec.first].get_m(sec.second);
		ret[0] = coord[0] ;
		ret[1] = (coord[1] -m) ;
		ret[2] = coord[2];
  } else {
    ret[0] = (2*GlobalParams.M_C_Dim1Out) * coord[0] / (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out);
    ret[1] = (2*GlobalParams.M_C_Dim2Out) * coord[1] / (GlobalParams.M_C_Dim2In + GlobalParams.M_C_Dim2Out);
    ret[2] = coord[2];
  }
  return ret;
}

Point<3, double> HomogenousTransformationRectangular::phys_to_math(Point<3, double> coord) const {
  Point<3, double> ret;
  if(coord[2] < GlobalParams.M_R_ZLength/(-2.0)) {
    ret[0] = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) * coord[0] / (2*GlobalParams.M_C_Dim1In);
    ret[1] = (GlobalParams.M_C_Dim2In + GlobalParams.M_C_Dim2Out) * coord[1] / (2*GlobalParams.M_C_Dim2In);
    ret[2] = coord[2];
  } else if(coord[2] >= GlobalParams.M_R_ZLength/(-2.0) && coord[2] < GlobalParams.M_R_ZLength/(2.0)) {
  	std::pair<int, double> sec = Z_to_Sector_and_local_z(coord[2]);
		double m = case_sectors[sec.first].get_m(sec.second);
		ret[0] = coord[0] ;
		ret[1] = (coord[1] ) +m;
		ret[2] = coord[2];
  } else {
    ret[0] = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out) * coord[0] / (2*GlobalParams.M_C_Dim1In);
    ret[1] = (GlobalParams.M_C_Dim2In + GlobalParams.M_C_Dim2Out) * coord[1] / (2*GlobalParams.M_C_Dim2In);
    ret[2] = coord[2];
  }
  return ret;
}


bool HomogenousTransformationRectangular::PML_in_X(Point<3, double> &p) const {
  return p(0) < XMinus ||p(0) > XPlus;
}

bool HomogenousTransformationRectangular::PML_in_Y(Point<3, double> &p) const {
  return p(1) < YMinus ||p(1) > YPlus;
}

bool HomogenousTransformationRectangular::PML_in_Z(Point<3, double> &p)  const{
  return p(2) < ZMinus ||p(2) > ZPlus;
}

double HomogenousTransformationRectangular::Preconditioner_PML_Z_Distance(Point<3, double> &p, unsigned int rank ) const{
  double width = GlobalParams.LayerThickness * 1.0;

  return p(2) +GlobalParams.M_R_ZLength/2.0 - ((double)rank)*width;

}

double HomogenousTransformationRectangular::PML_X_Distance(Point<3, double> &p) const{
  if(p(0) >0){
    return p(0) - XPlus ;
  } else {
    return -p(0) + XMinus;
  }
}

double HomogenousTransformationRectangular::PML_Y_Distance(Point<3, double> &p) const{
  if(p(1) >0){
    return p(1) - YPlus;
  } else {
    return -p(1) + YMinus;
  }
}

double HomogenousTransformationRectangular::PML_Z_Distance(Point<3, double> &p) const{
  if(p(2) < 0) {
    return - (p(2) + (GlobalParams.M_R_ZLength / 2.0));
  } else {
    return p(2) - (GlobalParams.M_R_ZLength / 2.0);
  }
}

Tensor<2,3,std::complex<double>> HomogenousTransformationRectangular::get_Tensor(Point<3, double> & position) const {
  Tensor<2,3,double> transform = get_Space_Transformation_Tensor_Homogenized(position);
  return Apply_PML_To_Tensor(position, transform);
}

Tensor<2,3,std::complex<double>> HomogenousTransformationRectangular::get_Preconditioner_Tensor(Point<3, double> & position, int block) const {
  Tensor<2,3,double> transform = get_Space_Transformation_Tensor_Homogenized(position);
  return Apply_PML_To_Tensor_For_Preconditioner(position, transform, block);
}

Tensor<2,3,double> HomogenousTransformationRectangular::get_Space_Transformation_Tensor_Homogenized(Point<3, double> & position) const {
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

  return transformation;
}

Tensor<2,3,double> HomogenousTransformationRectangular::get_Space_Transformation_Tensor(Point<3, double> & position) const {
  std::pair<int, double> sector_z = Z_to_Sector_and_local_z(position[2]);

  Tensor<2,3, double> transformation = case_sectors[sector_z.first].TransformationTensorInternal(position[0], position[1], sector_z.second);

  return transformation;
}

Tensor<2,3, std::complex<double>> HomogenousTransformationRectangular::Apply_PML_To_Tensor(Point<3, double> & position, Tensor<2,3,double> transformation) const {
	 Tensor<2,3, std::complex<double>> MaterialTensor;

	    for(int i = 0; i < 3; i++) {
	      for(int j = 0; j < 3; j++) {
	        MaterialTensor[i][j] = transformation[i][j]* std::complex<double>(1.0, 0.0);
	      }
	    }

	    std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);

	    if(PML_in_X(position)){
	      double r,d;
	      r = PML_X_Distance(position);
	      if(position[0] < 0){
	        d = GlobalParams.M_BC_XMinus;
	      } else {
	        d = GlobalParams.M_BC_XPlus;
	      }
	      // sx.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaXMax );
	      sx.imag( ((r*r)/(d*d))*GlobalParams.M_BC_SigmaXMax );
	    }

	    if(PML_in_Y(position)){
	      double r,d;
	      r = PML_Y_Distance(position);
	      if(position[1] < 0){
	        d = GlobalParams.M_BC_YMinus;
	      } else {
	        d = GlobalParams.M_BC_YPlus;
	      }

	      // sy.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaYMax );
	      sy.imag( pow(r/d , 2) * GlobalParams.M_BC_SigmaYMax);
	    }


	    if(PML_in_Z(position)){
	      double r,d;
	      r = PML_Z_Distance(position);
	      d = GlobalParams.M_BC_Zplus * GlobalParams.LayerThickness;
	      // sz.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaZMax );
	      sz.imag( pow(r/d ,2) * GlobalParams.M_BC_SigmaZMax );
	    }

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

Tensor<2,3, std::complex<double>> HomogenousTransformationRectangular::Apply_PML_To_Tensor_For_Preconditioner(Point<3, double> & position, Tensor<2,3,double> transformation, int) const {
	Tensor<2,3, std::complex<double>> MaterialTensor;

	  for(int i = 0; i < 3; i++) {
	    for(int j = 0; j < 3; j++) {
	      MaterialTensor[i][j] = transformation[i][j]* std::complex<double>(1.0, 0.0);
	    }
	  }

	  std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0),sz_p(0.0,0.0);
	  if(PML_in_X(position)){
	    double r,d;
	    r = PML_X_Distance(position);
	    if(position[0] < 0){
	      d = GlobalParams.M_BC_XMinus;
	    } else {
	      d = GlobalParams.M_BC_XPlus;
	    }
	    // sx.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaXMax );
	    sx.imag( pow(r/d , 2) * GlobalParams.M_BC_SigmaXMax );
	  }
	  if(PML_in_Y(position)){
	    double r,d;
	    r = PML_Y_Distance(position);
	    if(position[1] < 0){
	      d = GlobalParams.M_BC_YMinus;
	    } else {
	      d = GlobalParams.M_BC_YPlus;
	    }

	    // sy.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaYMax );
	    sy.imag( pow(r/d , 2) * GlobalParams.M_BC_SigmaYMax);
	  }

	  if(Preconditioner_PML_Z_Distance(position, rank) > 0){
	  	double r_temp = Preconditioner_PML_Z_Distance(position, rank);
	  	double d_temp = GlobalParams.LayerThickness;
	  // sz_p.real( pow(r_temp/d_temp , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaZMax );

	  	sz.imag( pow(r_temp/d_temp , 2) * GlobalParams.M_BC_SigmaZMax);
	  }

	  if(PML_in_Z(position)){
	    double r,d;
	    r = PML_Z_Distance(position);
	    d = GlobalParams.M_BC_Zplus * GlobalParams.LayerThickness;
	    // sz.real( 1 + pow(r/d , GlobalParams.M_BC_DampeningExponent) * GlobalParams.M_BC_KappaZMax );
	    sz.imag( pow(r/d , 2) * GlobalParams.M_BC_SigmaZMax );
	  }

	  // sz += sz_p;

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

  std::complex<double> ret = 0;
	try{
		// std::cout << "Process " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " computing signal at " << in_z << std::endl;
		ret = gauss_product_2D_sphere(in_z,10,r,0,0, in_w);
	} catch (VectorTools::ExcPointNotAvailableHere &e) {
		// std::cout << "Failed for " << in_z << " in " <<rank << std::endl;
		ret = 0;
	}
	ret.real( Utilities::MPI::sum(ret.real(), MPI_COMM_WORLD));
	ret.imag( Utilities::MPI::sum(ret.imag(), MPI_COMM_WORLD));
  return ret;
}

double HomogenousTransformationRectangular::get_dof(int dof) const {
  if(dof < (int)NDofs() && dof >= 0) {
    int sector = floor(dof/2);
    if(sector == sectors) {
      return case_sectors[sector-1].dofs_r[dof%2];
    } else {
      return case_sectors[sector].dofs_l[dof%2];
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in HomogenousTransformationRectangular::get_dof!" <<std::endl;
    return 0.0;
  }
}

double HomogenousTransformationRectangular::get_free_dof(int in_dof) const {
  int dof = in_dof + 2 ;
  if(dof < (int)NDofs()-2 && dof >= 0) {
    int sector = floor(dof/2);
    if(sector == sectors) {
      return case_sectors[sector-1].dofs_r[dof%2];
    } else {
      return case_sectors[sector].dofs_l[dof%2];
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in HomogenousTransformationRectangular::get_free_dof!" <<std::endl;
    return 0.0;
  }
}

void HomogenousTransformationRectangular::set_dof(int dof, double in_val) {
  if(dof < (int)NDofs() && dof >= 0) {
    int sector = floor(dof/2);
    if(sector == sectors) {
      case_sectors[sector-1].dofs_r[dof%2] = in_val;
    } else if (sector == 0) {
      case_sectors[0].dofs_l[dof%2] = in_val;
    } else {
      case_sectors[sector].dofs_l[dof%2] = in_val;
      case_sectors[sector-1].dofs_r[dof%2] = in_val;
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in HomogenousTransformationRectangular::set_dof!" <<std::endl;
  }
}

void HomogenousTransformationRectangular::set_free_dof(int in_dof, double in_val) {
  int dof = in_dof + 2;
  if(dof < (int)NDofs() -2 && dof >= 0) {
    int sector = floor(dof/2);
    if(sector == sectors) {
      case_sectors[sector-1].dofs_r[dof%2] = in_val;
    } else if (sector == 0) {
      case_sectors[0].dofs_l[dof%2] = in_val;
    } else {
      case_sectors[sector].dofs_l[dof%2] = in_val;
      case_sectors[sector-1].dofs_r[dof%2] = in_val;
    }
  } else {
    std::cout << "Critical: DOF-index out of bounds in HomogenousTransformationRectangular::set_free_dof!" <<std::endl;
  }
}

double HomogenousTransformationRectangular::Sector_Length()  const{
  return GlobalParams.M_R_ZLength / (double)GlobalParams.M_W_Sectors;
}

void HomogenousTransformationRectangular::estimate_and_initialize() {
  if(GlobalParams.M_PC_Use) {
    Sector<2> the_first(true, false, GlobalParams.sd.z[0], GlobalParams.sd.z[1]);
    the_first.set_properties_force(GlobalParams.sd.m[0],GlobalParams.sd.m[1],GlobalParams.sd.v[0],GlobalParams.sd.v[1]);
    case_sectors.push_back(the_first);
    for(int i = 1; i < GlobalParams.sd.Sectors-2; i++) {
      Sector<2> intermediate(false, false, GlobalParams.sd.z[i], GlobalParams.sd.z[i+1] );
      intermediate.set_properties_force(GlobalParams.sd.m[i],GlobalParams.sd.m[i+1],GlobalParams.sd.v[i],GlobalParams.sd.v[i+1]);
      case_sectors.push_back(intermediate);
    }
    Sector<2> the_last(false, true, GlobalParams.sd.z[GlobalParams.sd.Sectors-1], GlobalParams.sd.z[GlobalParams.sd.Sectors]);
    the_last.set_properties_force(GlobalParams.sd.m[GlobalParams.sd.Sectors-1],GlobalParams.sd.m[GlobalParams.sd.Sectors],GlobalParams.sd.v[GlobalParams.sd.Sectors-1],GlobalParams.sd.v[GlobalParams.sd.Sectors]);
    case_sectors.push_back(the_first);
  } else {
    case_sectors.reserve(sectors);
    double m_0 = GlobalParams.M_W_Delta/2.0;
    double m_1 = -GlobalParams.M_W_Delta/2.0;
    if(sectors == 1) {
      Sector<2> temp12(true, true, -GlobalParams.M_R_ZLength/2, GlobalParams.M_R_ZLength/2 );
      case_sectors.push_back(temp12);
      case_sectors[0].set_properties_force(GlobalParams.M_W_Delta/2.0,-GlobalParams.M_W_Delta/2.0, GlobalParams.M_C_Dim1In, GlobalParams.M_C_Dim1Out, 0, 0);
    } else {
      double length = Sector_Length();
      Sector<2> temp(true, false, -GlobalParams.M_R_ZLength/(2.0), -GlobalParams.M_R_ZLength/2.0 + length );
      case_sectors.push_back(temp);
      for(int  i = 1; i < sectors; i++) {
        Sector<2> temp2( false, false, -GlobalParams.M_R_ZLength/(2.0) + length*(1.0 *i), -GlobalParams.M_R_ZLength/(2.0) + length*(i + 1.0) );
        case_sectors.push_back(temp2);
      }

      double length_rel = 1.0/((double)(sectors));
      case_sectors[0].set_properties_force(
          m_0,
          InterpolationPolynomialZeroDerivative(length_rel, m_0, m_1),
          0,
          InterpolationPolynomialDerivative(length_rel, m_0, m_1, 0, 0)
      );
      for(int  i = 1; i < sectors ; i++) {
        double z_l = i*length_rel;
        double z_r = (i+1)*length_rel;
        case_sectors[i].set_properties_force(
            InterpolationPolynomialZeroDerivative(z_l, m_0, m_1),
            InterpolationPolynomialZeroDerivative(z_r, m_0, m_1),
            InterpolationPolynomialDerivative(z_l, m_0, m_1, 0, 0),
            InterpolationPolynomialDerivative(z_r, m_0, m_1, 0, 0)
        );
      }
    }
  }
    // for (unsigned int i = 0;  i < NFreeDofs(); ++ i) {
    //  InitialDofs[i] = this->get_dof(i, true);
    //}

}

double HomogenousTransformationRectangular::get_r(double ) const {
  //std::pair<int, double> two = Z_to_Sector_and_local_z(z_in);
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
  for(int i= 0; i < total; i++ ){
    ret[i] = get_dof(i);
  }
  return ret;
}

unsigned int HomogenousTransformationRectangular::NFreeDofs() const {
  return NDofs() - 4;
}

bool HomogenousTransformationRectangular::IsDofFree(int index) const {
  return index > 1 && index < (int)NDofs()-1;
}

void HomogenousTransformationRectangular::Print ()  const{
  std::cout << "Printing is not yet implemented." << std::endl;
}

unsigned int HomogenousTransformationRectangular::NDofs()  const{
  return sectors * 2 + 2;
}
#endif
