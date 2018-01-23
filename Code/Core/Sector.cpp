#ifndef SectorFlagCPP
#define SectorFlagCPP

#include "Sector.h"

#include <deal.II/base/tensor.h>
#include "../Helpers/staticfunctions.h"
#include <deal.II/base/tensor.h>
using namespace dealii;


template<unsigned int Dofs_Per_Sector> Sector<Dofs_Per_Sector>::~Sector() {

}

template<> Sector<3>::~Sector() {

}

template<> Sector<2>::~Sector() {

}

template<unsigned int Dofs_Per_Sector> Sector<Dofs_Per_Sector>::Sector(bool in_left, bool in_right, double in_z_0, double in_z_1):left(in_left), right(in_right), boundary(in_left && in_right), z_0(in_z_0), z_1(in_z_1) {
  dofs_l = new double[Dofs_Per_Sector];
  dofs_r = new double[Dofs_Per_Sector];
  derivative = new unsigned int[Dofs_Per_Sector];
  zero_derivative = new bool[Dofs_Per_Sector];
  if(Dofs_Per_Sector == 3) {
    zero_derivative[0] = true;
    zero_derivative[1] = false;
    zero_derivative[2] = true;
    derivative[0] = 0;
    derivative[1] = 2;
    derivative[2] = 0;
  }
  if(Dofs_Per_Sector == 2) {
    zero_derivative[0] = false;
    zero_derivative[1] = true;
    derivative[0] = 1;
    derivative[1] = 0;
  }

  for(unsigned int i = 0; i < Dofs_Per_Sector; i++) {
    dofs_l[i] = 0;
    dofs_r[i] = 0;

  }
	NInternalBoundaryDofs =0;
	LowestDof = 0;
	NActiveCells = 0;
	NDofs = Dofs_Per_Sector;
}



template<> void Sector<2>::set_properties(double , double, double, double, double, double) {
  std::cout << "Wrong call in Sector." << std::endl;
}

template<> void Sector<3>::set_properties(double , double , double, double) {
  std::cout << "Wrong call in Sector." << std::endl;
}

template<> void Sector<2>::set_properties(double in_m_0, double in_m_1, double in_v_0, double in_v_1) {
  if(!left) {
    dofs_l[0] = in_m_0;
    dofs_l[1] = in_v_0;
  }
  if(!right) {
    dofs_r[0] = in_m_1;
    dofs_r[1] = in_v_1;
  }

}

template<> void Sector<3>::set_properties(double in_m_0, double in_m_1, double in_r_0, double in_r_1, double in_v_0, double in_v_1) {
  if(!left) {
    dofs_l[0] = in_r_0;
    dofs_l[1] = in_m_0;
    dofs_l[2] = in_v_0;
  }
  if(!right) {
    dofs_r[0] = in_r_1;
    dofs_r[1] = in_m_1;
    dofs_r[2] = in_v_1;
  }
}

template<> void Sector<2>::set_properties_force(double , double, double, double, double, double) {
  std::cout << "Wrong call in Sector." << std::endl;
}

template<> void Sector<3>::set_properties_force(double , double , double, double) {
  std::cout << "Wrong call in Sector." << std::endl;
}

template<> void Sector<2>::set_properties_force(double in_m_0, double in_m_1, double in_v_0, double in_v_1) {
    dofs_l[0] = in_m_0;
    dofs_l[1] = in_v_0;
    dofs_r[0] = in_m_1;
    dofs_r[1] = in_v_1;
}

template<> void Sector<3>::set_properties_force(double in_m_0, double in_m_1, double in_r_0, double in_r_1, double in_v_0, double in_v_1) {
    dofs_l[0] = in_r_0;
    dofs_l[1] = in_m_0;
    dofs_l[2] = in_v_0;
    dofs_r[0] = in_r_1;
    dofs_r[1] = in_m_1;
    dofs_r[2] = in_v_1;
}

template<unsigned int Dofs_Per_Sector> void Sector<Dofs_Per_Sector>::set_properties( double , double , double, double ) {
  std::cout << "The code does not work for this number of dofs per Sector." << std::endl;
  return;
}

template<unsigned int Dofs_Per_Sector> void Sector<Dofs_Per_Sector>::set_properties(double , double, double, double, double, double) {
  std::cout << "The code does not work for this number of dofs per Sector." << std::endl;
  return;
}

template<unsigned int Dofs_Per_Sector> void Sector<Dofs_Per_Sector>::set_properties_force( double , double , double, double ) {
  std::cout << "The code does not work for this number of dofs per Sector." << std::endl;
  return;
}

template<unsigned int Dofs_Per_Sector> void Sector<Dofs_Per_Sector>::set_properties_force(double , double, double, double, double, double) {
  std::cout << "The code does not work for this number of dofs per Sector." << std::endl;
  return;
}

template<unsigned int Dofs_Per_Sector> double Sector<Dofs_Per_Sector>::get_dof(unsigned int i, double z) const {
  if(i > 0 && i < NDofs) {
    if ( z < 0.0 ) z = 0.0;
    if ( z > 1.0 ) z = 1.0;
    if(zero_derivative[i]) {
      return InterpolationPolynomialZeroDerivative(z, dofs_l[i], dofs_r[i]);
    } else {
      return InterpolationPolynomial(z, dofs_l[i], dofs_r[i], dofs_l[derivative[i]], dofs_r[derivative[i]]);
    }
  } else {
    std::cout << "There seems to be an error in Sector::get_dof. i > 0 && i < dofs_per_sector false."<<std::endl;
    return 0;
  }

}

template<unsigned int Dofs_Per_Sector> double Sector<Dofs_Per_Sector>::get_r(double z) const {
	if ( z < 0.0 ) z = 0.0;
	if ( z > 1.0 ) z = 1.0;
	if(Dofs_Per_Sector < 3){
	  deallog << "Error in Sector: Acces to radius dof without existence." << std::endl;
	  return 0;
	}
	return InterpolationPolynomialZeroDerivative(z, dofs_l[0], dofs_r[0]);
}

template<unsigned int Dofs_Per_Sector> double Sector<Dofs_Per_Sector>::get_m(double z) const {
	if ( z < 0.0 ) z = 0.0;
	if ( z > 1.0 ) z = 1.0;
	if(Dofs_Per_Sector==2) {
	  return InterpolationPolynomial(z, dofs_l[0], dofs_r[0], dofs_l[1], dofs_r[1]);
	} else {
	  return InterpolationPolynomial(z, dofs_l[1], dofs_r[1], dofs_l[2], dofs_r[2]);
	}

}

template<unsigned int Dofs_Per_Sector> double Sector<Dofs_Per_Sector>::get_v(double z) const {
	if ( z < 0.0 ) z = 0.0;
	if ( z > 1.0 ) z = 1.0;
	if(Dofs_Per_Sector==2) {
    return InterpolationPolynomialZeroDerivative(z, dofs_l[1], dofs_r[1]);
  } else {
    return InterpolationPolynomialZeroDerivative(z, dofs_l[2], dofs_r[2]);
  }
}

template<unsigned int Dofs_Per_Sector> double Sector<Dofs_Per_Sector>::getQ1 ( double z) const {
	return 1/(dofs_l[0] + z*z*z*(2*dofs_l[0] - 2*dofs_r[0]) - z*z*(3*dofs_l[0] - 3*dofs_r[0]));
}

template<unsigned int Dofs_Per_Sector> double Sector<Dofs_Per_Sector>::getQ2 ( double z) const {
	return 1/(dofs_l[0] + z*z*z*(2*dofs_l[0] - 2*dofs_r[0]) - z*z*(3*dofs_l[0] - 3*dofs_r[0]));
}

template<unsigned int Dofs_Per_Sector> double Sector<Dofs_Per_Sector>::getQ3 ( double )  const {
  return 0.0;
}

template<> Tensor<2,3, double> Sector<3>::TransformationTensorInternal (double in_x, double in_y, double z) const {
	if(z<0 || z>1) std::cout << "Falty implementation of internal Tensor calculation: z: " << z << std::endl;
	double RadiusInMultiplyer = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/(2* dofs_l[0]);
	double RadiusOutMultiplyer = (GlobalParams.M_C_Dim1In + GlobalParams.M_C_Dim1Out)/(2* dofs_r[0]);

	double temp = 1 / (RadiusInMultiplyer - 3*RadiusInMultiplyer*z*z + 2*RadiusInMultiplyer*z*z*z + 3*RadiusOutMultiplyer*z*z - 2*RadiusOutMultiplyer*z*z*z);
	double zz = z*z;
	double zzz = zz*z;
	double zzzz = zz*zz;
	double zzzzz = zzz*zz;
	double help1 = (RadiusInMultiplyer - 3*RadiusInMultiplyer*zz + 2*RadiusInMultiplyer*zzz + 3*RadiusOutMultiplyer*zz - 2*RadiusOutMultiplyer*zzz);
	double help2 = (help1*(dofs_l[2] - 6*dofs_l[1]*z + 6*dofs_r[1]*z - 4*dofs_l[2]*z - 2*dofs_r[2]*z + 6*dofs_l[1]*zz - 6*dofs_r[1]*zz + 3*dofs_l[2]*zz + 3*dofs_r[2]*zz) + 6*z*(RadiusInMultiplyer - RadiusOutMultiplyer)*(z - 1)*(dofs_l[1] + zzz*(2*dofs_l[1] - 2*dofs_r[1] + dofs_l[2] + dofs_r[2]) + dofs_l[2]*z - zz*(3*dofs_l[1] - 3*dofs_r[1] + 2*dofs_l[2] + dofs_r[2])))/help1 - (6*in_y*z*(RadiusInMultiplyer - RadiusOutMultiplyer)*(z - 1))/help1;

	Tensor<2,3, double> u;
	u[0][0]= temp;
	u[0][1]= 0.0;
	u[0][2]= 0.0;
	u[1][0]= 0.0;
	u[1][1]= temp;
	u[1][2]= 0.0;
	u[2][0]= -(6*in_x*z*(RadiusInMultiplyer-RadiusOutMultiplyer)*(z - 1))/help1;
	u[2][1]= (RadiusInMultiplyer*dofs_l[2] + 12*dofs_l[1]*RadiusInMultiplyer*zz + 36*dofs_l[1]*RadiusInMultiplyer*zzz - 6*dofs_l[1]*RadiusOutMultiplyer*zz - 6*dofs_r[1]*RadiusInMultiplyer*zz - 60*dofs_l[1]*RadiusInMultiplyer*zzzz - 36*dofs_l[1]*RadiusOutMultiplyer*zzz - 36*dofs_r[1]*RadiusInMultiplyer*zzz + 24*dofs_l[1]*RadiusInMultiplyer*zzzzz + 60*dofs_l[1]*RadiusOutMultiplyer*zzzz + 60*dofs_r[1]*RadiusInMultiplyer*zzzz + 36*dofs_r[1]*RadiusOutMultiplyer*zzz - 24*dofs_l[1]*RadiusOutMultiplyer*zzzzz - 24*dofs_r[1]*RadiusInMultiplyer*zzzzz - 60*dofs_r[1]*RadiusOutMultiplyer*zzzz + 24*dofs_r[1]*RadiusOutMultiplyer*zzzzz - 6*RadiusInMultiplyer*dofs_l[2]*zz + 32*RadiusInMultiplyer*dofs_l[2]*zzz + 3*RadiusInMultiplyer*dofs_r[2]*zz + 9*RadiusOutMultiplyer*dofs_l[2]*zz - 35*RadiusInMultiplyer*dofs_l[2]*zzzz + 12*RadiusInMultiplyer*dofs_r[2]*zzz - 32*RadiusOutMultiplyer*dofs_l[2]*zzz + 12*RadiusInMultiplyer*dofs_l[2]*zzzzz - 25*RadiusInMultiplyer*dofs_r[2]*zzzz + 35*RadiusOutMultiplyer*dofs_l[2]*zzzz - 12*RadiusOutMultiplyer*dofs_r[2]*zzz + 12*RadiusInMultiplyer*dofs_r[2]*zzzzz - 12*RadiusOutMultiplyer*dofs_l[2]*zzzzz + 25*RadiusOutMultiplyer*dofs_r[2]*zzzz - 12*RadiusOutMultiplyer*dofs_r[2]*zzzzz - 6*RadiusInMultiplyer*in_y*zz + 6*RadiusOutMultiplyer*in_y*zz - 12*dofs_l[1]*RadiusInMultiplyer*z + 6*dofs_l[1]*RadiusOutMultiplyer*z + 6*dofs_r[1]*RadiusInMultiplyer*z - 4*RadiusInMultiplyer*dofs_l[2]*z - 2*RadiusInMultiplyer*dofs_r[2]*z + 6*RadiusInMultiplyer*in_y*z - 6*RadiusOutMultiplyer*in_y*z)/help1;
	u[2][2]= 1.0;
	double Q [3];

	Q[0] = 1/(RadiusInMultiplyer + zzz*(2*RadiusInMultiplyer - 2*RadiusOutMultiplyer) - zz*(3*RadiusInMultiplyer - 3*RadiusOutMultiplyer));
	Q[1] = 1/(RadiusInMultiplyer + zzz*(2*RadiusInMultiplyer - 2*RadiusOutMultiplyer) - zz*(3*RadiusInMultiplyer - 3*RadiusOutMultiplyer));
	if(Q[1] < 0) Q[1] *= -1.0;
	if(Q[0] < 0) Q[0] *= -1.0;
	Q[2] = sqrt( help2*help2 + (36*in_x*in_x*zz*(RadiusInMultiplyer - RadiusOutMultiplyer)*(RadiusInMultiplyer - RadiusOutMultiplyer)*(z - 1)*(z - 1))/(help1*help1) + 1);

	Tensor<2,3,double> ginv;
	for(int i = 0; i<3; i++) {
		for(int j = 0; j<3; j++) {
			for(int k = 0; k< 3; k++) ginv[i][j] += u[i][k] * u[j][k];
		}
	}

	Tensor<2,3,double> g;

	g = invert(ginv);

	double sp = dotproduct(u[0], crossproduct(u[1], u[2]));
	if(sp < 0) sp *= -1.0;
	for(int i = 0; i< 3; i++) {
		for(int j = 0; j<3; j++) {
			g[i][j] *= sp * Q[0]*Q[1]*Q[2] / (Q[i] * Q[j]);
		}
	}

	return g;
}

template<> Tensor<2,3, double> Sector<2>::TransformationTensorInternal (double , double , double z) const {
   if(z<0 || z>1) std::cout << "Falty implementation of internal Tensor calculation: z: " << z << std::endl;

   double zz = z*z;

   Tensor<2,3, double> u;
   u[0][0]= 1;
   u[0][1]= 0.0;
   u[0][2]= 0.0;
   u[1][0]= 0.0;
   u[1][1]= 1;
   u[1][2]= 0.0;
   u[2][0]= 0.0;
   u[2][1]= 6*dofs_l[0]*z - dofs_l[1] - 6*dofs_r[0]*z + 4*dofs_l[1]*z + 2*dofs_r[1]*z - 6*dofs_l[0]*zz + 6*dofs_r[0]*zz - 3*dofs_l[1]*zz - 3*dofs_r[1]*zz;
   u[2][2]= 1.0;
   double Q [3];
   Q[0] = 1;
   Q[1] = 1;
   Q[2] = std::sqrt((dofs_l[1] - 2*z*(3*dofs_l[0] - 3*dofs_r[0] + 2*dofs_l[1] + dofs_r[1]) + 3*zz*std::pow(2*dofs_l[0] - 2*dofs_r[0] + dofs_l[1] + dofs_r[1], 2)) + 1);

   Tensor<2,3,double> ginv;
   for(int i = 0; i<3; i++) {
     for(int j = 0; j<3; j++) {
       for(int k = 0; k< 3; k++) ginv[i][j] += u[i][k] * u[j][k];
     }
   }

   Tensor<2,3,double> g;

   g = invert(ginv);

   double sp = dotproduct(u[0], crossproduct(u[1], u[2]));
   if(sp < 0) sp *= -1.0;
   for(int i = 0; i< 3; i++) {
     for(int j = 0; j<3; j++) {
       g[i][j] *= sp * Q[0]*Q[1]*Q[2] / (Q[i] * Q[j]);
     }
   }

   return g;
 }

template<unsigned int Dimension> Tensor<2,3,double> Sector<Dimension>::TransformationTensorInternal(double , double , double ) const {
  Tensor<2,3,double> ret;
  std::cout << "The code does not work for you Sector specification." << Dimension << std::endl;
  return ret;
}



template<unsigned int Dofs_Per_Sector> unsigned int Sector<Dofs_Per_Sector>::getLowestDof() const {
	return LowestDof;
}

template<unsigned int Dofs_Per_Sector> unsigned int Sector<Dofs_Per_Sector>::getNDofs() const {
	return NDofs;
}

template<unsigned int Dofs_Per_Sector> unsigned int Sector<Dofs_Per_Sector>::getNInternalBoundaryDofs() const {
	return NInternalBoundaryDofs;
}

template<unsigned int Dofs_Per_Sector> unsigned int Sector<Dofs_Per_Sector>::getNActiveCells() const {
	return NActiveCells;
}

template<unsigned int Dofs_Per_Sector> void Sector<Dofs_Per_Sector>::setLowestDof( unsigned int in_lowestdof) {
	LowestDof = in_lowestdof;
}

template<unsigned int Dofs_Per_Sector> void Sector<Dofs_Per_Sector>::setNDofs( unsigned int in_ndofs) {
	NDofs = in_ndofs;
}

template<unsigned int Dofs_Per_Sector> void Sector<Dofs_Per_Sector>::setNInternalBoundaryDofs( unsigned int in_ninternalboundarydofs) {
	NInternalBoundaryDofs = in_ninternalboundarydofs;
}

template<unsigned int Dofs_Per_Sector> void Sector<Dofs_Per_Sector>::setNActiveCells( unsigned int in_nactivecells) {
	NActiveCells = in_nactivecells;
}

template class Sector<2>;
template class Sector<3>;

#endif
