#include "Sector.h"
#include <deal.II/base/tensor.h>
#include "staticfunctions.cpp"
using namespace dealii;

Sector::Sector(bool in_left, bool in_right, double in_z_0, double in_z_1):left(in_left), right(in_right), boundary(in_left && in_right), z_0(in_z_0), z_1(in_z_1) {
	r_0 = 0;
	r_1 = 0;
	v_0 = 0;
	v_1 = 0;
	m_0 = 0;
	m_1 = 0;
}

void Sector::set_properties(double in_m_0, double in_m_1, double in_r_0, double in_r_1, double in_v_0, double in_v_1){
	if(!left) {
		r_0 = in_r_0;
		m_0 = in_m_0;
		v_0 = in_v_0;
	}
	if(!right) {
		r_1 = in_r_1;
		m_1 = in_m_1;
		v_1 = in_v_1;
	}
}

void Sector::set_properties_force(double in_m_0, double in_m_1, double in_r_0, double in_r_1, double in_v_0, double in_v_1){
	r_0 = in_r_0;
	m_0 = in_m_0;
	v_0 = in_v_0;
	r_1 = in_r_1;
	m_1 = in_m_1;
	v_1 = in_v_1;

}

double Sector::get_r(double z) {
	if ( z < 0.0 ) z = 0.0;
	if ( z > 1.0 ) z = 1.0;
	return InterpolationPolynomialZeroDerivative(z, r_0, r_1);
}

double Sector::get_m(double z) {
	if ( z < 0.0 ) z = 0.0;
	if ( z > 1.0 ) z = 1.0;
	return InterpolationPolynomial(z, m_0, m_1, v_0, v_1);
}

double Sector::get_v(double z) {
	if ( z < 0.0 ) z = 0.0;
	if ( z > 1.0 ) z = 1.0;
	return InterpolationPolynomialZeroDerivative(z, v_0, v_1);
}

double Sector::getQ1 ( double z) {
	return 1/(r_0 + z*z*z*(2*r_0 - 2*r_1) - z*z*(3*r_0 - 3*r_1));
}

double Sector::getQ2 ( double z) {
	return 1/(r_0 + z*z*z*(2*r_0 - 2*r_1) - z*z*(3*r_0 - 3*r_1));
}

Tensor<2,3, double> Sector::TransformationTensorInternal (double in_x, double in_y, double z) {
	if(z<0 || z>1) std::cout << "Falty implementation of internal Tensor calculation: z: " << z << std::endl;
	double RadiusInMultiplyer = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/(2* r_0);
	double RadiusOutMultiplyer = (GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut)/(2* r_1);

	double temp = 1 / (RadiusInMultiplyer - 3*RadiusInMultiplyer*z*z + 2*RadiusInMultiplyer*z*z*z + 3*RadiusOutMultiplyer*z*z - 2*RadiusOutMultiplyer*z*z*z);
	double zz = z*z;
	double zzz = zz*z;
	double zzzz = zz*zz;
	double zzzzz = zzz*zz;
	double help1 = (RadiusInMultiplyer - 3*RadiusInMultiplyer*zz + 2*RadiusInMultiplyer*zzz + 3*RadiusOutMultiplyer*zz - 2*RadiusOutMultiplyer*zzz);
	double help2 = (help1*(v_0 - 6*m_0*z + 6*m_1*z - 4*v_0*z - 2*v_1*z + 6*m_0*zz - 6*m_1*zz + 3*v_0*zz + 3*v_1*zz) + 6*z*(RadiusInMultiplyer - RadiusOutMultiplyer)*(z - 1)*(m_0 + zzz*(2*m_0 - 2*m_1 + v_0 + v_1) + v_0*z - zz*(3*m_0 - 3*m_1 + 2*v_0 + v_1)))/help1 - (6*in_y*z*(RadiusInMultiplyer - RadiusOutMultiplyer)*(z - 1))/help1;

	Tensor<2,3, double> u;
	u[0][0]= temp;
	u[0][1]= 0.0;
	u[0][2]= 0.0;
	u[1][0]= 0.0;
	u[1][1]= temp;
	u[1][2]= 0.0;
	u[2][0]= -(6*in_x*z*(RadiusInMultiplyer-RadiusOutMultiplyer)*(z - 1))/help1;
	u[2][1]= (RadiusInMultiplyer*v_0 + 12*m_0*RadiusInMultiplyer*zz + 36*m_0*RadiusInMultiplyer*zzz - 6*m_0*RadiusOutMultiplyer*zz - 6*m_1*RadiusInMultiplyer*zz - 60*m_0*RadiusInMultiplyer*zzzz - 36*m_0*RadiusOutMultiplyer*zzz - 36*m_1*RadiusInMultiplyer*zzz + 24*m_0*RadiusInMultiplyer*zzzzz + 60*m_0*RadiusOutMultiplyer*zzzz + 60*m_1*RadiusInMultiplyer*zzzz + 36*m_1*RadiusOutMultiplyer*zzz - 24*m_0*RadiusOutMultiplyer*zzzzz - 24*m_1*RadiusInMultiplyer*zzzzz - 60*m_1*RadiusOutMultiplyer*zzzz + 24*m_1*RadiusOutMultiplyer*zzzzz - 6*RadiusInMultiplyer*v_0*zz + 32*RadiusInMultiplyer*v_0*zzz + 3*RadiusInMultiplyer*v_1*zz + 9*RadiusOutMultiplyer*v_0*zz - 35*RadiusInMultiplyer*v_0*zzzz + 12*RadiusInMultiplyer*v_1*zzz - 32*RadiusOutMultiplyer*v_0*zzz + 12*RadiusInMultiplyer*v_0*zzzzz - 25*RadiusInMultiplyer*v_1*zzzz + 35*RadiusOutMultiplyer*v_0*zzzz - 12*RadiusOutMultiplyer*v_1*zzz + 12*RadiusInMultiplyer*v_1*zzzzz - 12*RadiusOutMultiplyer*v_0*zzzzz + 25*RadiusOutMultiplyer*v_1*zzzz - 12*RadiusOutMultiplyer*v_1*zzzzz - 6*RadiusInMultiplyer*in_y*zz + 6*RadiusOutMultiplyer*in_y*zz - 12*m_0*RadiusInMultiplyer*z + 6*m_0*RadiusOutMultiplyer*z + 6*m_1*RadiusInMultiplyer*z - 4*RadiusInMultiplyer*v_0*z - 2*RadiusInMultiplyer*v_1*z + 6*RadiusInMultiplyer*in_y*z - 6*RadiusOutMultiplyer*in_y*z)/help1;
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

	double det = ginv[0][0]*( ginv[2][2]*ginv[1][1] - ginv[2][1]*ginv[1][2]) - ginv[1][0]*(ginv[2][2]*ginv[0][1] - ginv[2][1]*ginv[0][2]) + ginv[2][0]*(ginv[1][2]*ginv[0][1] - ginv[1][1]*ginv[0][2]);

	Tensor<2,3,double> g;
	g[0][0] = (ginv[2][2] * ginv[1][1] - ginv[2][1]*ginv[1][2]);
	g[0][1] = - (ginv[2][2] * ginv[0][1] - ginv[2][1]*ginv[0][2]);
	g[0][2] = (ginv[1][2] * ginv[0][1] - ginv[1][1]*ginv[0][2]);
	g[1][0] = - (ginv[2][2] * ginv[1][0] - ginv[2][0]*ginv[1][2]);
	g[1][1] = (ginv[2][2] * ginv[0][0] - ginv[2][0]*ginv[0][2]);
	g[1][2] = - (ginv[1][2] * ginv[0][0] - ginv[1][0]*ginv[0][2]);
	g[2][0] = (ginv[2][1] * ginv[1][0] - ginv[2][0]*ginv[1][1]);
	g[2][1] =  -(ginv[2][1] * ginv[0][0] - ginv[2][0]*ginv[0][1]);
	g[2][2] = (ginv[1][1] * ginv[0][0] - ginv[1][0]*ginv[0][1]);

	g *= 1/det;
	double sp = dotproduct(u[0], crossproduct(u[1], u[2]));
	if(sp < 0) sp *= -1.0;
	for(int i = 0; i< 3; i++) {
		for(int j = 0; j<3; j++) {
			g[i][j] *= sp * Q[0]*Q[1]*Q[2] / (Q[i] * Q[j]);
		}
	}



	return g;
}

unsigned int Sector::getLowestDof() {
	return LowestDof;
}

unsigned int Sector::getNDofs() {
	return NDofs;
}

unsigned int Sector::getNInternalBoundaryDofs() {
	return NInternalBoundaryDofs;
}

unsigned int Sector::getNActiveCells() {
	return NActiveCells;
}

void Sector::setLowestDof( unsigned int in_lowestdof) {
	LowestDof = in_lowestdof;
}

void Sector::setNDofs( unsigned int in_ndofs) {
	NDofs = in_ndofs;
}

void Sector::setNInternalBoundaryDofs( unsigned int in_ninternalboundarydofs) {
	NInternalBoundaryDofs = in_ninternalboundarydofs;
}

void Sector::setNActiveCells( unsigned int in_nactivecells) {
	NActiveCells = in_nactivecells;
}
