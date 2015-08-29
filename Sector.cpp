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

Tensor<2,3, double> Sector::TransformationTensorInternal (double in_x, double in_y, double z) {
	double temp = 1 / (r_0 - 3*r_0*z*z + 2*r_0*z*z*z + 3*r_1*z*z - 2*r_1*z*z*z);
	double zz = z*z;
	double zzz = zz*z;
	double zzzz = zz*zz;
	double zzzzz = zzz*zz;
	double zzzzzz = zzz*zzz;
	double help1 = (r_0 - 3*r_0*zz + 2*r_0*zzz + 3*r_1*zz - 2*r_1*zzz);
	double help2 = (help1*(v_0 - 6*m_0*z + 6*m_1*z - 4*v_0*z - 2*v_1*z + 6*m_0*zz - 6*m_1*zz + 3*v_0*zz + 3*v_1*zz) + 6*z*(r_0 - r_1)*(z - 1)*(m_0 + zzz*(2*m_0 - 2*m_1 + v_0 + v_1) + v_0*z - zz*(3*m_0 - 3*m_1 + 2*v_0 + v_1)))/help1 - (6*in_y*z*(r_0 - r_1)*(z - 1))/help1;
	Tensor<2,3, double> u;
	u[0][0]= temp;
	u[0][1]= 0.0;
	u[0][2]= 0.0;
	u[1][0]= 0.0;
	u[1][1]= temp;
	u[1][2]= 0.0;
	u[2][0]= -(6*in_x*z*(r_0-r_1)*(z - 1))/help1, 0.0;
	u[2][1]= (r_0*v_0 + 12*m_0*r_0*zz + 36*m_0*r_0*zzz - 6*m_0*r_1*zz - 6*m_1*r_0*zz - 60*m_0*r_0*zzzz - 36*m_0*r_1*zzz - 36*m_1*r_0*zzz + 24*m_0*r_0*zzzzz + 60*m_0*r_1*zzzz + 60*m_1*r_0*zzzz + 36*m_1*r_1*zzz - 24*m_0*r_1*zzzzz - 24*m_1*r_0*zzzzz - 60*m_1*r_1*zzzz + 24*m_1*r_1*zzzzz - 6*r_0*v_0*zz + 32*r_0*v_0*zzz + 3*r_0*v_1*zz + 9*r_1*v_0*zz - 35*r_0*v_0*zzzz + 12*r_0*v_1*zzz - 32*r_1*v_0*zzz + 12*r_0*v_0*zzzzz - 25*r_0*v_1*zzzz + 35*r_1*v_0*zzzz - 12*r_1*v_1*zzz + 12*r_0*v_1*zzzzz - 12*r_1*v_0*zzzzz + 25*r_1*v_1*zzzz - 12*r_1*v_1*zzzzz - 6*r_0*in_y*zz + 6*r_1*in_y*zz - 12*m_0*r_0*z + 6*m_0*r_1*z + 6*m_1*r_0*z - 4*r_0*v_0*z - 2*r_0*v_1*z + 6*r_0*in_y*z - 6*r_1*in_y*z)/help1;
	u[2][2]= 1.0;
	double Q [3];

	Q[0] = 1/(r_0 + zzz*(2*r_0 - 2*r_1) - zz*(3*r_0 - 3*r_1));
	Q[1] = 1/(r_0 + zzz*(2*r_0 - 2*r_1) - zz*(3*r_0 - 3*r_1));
	if(Q[1] < 0) Q[1] *= -1.0;
	if(Q[0] < 0) Q[0] *= -1.0;
	Q[2] = sqrt( help2*help2 + (36*in_x*in_x*zz*(r_0 - r_1)*(r_0 - r_1)*(z - 1)*(z - 1))/(help1*help1) + 1);

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

	double sp = dotproduct(u[0], crossproduct(u[1], u[2]));
	if(sp < 0) sp *= -1.0;
	for(int i = 0; i< 3; i++) {
		for(int j = 0; j<3; j++) {
			g[i][j] *= sp * Q[0]*Q[1]*Q[2] / (Q[i] * Q[j]);
		}
	}
	return g;
}
