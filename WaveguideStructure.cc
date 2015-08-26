#include <deal.II/base/tensor.h>
#include <math.h>
#include "ParameterStruct.cc"

class Sector {
	public:
		const bool left;
		const bool right;
		const bool boundary;
		double r_0, r_1, v_0, v_1, m_0, m_1;
		const double z_0, z_1;
		Tensor<2,3, double> TransformationTensorInternal (double in_x, double in_y, double in_z);
		Sector(bool, bool, double, double);
		void set_properties(double , double , double , double, double, double);
		void set_properties_force(double , double , double , double, double, double);
};

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

class WaveguideStructure {
	public:
		Sector case_sectors [];
		const double epsilon_K, epsilon_M;
		const int sectors;
		const double sector_z_length;
		const double z_min, z_max;
		const double deltaY;
		const double r_0, r_1;
		const double v_0, v_1;
		const double m_0, m_1;
		const Parameters parameters;
		
		WaveguideStructure (Parameters &);
		Tensor<2,3, double> TransformationTensor (double in_x, double in_y, double in_z);
		void run() ;
		void estimate_and_initialize();
		double m(double);
		double v(double);
		
};

void WaveguideStructure::estimate_and_initialize() {
	if(sectors == 1) {
		case_sectors[0] = new Sector(true, true, -parameters.PRM_M_R_ZLength/2, parameters.PRM_M_R_ZLength/2 );
		case_sectors[0].set_properties_force(m_0, m_1, r_0, r_1, v_0, v_1);
	} else {
		double length = parameters.PRM_M_R_ZLength / (1.0 * sectors);
		case_sectors[0] = new Sector(true, false, -parameters.PRM_M_R_ZLength/2, -parameters.PRM_M_R_ZLength/2 + length );
		for(int  i = 1; i < sectors -1; i++) {
			case_sectors[i] = new Sector( false, false, -parameters.PRM_M_R_ZLength/2 + length*(1.0 *i), -parameters.PRM_M_R_ZLength/2 + length*(i + 1.0) );
		}
		case_sectors[sectors -1] = new Sector( false, true, parameters.PRM_M_R_ZLength/2 - length, parameters.PRM_M_R_ZLength/2 );

		double length_rel = 1.0/sectors;
		case_sectors[0].set_properties_force(m_0, m(length_rel), r_0, r_0 + (r_1 - r_0) /(sectors*1.0) , v_0, v(length_rel)) ;
		for(int  i = 1; i < sectors -1; i++) {
			case_sectors[i].set_properties_force(m(i*length_rel), m((i+1)*length_rel), r_0 + i*(r_1 - r_0) /(sectors*1.0), r_0 + (i+1)*(r_1 - r_0) /(sectors*1.0) , v(i*length_rel), v((i+1.0)*length_rel)) ;
		}
		case_sectors[sectors -1].set_properties_force(m(1.0 - length_rel), m_1, r_1 - (r_1 - r_0) /(sectors*1.0), r_1  , v(1.0- length_rel), v_1) ;
	}

}

double WaveguideStructure::m(double z) {
	return (v_0 + v_1 + 2*m_0 - 2*m_1)*z*z*z + (-2.0 * v_0 - v_1 - 3*m_0 + 3*m_1)*z*z + v_0*z + m_0;
}

double WaveguideStructure::v(double z) {
	return 3*(v_0 + v_1 + 2*m_0 - 2*m_1)*z*z + 2*(-2.0 * v_0 - v_1 - 3*m_0 + 3*m_1)*z + v_0;
}

inline Tensor<1, 3 , double> crossproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b) {
	Tensor<1,3,double> ret;
	ret[0] = a[1] * b[2] - a[2] * b[1];
	ret[1] = a[2] * b[0] - a[0] * b[2];
	ret[2] = a[0] * b[1] - a[1] * b[0];
	return ret;
}

inline double dotproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Tensor<2,3, double> WaveguideStructure::TransformationTensor (double in_x, double in_y, double in_z) {
	int idx = (in_z - z_min)/sector_z_length;
	if(sectors <= idx) {
		return case_sectors[idx].TransformationTensorInternal(in_x, in_y, 1.0 );
	} else {
		return case_sectors[idx].TransformationTensorInternal(in_x, in_y, (in_z - z_min -idx*sector_z_length)/sector_z_length );
	}
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
	Tensor<2,3,double> u;
	u[0][0] = temp;
	u[0][1] = 0.0;
	u[0][2] = 0.0;
	u[1][0] = 0.0;
	u[1][1] = temp;
	u[1][2] = 0.0;
	u[2][0] = -(6*in_x*z*(r_0-r_1)*(z - 1))/help1;
	u[2][1] = (r_0*v_0 + 12*m_0*r_0*zz + 36*m_0*r_0*zzz - 6*m_0*r_1*zz - 6*m_1*r_0*zz - 60*m_0*r_0*zzzz - 36*m_0*r_1*zzz - 36*m_1*r_0*zzz + 24*m_0*r_0*zzzzz + 60*m_0*r_1*zzzz + 60*m_1*r_0*zzzz + 36*m_1*r_1*zzz - 24*m_0*r_1*zzzzz - 24*m_1*r_0*zzzzz - 60*m_1*r_1*zzzz + 24*m_1*r_1*zzzzz - 6*r_0*v_0*zz + 32*r_0*v_0*zzz + 3*r_0*v_1*zz + 9*r_1*v_0*zz - 35*r_0*v_0*zzzz + 12*r_0*v_1*zzz - 32*r_1*v_0*zzz + 12*r_0*v_0*zzzzz - 25*r_0*v_1*zzzz + 35*r_1*v_0*zzzz - 12*r_1*v_1*zzz + 12*r_0*v_1*zzzzz - 12*r_1*v_0*zzzzz + 25*r_1*v_1*zzzz - 12*r_1*v_1*zzzzz - 6*r_0*in_y*zz + 6*r_1*in_y*zz - 12*m_0*r_0*z + 6*m_0*r_1*z + 6*m_1*r_0*z - 4*r_0*v_0*z - 2*r_0*v_1*z + 6*r_0*in_y*z - 6*r_1*in_y*z)/help1;
	u[2][2] = 1.0;
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

WaveguideStructure::WaveguideStructure(Parameters &in_params)
		:
		epsilon_K(in_params.PRM_M_W_EpsilonIn),
		epsilon_M(in_params.PRM_M_W_EpsilonOut),
		r_0(in_params.PRM_M_C_RadiusIn),
		r_1(in_params.PRM_M_C_RadiusOut),
		deltaY(in_params.PRM_M_W_Delta),
		sectors(in_params.PRM_M_W_Sectors),
		parameters(in_params),
		v_0(in_params.PRM_M_C_TiltIn),
		v_1(in_params.PRM_M_C_TiltOut),
		m_0(in_params.PRM_M_W_Delta/2.0),
		m_1(in_params.PRM_M_W_Delta/(-2.0))
{
	case_sectors = Sector[sectors];

}
