#ifndef WaveguideStructureCppFlag
#define WaveguideStructureCppFlag

#include "WaveguideStructure.h"
#include <math.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

using namespace dealii;

void WaveguideStructure::estimate_and_initialize() {
	if(sectors == 1) {
		Sector temp12(true, true, -parameters.PRM_M_R_ZLength/2, parameters.PRM_M_R_ZLength/2 );
		case_sectors.reserve(1);
		case_sectors.push_back(temp12);
		case_sectors[0].set_properties_force(m_0, m_1, r_0, r_1, v_0, v_1);
	} else {
		case_sectors.reserve(sectors);
		double length = parameters.PRM_M_R_ZLength / (1.0 * sectors);
		Sector temp(true, false, -parameters.PRM_M_R_ZLength/2, -parameters.PRM_M_R_ZLength/2 + length );
		case_sectors.push_back(temp);
		for(int  i = 1; i < sectors -1; i++) {
			Sector temp2( false, false, -parameters.PRM_M_R_ZLength/2 + length*(1.0 *i), -parameters.PRM_M_R_ZLength/2 + length*(i + 1.0) );
			case_sectors.push_back(temp2);
		}
		Sector temp3( false, true, parameters.PRM_M_R_ZLength/2 - length, parameters.PRM_M_R_ZLength/2 );
		case_sectors.push_back(temp3);

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



Tensor<2,3, double> WaveguideStructure::TransformationTensor (double in_x, double in_y, double in_z) {
	int idx = (in_z - z_min)/sector_z_length;
	if(sectors <= idx) {
		return case_sectors[idx].TransformationTensorInternal(in_x, in_y, 1.0 );
	} else {
		return case_sectors[idx].TransformationTensorInternal(in_x, in_y, (in_z - z_min -idx*sector_z_length)/sector_z_length );
	}
}

double WaveguideStructure::getQ1 (Point<3> &p) {
	return 1/(r_0 + p[3]*p[3]*p[3]*(2*r_0 - 2*r_1) - p[3]*p[3]*(3*r_0 - 3*r_1));
}

double WaveguideStructure::getQ2 (Point<3> &p) {
	return 1/(r_0 + p[3]*p[3]*p[3]*(2*r_0 - 2*r_1) - p[3]*p[3]*(3*r_0 - 3*r_1));
}

double WaveguideStructure::get_dof (int i) {
	int temp = i % 3;
	int sec = i / 3;
	double val = 0.0;
	if(temp == 0) {
		// request m
		val = case_sectors[sec].m_1;
	}
	if(temp == 1) {
		// request r
		val = case_sectors[sec].r_1;
	}
	if(temp == 2) {
		// request v
		val = case_sectors[sec].v_1;
	}
	return val;
}

void WaveguideStructure::set_dof (int i, double val) {
	int temp = i % 3;
	int sec = i / 3;
	if(temp == 0) {
		// request m
		case_sectors[sec].m_1	= val;
		case_sectors[sec+1].m_0	= val;
	}
	if(temp == 1) {
		// request r
		case_sectors[sec].r_1	= val;
		case_sectors[sec+1].r_0	= val;
	}
	if(temp == 2) {
		// request v
		case_sectors[sec].v_1	= val;
		case_sectors[sec].v_0	= val;
	}
}



WaveguideStructure::WaveguideStructure(Parameters &in_params)
		:
		epsilon_K(in_params.PRM_M_W_EpsilonIn),
		epsilon_M(in_params.PRM_M_W_EpsilonOut),
		sectors(in_params.PRM_M_W_Sectors),
		sector_z_length(in_params.PRM_M_R_ZLength / (sectors*1.0)),
		z_min(in_params.PRM_M_R_ZLength / (-2.0)),
		z_max(in_params.PRM_M_R_ZLength / (2.0)),
		deltaY(in_params.PRM_M_W_Delta),
		r_0(in_params.PRM_M_C_RadiusIn),
		r_1(in_params.PRM_M_C_RadiusOut),
		v_0(in_params.PRM_M_C_TiltIn),
		v_1(in_params.PRM_M_C_TiltOut),
		m_0(in_params.PRM_M_W_Delta/2.0),
		m_1(in_params.PRM_M_W_Delta/(-2.0)),
		parameters(in_params)
{
	//case_sectors = Sector[sectors];

}

#endif
