#ifndef WaveguideStructureCppFlag
#define WaveguideStructureCppFlag

#include "WaveguideStructure.h"
#include <math.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

using namespace dealii;

void WaveguideStructure::estimate_and_initialize() {
	highest = 1.0;
	lowest = 1.0;
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
			Sector temp2( false, false, -parameters.PRM_M_R_ZLength/(2.0) + length*(1.0 *i), -parameters.PRM_M_R_ZLength/(2.0) + length*(i + 1.0) );
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
	double temp_z = in_z;
	if(temp_z < GlobalParams.PRM_M_R_ZLength/(-2.0)) {
		temp_z = GlobalParams.PRM_M_R_ZLength/(-2.0);
	}
	if(temp_z > GlobalParams.PRM_M_R_ZLength/(2.0)) {
		temp_z = GlobalParams.PRM_M_R_ZLength/2.0;
	}
	int idx = (temp_z + GlobalParams.PRM_M_R_ZLength/2.0)/sector_z_length;
	if(idx < 0) {
		Tensor<2,3, double> g = case_sectors[0].TransformationTensorInternal(in_x, in_y, 0.0 );
		double stretch = case_sectors[0].getQ1(0.0);
		if(stretch > highest)highest = stretch;
		if(stretch > lowest)lowest = stretch;
		return g;
	} else {
		if(idx < sectors) {
			Tensor<2,3, double> g = case_sectors[idx].TransformationTensorInternal(in_x, in_y, (in_z + GlobalParams.PRM_M_R_ZLength/2.0 -idx*sector_z_length)/sector_z_length );
			double stretch = case_sectors[idx].getQ1((in_z + GlobalParams.PRM_M_R_ZLength/2.0 -idx*sector_z_length)/sector_z_length);
			if(stretch > highest)highest = stretch;
			if(stretch > lowest)lowest = stretch;
			return g;
		} else {
			Tensor<2,3, double> g = case_sectors[sectors-1].TransformationTensorInternal(in_x, in_y, 1.0 );
			double stretch = case_sectors[sectors -1].getQ1(1.0);
			if(stretch > highest)highest = stretch;
			if(stretch > lowest)lowest = stretch;
			return g;
		}
	}
}

double WaveguideStructure::getQ1 (double z) {
	if(z < GlobalParams.PRM_M_R_ZLength/(-2.0)) {
			z = GlobalParams.PRM_M_R_ZLength/(-2.0);
		}
		if(z > GlobalParams.PRM_M_R_ZLength/(2.0)) {
			z = GlobalParams.PRM_M_R_ZLength/2.0;
		}
		int idx = (z + GlobalParams.PRM_M_R_ZLength/2.0)/sector_z_length;
		if(idx < 0) {
			return case_sectors[0].getQ2(0.0 );
		} else {
			if(idx < sectors) {
				return case_sectors[idx].getQ2((z + GlobalParams.PRM_M_R_ZLength/2.0 -idx*sector_z_length)/sector_z_length );
			} else {
				return case_sectors[sectors-1].getQ2( 1.0 );
			}
	}
}

double WaveguideStructure::getQ2 ( double z) {

	if(z < GlobalParams.PRM_M_R_ZLength/(-2.0)) {
		z = GlobalParams.PRM_M_R_ZLength/(-2.0);
	}
	if(z > GlobalParams.PRM_M_R_ZLength/(2.0)) {
		z = GlobalParams.PRM_M_R_ZLength/2.0;
	}
	int idx = (z + GlobalParams.PRM_M_R_ZLength/2.0)/sector_z_length;
	if(idx < 0) {
		return case_sectors[0].getQ2( 0.0 );
	} else {
		if(idx < sectors) {
			return case_sectors[idx].getQ2((z + GlobalParams.PRM_M_R_ZLength/2.0 -idx*sector_z_length)/sector_z_length );
		} else {
			return case_sectors[sectors-1].getQ2( 1.0 );
		}
	}

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
		double max = (GlobalParams.PRM_M_C_RadiusIn > GlobalParams.PRM_M_C_RadiusOut)? GlobalParams.PRM_M_C_RadiusIn : GlobalParams.PRM_M_C_RadiusOut;
		if(val > max*3/2) {
			val =max;
		}
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
		sector_z_length(in_params.PRM_M_R_ZLength/ (sectors*1.0)),
		z_min(in_params.PRM_M_R_ZLength / (-2.0) - in_params.PRM_M_BC_XYin),
		z_max(in_params.PRM_M_R_ZLength / (2.0) + 2 * in_params.PRM_M_BC_XYout),
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
