#ifndef WaveguideStructureCppFlag
#define WaveguideStructureCppFlag

#include "WaveguideStructure.h"
#include <math.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

using namespace dealii;

WaveguideStructure::~WaveguideStructure() {

}

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
		double length = Sector_Length();
		Sector temp(true, false, -parameters.PRM_M_R_ZLength/(2.0), -parameters.PRM_M_R_ZLength/2.0 + length );
		case_sectors.push_back(temp);
		for(int  i = 1; i < sectors - GlobalParams.PRM_M_BC_XYout-1; i++) {
			Sector temp2( false, false, -parameters.PRM_M_R_ZLength/(2.0) + length*(1.0 *i), -parameters.PRM_M_R_ZLength/(2.0) + length*(i + 1.0) );
			case_sectors.push_back(temp2);
		}
		int t = sectors - GlobalParams.PRM_M_BC_XYout-1;
		Sector temp3( false, true, -parameters.PRM_M_R_ZLength/(2.0) + length*(1.0 *t), -parameters.PRM_M_R_ZLength/(2.0) + length*(t + 1.0) );
		case_sectors.push_back(temp3);
		for(int  i = sectors - GlobalParams.PRM_M_BC_XYout ; i < sectors ; i++) {
			Sector temp4( true, true, -parameters.PRM_M_R_ZLength/(2.0) + length*i, -parameters.PRM_M_R_ZLength/(2.0) + length*(i + 1.0) );
			case_sectors.push_back(temp4);
		}

		double length_rel = 1.0/((double)(sectors-GlobalParams.PRM_M_BC_XYout));
		case_sectors[0].set_properties_force(m_0, InterpolationPolynomialZeroDerivative(length_rel, m_0, m_1), r_0, InterpolationPolynomialZeroDerivative(length_rel, r_0, r_1) , 0, InterpolationPolynomialDerivative(length_rel, m_0, m_1, 0, 0)) ;
		for(int  i = 1; i < sectors -GlobalParams.PRM_M_BC_XYout; i++) {
			double z_l = i*length_rel;
			double z_r = (i+1)*length_rel;
			case_sectors[i].set_properties_force(InterpolationPolynomialZeroDerivative(z_l, m_0, m_1), InterpolationPolynomialZeroDerivative(z_r, m_0, m_1), InterpolationPolynomialZeroDerivative(z_l, r_0, r_1), InterpolationPolynomialZeroDerivative(z_r, r_0, r_1) , InterpolationPolynomialDerivative(z_l, m_0, m_1, 0, 0), InterpolationPolynomialDerivative(z_r, m_0, m_1, 0, 0)) ;
		}
		for(int  i = sectors -GlobalParams.PRM_M_BC_XYout; i < sectors; i++) {
			case_sectors[i].set_properties_force(m_1, m_1, r_1, r_1, 0, 0) ;
		}
	}

	for (unsigned int i = 0;  i < NFreeDofs(); ++ i) {
		InitialDofs[i] = this->get_dof(i, true);
	}

	Print();
}

void WaveguideStructure::Print () {
	if(GlobalParams.MPI_Rank == 0) {
		//std::cout << "Structure for Process " <<GlobalParams.MPI_Rank << std::endl;
		std::cout << "-------------------------------------------" << std::endl;
		std::cout << "Sectors: " << sectors << std::endl;
		for(int i = 0 ; i< sectors; i++) {

			std::cout << "z_0: " << std::setw(13)<< case_sectors[i].z_0 << "\t z_1: "<< std::setw(13)<< case_sectors[i].z_1 << std::endl;
			std::cout << "m_0: " << std::setw(13)<< case_sectors[i].m_0 << "\t m_1: "<< std::setw(13)<< case_sectors[i].m_1 << std::endl;
			std::cout << "r_0: " << std::setw(13)<< case_sectors[i].r_0 << "\t r_1: "<< std::setw(13)<< case_sectors[i].r_1 << std::endl;
			std::cout << "v_0: " << std::setw(13)<< case_sectors[i].v_0 << "\t v_1: "<< std::setw(13)<< case_sectors[i].v_1 << std::endl;
			std::cout << "-------------------------------------------" << std::endl;
		}
	}
}

dealii::Vector<double> WaveguideStructure::Dofs() {
	dealii::Vector<double> ret;
	ret.reinit(this->NFreeDofs());
	for (unsigned int i = 0;  i < NFreeDofs(); ++ i) {
		ret[i] = this->get_dof(i, true);
	}
	return ret;
}

double WaveguideStructure::estimate_m(double z) {
	return (v_0 + v_1 + 2*m_0 - 2*m_1)*z*z*z + (-2.0 * v_0 - v_1 - 3*m_0 + 3*m_1)*z*z + v_0*z + m_0;
}

double WaveguideStructure::estimate_v(double z) {
	return 3*(v_0 + v_1 + 2*m_0 - 2*m_1)*z*z + 2*(-2.0 * v_0 - v_1 - 3*m_0 + 3*m_1)*z + v_0;
}

Tensor<2,3, double> WaveguideStructure::TransformationTensor (double in_x, double in_y, double in_z) {
	std::pair<int, double> temp = Z_to_Sector_and_local_z(in_z);

	double stretch = case_sectors[temp.first].getQ1(temp.second);
	if(stretch > highest)highest = stretch;
	if(stretch < lowest)lowest = stretch;

	return case_sectors[temp.first].TransformationTensorInternal(in_x, in_y, temp.second );

}

double WaveguideStructure::getQ1 (double z) {
	std::pair<int, double> temp = Z_to_Sector_and_local_z(z);
	return case_sectors[temp.first].getQ1(temp.second);
}

double WaveguideStructure::getQ2 ( double z) {
	std::pair<int, double> temp = Z_to_Sector_and_local_z(z);
	return case_sectors[temp.first].getQ2(temp.second);
}

double WaveguideStructure::get_dof (int i, bool free) {

	int temp = i % 3;
	int sec = i / 3;
	if(free) {
		if(temp == 0) {
			// request m
			return case_sectors[sec].m_1;
		}
		if(temp == 1) {
			// request r
			return case_sectors[sec].r_1;
		}
		if(temp == 2) {
			// request v
			return case_sectors[sec].v_1;
		}
	} else {
		if (i < 3) {
			if(temp == 0) {
				// request m
				return case_sectors[0].m_0;
			}
			if(temp == 1) {
				// request r
				return case_sectors[0].r_0;
			}
			if(temp == 2) {
				// request v
				return case_sectors[0].v_0;
			}
		} else {
			if(temp == 0) {
				// request m
				return case_sectors[sec-1].m_1;
			}
			if(temp == 1) {
				// request r
				return case_sectors[sec-1].r_1;
			}
			if(temp == 2) {
				// request v
				return case_sectors[sec-1].v_1;
			}
		}
	}
	return 0.0;
}

void WaveguideStructure::set_dof (int i, double val, bool free) {
	int temp = i % 3;
	int sec = i / 3 ;
	if(free) {

		if(temp == 0) {
			case_sectors[sec  ].m_1	= val;
			case_sectors[sec+1].m_0	= val;
		}
		if(temp == 1) {
			case_sectors[sec  ].r_1	= val;
			case_sectors[sec+1].r_0	= val;
		}
		if(temp == 2) {
			case_sectors[sec  ].v_1	= val;
			case_sectors[sec+1].v_0	= val;
		}

	} else {
		if (i < 3) {
			if(temp == 0) {
				case_sectors[0].m_0	= val;
			}
			if(temp == 1) {
				case_sectors[0].r_0	= val;
			}
			if(temp == 2) {
				case_sectors[0].v_0	= val;
			}
		} else {
			if(i >= 3*GlobalParams.PRM_M_W_Sectors) {
				if(temp == 0) {
					case_sectors[case_sectors.size() -1 ].m_1	= val;
				}
				if(temp == 1) {
					case_sectors[case_sectors.size() -1 ].r_1	= val;
				}
				if(temp == 2) {
					case_sectors[case_sectors.size() -1 ].v_1	= val;
				}
			} else {
				if(temp == 0) {
					case_sectors[sec-1].m_1	= val;
					case_sectors[sec].m_0	= val;
				}
				if(temp == 1) {
					case_sectors[sec-1].r_1	= val;
					case_sectors[sec].r_0	= val;
				}
				if(temp == 2) {
					case_sectors[sec-1].v_1	= val;
					case_sectors[sec].v_0	= val;
				}
			}
		}
	}
}

WaveguideStructure::WaveguideStructure(const Parameters &in_params)
		:
		epsilon_K(in_params.PRM_M_W_EpsilonIn),
		epsilon_M(in_params.PRM_M_W_EpsilonOut),
		sectors(in_params.PRM_M_W_Sectors),
		deltaY(in_params.PRM_M_W_Delta),
		r_0(in_params.PRM_M_C_RadiusIn),
		r_1(in_params.PRM_M_C_RadiusOut),
		v_0(in_params.PRM_M_C_TiltIn),
		v_1(in_params.PRM_M_C_TiltOut),
		m_0(in_params.PRM_M_W_Delta/2.0),
		m_1(in_params.PRM_M_W_Delta/(-2.0)),
		parameters(in_params),
		InitialQuality(0.0)
{
	//case_sectors = Sector[sectors];
	InitialDofs.reinit(3*(sectors-GlobalParams.PRM_M_BC_XYout) -3);
}

double WaveguideStructure::Sector_Length() {
	return GlobalParams.PRM_M_R_ZLength / (GlobalParams.PRM_M_W_Sectors - GlobalParams.PRM_M_BC_XYout);
}

double WaveguideStructure::Layer_Length() {
	return Sector_Length() * ((double)GlobalParams.PRM_M_W_Sectors) / ((double) GlobalParams.MPI_Size);
}

double WaveguideStructure::System_Length() {
	return GlobalParams.PRM_M_R_ZLength + GlobalParams.PRM_M_BC_XYout * Sector_Length();
}

std::pair<int, double> WaveguideStructure::Z_to_Sector_and_local_z(double in_z) {
	double sector_length = Sector_Length();
	double total_length = System_Length();
	std::pair<int, double> ret;

	ret.first = std::floor((in_z + GlobalParams.PRM_M_R_ZLength/2.0 ) / sector_length);
	ret.second = (in_z + GlobalParams.PRM_M_R_ZLength/2.0  - (ret.first * sector_length)) / sector_length;

	if(ret.first == -1) {
		ret.first = 0;
		ret.second = 0.0;

		if(abs(( GlobalParams.PRM_M_R_ZLength / 2.0 ) - in_z ) > 0.0001) {
			deallog << "Internal Bug in coding. See WaveguideStructure.cpp - Z_to_Sector_and_local_z (Case 1)." << std::endl;
		}
	}

	if(ret.first >= sectors) {
		ret.first = sectors - 1;
		ret.second = 1.0;
		if(abs(( GlobalParams.PRM_M_R_ZLength / 2.0 ) + in_z - total_length ) > 0.0001) {
			deallog << "Internal Bug in coding. See WaveguideStructure.cpp - Z_to_Sector_and_local_z (Case 2)." << std::endl;
		}
	}

	return ret;
}
int WaveguideStructure::Z_to_Layer(double in_z) {
	double sector_length = Layer_Length();
	int ret;

	ret = std::floor((in_z + GlobalParams.PRM_M_R_ZLength/2.0 ) / sector_length);
	if(ret == -1) {
		ret = 0;
	}

	if(ret >= GlobalParams.MPI_Size) {
		ret = GlobalParams.MPI_Size - 1;
	}

	return ret;
}

double WaveguideStructure::get_r(double in_z) {
	std::pair<int, double> temp = Z_to_Sector_and_local_z(in_z);
	return case_sectors[temp.first].get_r(temp.second);
}

double WaveguideStructure::get_m(double in_z) {
	std::pair<int, double> temp = Z_to_Sector_and_local_z(in_z);
	return case_sectors[temp.first].get_m(temp.second);
}

double WaveguideStructure::get_v(double in_z) {
	std::pair<int, double> temp = Z_to_Sector_and_local_z(in_z);
	return case_sectors[temp.first].get_v(temp.second);
}

void WaveguideStructure::WriteConfigurationToConsole() {

}

unsigned int WaveguideStructure::NFreeDofs() {
	return 3*(sectors-GlobalParams.PRM_M_BC_XYout) -3;
}

unsigned int WaveguideStructure::NDofs() {
	return 3*sectors + 3;
}

bool WaveguideStructure::IsDofFree(int in_dof) {
	bool ret = true;
	if(in_dof < 3) ret = false;
	if(in_dof >= 3*(sectors-GlobalParams.PRM_M_BC_XYout) ) ret = false;
	return ret;
}
#endif
