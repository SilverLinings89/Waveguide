#ifndef WaveguideStructureFlag
#define WaveguideStructureFlag

#include <math.h>
#include <vector>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include "Parameters.h"
#include "Sector.h"

using namespace dealii;

class WaveguideStructure {
	public:
		std::vector<Sector> case_sectors;
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
		void 	run() ;
		void 	estimate_and_initialize();
		double 	m(double);
		double 	v(double);
		double 	getQ1 (Point<3> &);
		double 	getQ2 (Point<3> &);
		double 	get_dof (int );
		void	set_dof (int , double );

};

#endif
