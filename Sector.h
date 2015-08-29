#ifndef SectorFlag
#define SectorFlag

#include <deal.II/base/tensor.h>

using namespace dealii;

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

#endif
