#include "Optimization.h"
#include "Waveguide.h"

using namespace dealii;

Optimization::Optimization( Parameters in_System_Parameters ,Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> >  &in_wg, WaveguideStructure &in_structure)
	:
		dofs((in_System_Parameters.PRM_M_W_Sectors +1)*3 -6),
		System_Parameters(in_System_Parameters),
		waveguide(in_wg),
		structure(in_structure)

	{
}

void Optimization::run() {
	double gradient [dofs];
	double step_width = 1.0;
	bool abort = false;
	int steps_counter = 0;
	structure.estimate_and_initialize();
	waveguide.run();
	steps_counter ++;
	waveguide.store();
	double quality = waveguide.evaluate_overall();
	std::cout << "Initial Passthrough-quality: " << 100*quality << "%." << std::endl;
	std::cout << "Calculating Gradients ..." << std::endl;
	while (steps_counter < System_Parameters.PRM_Op_MaxCases)
	{
		std::cout << "Current configuration: "<< std::endl;
		std::cout << "Radius: "<< std::endl;
		for(int i = 1;  i +=3; i<dofs) {
			std::cout << "\t r_" << 1+ i/3 << ": " << structure.get_dof(i) << std::endl;
		}
		std::cout << "Waveguide Center: "<< std::endl;
		for(int i = 0;  i +=3; i<dofs) {
			std::cout << "\t m_" <<1+ i/3 << ": " << structure.get_dof(i) << std::endl;
		}
		std::cout << "Waveguide Angle: "<< std::endl;
		for(int i = 2;  i +=3; i<dofs) {
			std::cout << "\t r_" <<1+ i/3 << ": " << structure.get_dof(i) << std::endl;
		}
		double norm = 0.0;
		for(int i = 0; i < dofs; i++) {
			std::cout << "\t Gradient "<< i << ": ..."<< std::endl;
			double val = structure.get_dof(i);
			structure.set_dof(i, val + step_width/10.0);
			waveguide.run();
			std::cout << "Starting Waveguide-calculation..." << std::endl;
			steps_counter ++;
			double temp_quality = waveguide.evaluate_overall();
			std::cout << "Quality after adjusting position (calculating gradient): " << temp_quality << std::endl;
			gradient[i] = quality - temp_quality;
			norm += gradient[i] * gradient[i];
			structure.set_dof(i, val - step_width/10.0);
		}
		std::cout << "Gradient calculation done." << std::endl;
		std::cout << "Gradient for dofs: (";
		for(int i = 0; i < dofs; i++) {
			std::cout << gradient[i];
			if( i < dofs - 1)std::cout << ",";
		}
		std::cout << ")" << std::endl;
		norm = sqrt(norm);
		std::cout << "Setting step to (";
		for(int i = 0; i < dofs; i++) {
			double step = (-1.0) * gradient[i] / norm;
			double val = structure.get_dof(i);
			std::cout << val + step ;
			if( i < dofs - 1)std::cout << ",";
			structure.set_dof(i, val + step);
		}
		std::cout << ")" << std::endl;

		std::cout << "Calculation solution after step ..." << std::endl;
		waveguide.run();
		steps_counter ++;
		double step_quality = waveguide.evaluate_overall();
		std::cout << "Quality after the step: " << step_quality ;
		if(step_quality < quality) {
			std::cout << "... not accepted (reduced quality). Undoing step and reducing step-width." << std::endl;
			std::cout << "New Step-width: " << step_width *0.1 << std::endl;
			for(int i = 0; i < dofs; i++) {
				double step = (-1.0) * gradient[i] / norm;
				double val = structure.get_dof(i);
				structure.set_dof(i, val - step);
				step_width *= 0.1;
			}
		} else {
			std::cout << "... accepted. Updating current quality." << std::endl;
			quality = step_quality;
		}

	}
}
