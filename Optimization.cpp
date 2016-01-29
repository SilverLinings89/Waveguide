#include "Optimization.h"
#include "Waveguide.h"

using namespace dealii;

template<typename Matrix,typename Vector>
Optimization<Matrix, Vector >::Optimization( Parameters in_System_Parameters ,Waveguide<Matrix, Vector >  &in_wg)
	:
		dofs((in_System_Parameters.PRM_M_W_Sectors +1)*3 -6),
		System_Parameters(in_System_Parameters),
		waveguide(in_wg)

	{
}

template<typename Matrix,typename Vector>
void Optimization<Matrix, Vector>::run() {
	std::vector<double> gradient(dofs);
	double step_width = GlobalParams.PRM_Op_InitialStepWidth;
	int steps_counter = 0;
	structure.estimate_and_initialize();
	int best = 0;
	waveguide.run();
	steps_counter ++;
	waveguide.store();
	double quality = waveguide.evaluate_overall();
	std::vector<double> old_position(dofs);
	if(GlobalParams.PRM_S_DoOptimization) {
		std::cout << "Initial Passthrough-quality: " << 100*quality << "%." << std::endl;
		std::cout << "Calculating Gradients ..." << std::endl;
		while (steps_counter < System_Parameters.PRM_Op_MaxCases && step_width > 0.00001)
		{
			std::cout << "Current configuration: ";
			std::cout << "Waveguide Center: ";
			for(int i = 0;  i<dofs; i +=3) {
				std::cout << " m_" <<1+ i/3 << ": " << structure.get_dof(i);
			}
			std::cout << "; Radius: ";
			for(int i = 1;  i<dofs; i +=3) {
				std::cout << " r_" << 1+ i/3 << ": " << structure.get_dof(i);
			}
			std::cout << "; Waveguide Angle: ";
			for(int i = 2; i<dofs; i +=3) {
				std::cout << " v_" <<1+ i/3 << ": " << structure.get_dof(i);
			}
			double norm = 0.0;
			std::cout << std::endl;
			for(int i = 0; i < dofs; i++) {
				std::cout << "Gradient "<< i << ": ...";
				double val = structure.get_dof(i);
				double new_val = val + step_width/10.0;
				if(i%3 == 0){
					if(new_val < GlobalParams.PRM_O_MinimumFactor * std::min(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut)) {
						new_val = GlobalParams.PRM_O_MinimumFactor * std::min(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut);
					}
					if(new_val > GlobalParams.PRM_O_MaximumFactor * std::max(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut)) {
						new_val = GlobalParams.PRM_O_MaximumFactor *  std::max(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut);
					}
				}
				if(i%3 == 1){
					if(new_val < - GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0) {
						new_val = - GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0;
					}
					if(new_val >  GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0) {
						new_val = - GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0;
					}
				}

				structure.set_dof(i, new_val);
				waveguide.rerun();
				steps_counter ++;
				double temp_quality = waveguide.evaluate_overall();
				std::cout << "Quality after adjusting position (calculating gradient): " << temp_quality << std::endl;
				gradient[i] = quality - temp_quality;
				norm += gradient[i] * gradient[i];
				structure.set_dof(i, val);
			}
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
				step *= step_width;
				old_position[i] = structure.get_dof(i);
				double val = structure.get_dof(i) + step;
				if(i%3 == 0){
					if(val < GlobalParams.PRM_O_MinimumFactor * std::min(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut)) {
						val = GlobalParams.PRM_O_MinimumFactor * std::min(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut);
					}
					if(val > GlobalParams.PRM_O_MaximumFactor * std::max(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut)) {
						val = GlobalParams.PRM_O_MaximumFactor *  std::max(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut);
					}
				}
				if(i%3 == 1){
					if(val < - GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0) {
						val = - GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0;
					}
					if(val >  GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0) {
						val = - GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0;
					}
				}
				std::cout << val;
				if( i < dofs - 1)std::cout << ",";
				structure.set_dof(i, val);
			}
			std::cout << ")" << std::endl;

			std::cout << "Calculating solution after step... " ;
			waveguide.rerun();
			steps_counter ++;
			double step_quality = waveguide.evaluate_overall();
			std::cout << "Quality after the step: " << step_quality ;
			if(step_quality < quality) {
				std::cout << "... not accepted (reduced quality). Undoing step and reducing step-width.";
				std::cout << "New Step-width: " << step_width *0.1 << std::endl;
				for(int i = 0; i < dofs; i++) {
					structure.set_dof(i, old_position[i]);
				}
				step_width *= 0.1;
			} else {
				std::cout << "... accepted. Updating current quality." << std::endl;
				best = steps_counter;
				quality = step_quality;
				waveguide.store();
			}

		}

		std::cout << "The best configuration was achieved in step number "<< best<<". The configuration is: ";
		std::cout << "Radius: ";
		for(int i = 0;  i<dofs; i +=3) {
			std::cout << " m_" <<1+ i/3 << ": " << structure.get_dof(i);
		}
		std::cout << "; \t Waveguide Center: ";
		for(int i = 1;  i<dofs; i +=3) {
			std::cout << " r_" << 1+ i/3 << ": " << structure.get_dof(i);
		}
		std::cout << "; Waveguide Angle: ";
		for(int i = 2; i<dofs; i +=3) {
			std::cout << " v_" <<1+ i/3 << ": " << structure.get_dof(i);
		}
	} else {
		std::cout << "Only one single calculation was done. The result has been saved. If you wish to optimize the shape, set the according parameter in the input-file." << std::endl;
	}
}
