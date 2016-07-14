#include "Optimization.h"
#include "Waveguide.h"
#include "GradientTable.h"

using namespace dealii;

template<typename Matrix,typename Vector>
Optimization<Matrix, Vector >::Optimization( Parameters in_System_Parameters ,Waveguide<Matrix, Vector >  &in_wg)
	:
		dofs(structure->NDofs()),
		freedofs(structure->NFreeDofs()),
		System_Parameters(in_System_Parameters),
		waveguide(in_wg)

	{
}

template<typename Matrix,typename Vector>
void Optimization<Matrix, Vector>::run() {
	structure->estimate_and_initialize();
	double step = 0.00001;
	dealii::Vector<double> r (GlobalParams.MPI_Size);
	dealii::FullMatrix<double> D(GlobalParams.MPI_Size, freedofs);
	dealii::FullMatrix<double> Prod(freedofs, freedofs);
	dealii::FullMatrix<double> Dinv(freedofs, freedofs);
	dealii::Vector<double> rt_1 (freedofs);
	dealii::Vector<double> rt_2 (freedofs);

	dealii::Vector<double> a(freedofs);
	if(!GlobalParams.PRM_S_DoOptimization) {
		waveguide.run();
	} else {
		for (unsigned int i = 0; i < GlobalParams.PRM_Op_MaxCases; i++){
			waveguide.run();
			MPI_Barrier(GlobalParams.MPI_Communicator);
			for ( unsigned int j = 0; j < GlobalParams.MPI_Size; i++ ){
				r[j] = 1 - waveguide.qualities[j];
			}
			for( unsigned int j = 0; j < freedofs; j++) {
				a(j) = structure->get_dof(j,true);
			}
			for (unsigned int j = 0; j < freedofs; j++) {
				double old = structure->get_dof(j,true);
				structure->set_dof(j, old + step, true);
				MPI_Barrier(GlobalParams.MPI_Communicator);
				waveguide.rerun();
				for(unsigned int k = 0; k < GlobalParams.MPI_Size; k++) {
					double res = 1 - waveguide.qualities[k];
					D[k][j]= (res - r[k])/step;
				}
				structure->set_dof(j, old , true);
				MPI_Barrier(GlobalParams.MPI_Communicator);
			}
			if (GlobalParams.MPI_Rank == 0) {
				D.Tvmult(rt_1, r, false);
				D.Tmmult(Prod, D, false);
				Dinv.invert(Prod);
				Dinv.vmult(rt_2, rt_1,false);
				a.add(-1.0,rt_2);
			}
			int count = 0;
			double * arr = new double[freedofs];
			if (GlobalParams.MPI_Rank == 0) {
				count = freedofs;
				for (int j = 0; j < freedofs; j++) {
					arr[j] = a(j);
				}
			}
			MPI_Scatter(arr, freedofs, MPI_DOUBLE, arr, freedofs, MPI_DOUBLE, 0, GlobalParams.MPI_Communicator);
			if(GlobalParams.MPI_Rank != 0) {
				for(int j = 0; j<freedofs; j++) {
					a(j) = arr[j];
				}
			}
			for(int j = 0; j<freedofs; j++) {
				structure->set_dof(j,a(j), true);
			}
			MPI_Barrier(GlobalParams.MPI_Communicator);

			if (GlobalParams.MPI_Rank == 0) {
				std::cout << " Neue Konfiguration: ";
				for (int j = 0; j < freedofs; j++) {
					std::cout << a(j) << ", ";
				}
				std::cout << std::endl;
			}
		}
	}
		/**
		waveguide.run();

		steps_counter ++;
		waveguide.store();
		double quality = waveguide.evaluate_overall();
		std::cout << " Initial quality: " ;
		for(unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); i++) {
			std::cout <<waveguide.qualities[i] << " " ;
		}
		std::cout << std::endl;
		const double initial_quality = quality;
		std::vector<double> old_position(freedofs);
		//if(GlobalParams.PRM_S_DoOptimization) {

			std::cout << "Initial Passthrough-quality: " << 100*quality << "%." << std::endl;
			std::cout << "Calculating Gradients ..." << std::endl;
			while (steps_counter < System_Parameters.PRM_Op_MaxCases && step_width > 0.00001)
			{
				GradientTable mytable(steps_counter,structure->InitialDofs, initial_quality, structure->Dofs () , quality );
				std::cout << "Current configuration: " << std::endl;
				structure->Print();
				double norm = 0.0;
				std::cout << std::endl;
				for(int i = 0; i < freedofs; i++) {
					std::cout << "Gradient "<< i << ": ..." << std::endl;
					double val = structure->get_dof(i, true);
					double new_val = val + step_width/10.0;
					if(i%3 == 0){
						if(new_val < - GlobalParams.PRM_O_MaximumFactor * GlobalParams.PRM_M_W_Delta/2.0) {
							new_val = - GlobalParams.PRM_O_MaximumFactor * GlobalParams.PRM_M_W_Delta/2.0;
						}
						if(new_val >  GlobalParams.PRM_O_MaximumFactor * GlobalParams.PRM_M_W_Delta/2.0) {
							new_val = GlobalParams.PRM_O_MaximumFactor * GlobalParams.PRM_M_W_Delta/2.0;
						}
					}
					if(i%3 == 1){
						if(new_val < GlobalParams.PRM_O_MinimumFactor * std::min(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut)) {
							new_val = GlobalParams.PRM_O_MinimumFactor * std::min(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut);
						}
						if(new_val > GlobalParams.PRM_O_MaximumFactor * std::max(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut)) {
							new_val = GlobalParams.PRM_O_MaximumFactor *  std::max(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut);
						}
					}

					structure->set_dof(i, new_val, true);
					waveguide.rerun();
					steps_counter ++;
					double temp_quality = waveguide.evaluate_overall();
					std::cout << "Quality after adjusting position (calculating gradient): " << temp_quality *100 << "% "<< std::endl;
					gradient[i] = quality - temp_quality;
					norm += gradient[i] * gradient[i];
					structure->set_dof(i, val, true);
					mytable.AddComputationResult(i, new_val-val, temp_quality);
				}
				std::cout << "Gradient for free dofs: (";
				for(int i = 0; i < freedofs; i++) {
					std::cout << gradient[i];
					if( i < freedofs - 1)std::cout << ",";
				}
				std::cout << ")" << std::endl;
				norm = sqrt(norm);
				std::cout << "Setting step to (";
				dealii::Vector<double> stepvector;
				stepvector.reinit(structure->NFreeDofs());

				for(int i = 0; i < freedofs; i++) {
					double step = (-1.0) * gradient[i] / norm;
					step *= step_width;
					old_position[i] = structure->get_dof(i, true);
					double val = structure->get_dof(i, true) + step;
					if(i%3 == 0){
						if(val < - GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0) {
							val = - GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0;
						}
						if(val >  GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0) {
							val = GlobalParams.PRM_O_MinimumFactor * GlobalParams.PRM_M_W_Delta/2.0;
						}
					}
					if(i%3 == 1){
						if(val < GlobalParams.PRM_O_MinimumFactor * std::min(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut)) {
							val = GlobalParams.PRM_O_MinimumFactor * std::min(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut);
						}
						if(val > GlobalParams.PRM_O_MaximumFactor * std::max(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut)) {
							val = GlobalParams.PRM_O_MaximumFactor *  std::max(GlobalParams.PRM_M_C_RadiusIn, GlobalParams.PRM_M_C_RadiusOut);
						}
					}

					std::cout << val;
					if( i < freedofs - 1)std::cout << ",";
					structure->set_dof(i, val, true);
					stepvector[i] = val - old_position[i];
				}
				std::cout << ")" << std::endl;

				std::cout << "Calculating solution after step... " ;
				waveguide.rerun();
				steps_counter ++;
				double step_quality = waveguide.evaluate_overall();
				std::cout << "Quality after the step: " << step_quality * 100 << "%";
				mytable.AddFullStepResult(stepvector,step_quality);
				if(step_quality < quality) {
					std::cout << "... not accepted (reduced quality). Undoing step and reducing step-width.";
					std::cout << "New Step-width: " << step_width *0.1 << std::endl;
					for(int i = 0; i < freedofs; i++) {
						structure->set_dof(i, old_position[i], true);
					}
					step_width *= 0.1;
				} else {
					std::cout << "... accepted. Updating current quality." << std::endl;
					best = steps_counter;
					quality = step_quality;
					waveguide.store();
				}
				mytable.PrintTable();
			}

			std::cout << "The best configuration was achieved in step number "<< best<<". The configuration is: " << std::endl;
			structure->Print();
	} else {
		std::cout << "Only one single calculation was done. The result has been saved. If you wish to optimize the shape, set the according parameter in the input-file." << std::endl;
	}**/

}

template<typename Matrix, typename Vector>
Optimization<Matrix, Vector>::~Optimization() {

}
