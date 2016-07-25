#include "Optimization.h"
#include "Waveguide.h"
#include "GradientTable.h"

using namespace dealii;

template<typename Matrix,typename Vector>
Optimization<Matrix, Vector >::Optimization( Parameters in_System_Parameters ,Waveguide<Matrix, Vector >  &in_wg)
	:
		pout(std::cout, GlobalParams.MPI_Rank==0),
		dofs(structure->NDofs()),
		residuals_count((int)floor((double)GlobalParams.MPI_Size * (((double)GlobalParams.PRM_M_W_Sectors - (double)GlobalParams.PRM_M_BC_XYout)/((double)GlobalParams.PRM_M_W_Sectors)))+1),
		freedofs(structure->NFreeDofs()),
		System_Parameters(in_System_Parameters),
		waveguide(in_wg)
	{

}

template<typename Matrix,typename Vector>
void Optimization<Matrix, Vector>::run() {
	structure->estimate_and_initialize();
	double step = 0.00001;
	dealii::Vector<double> r (residuals_count);
	dealii::FullMatrix<double> D(residuals_count, freedofs);
	dealii::FullMatrix<double> Prod(freedofs, freedofs);
	dealii::FullMatrix<double> Dinv(freedofs, freedofs);
	dealii::Vector<double> rt_1 (freedofs);
	dealii::Vector<double> rt_2 (freedofs);

	pout << "Residual Count: " << residuals_count << std::endl;

	dealii::Vector<double> a(freedofs);
	if(!GlobalParams.PRM_S_DoOptimization) {
		waveguide.run();
	} else {
		for (int i = 0; i < GlobalParams.PRM_Op_MaxCases; i++){
			if(i == 0) {
				waveguide.run();
			} else {
				waveguide.rerun();
			}
			pout << "Run Complete for proc. ";
			pout << GlobalParams.MPI_Rank;
			pout <<", calculating Gradient" << std::endl;
			MPI_Barrier(GlobalParams.MPI_Communicator);
			pout<< "Residuals: ";

			double quality = 0;

			if((structure->Z_to_Sector_and_local_z(GlobalParams.PRM_M_R_ZLength/2.0 - 0.00001)).first == GlobalParams.MPI_Rank) {
				quality = waveguide.evaluate_for_z(GlobalParams.PRM_M_R_ZLength/2.0 - 0.00001);
			}

			quality = Utilities::MPI::max(quality, MPI_COMM_WORLD);
			double reference = 1.0;
			bool cont = true;
			if(GlobalParams.MPI_Rank == 0){
				reference = waveguide.evaluate_for_z(- GlobalParams.PRM_M_R_ZLength / 2.0);
				if ( reference < 0.00001) {
					cont = false;
				}
			}
			if(!cont) {
				pout << "No residual on incoming side - No signal."<< std::endl;
				exit(0);
			}
			for (int j = 0; j < residuals_count-1; j++ ){
				r[j] = 1- (abs(waveguide.qualities[j])/reference);
				pout << r[j] << " ,";
			}
			r[residuals_count-1] = quality;
			pout<< std::endl;

			for(  int j = 0; j < freedofs; j++) {
				a(j) = structure->get_dof(j,true);
			}
			pout << "Starting gradient estimation" << std::endl;
			for (int j = 0; j < freedofs; j++) {
				double old = structure->get_dof(j,true);
				structure->set_dof(j, old + step, true);
				MPI_Barrier(GlobalParams.MPI_Communicator);
				pout << "Gradient step "<< j+1 <<" starting ..." << std::endl;
				waveguide.rerun();
				pout << "Gradient step " << j+1 << " of " << freedofs << " done." << std::endl;
				if((structure->Z_to_Sector_and_local_z(GlobalParams.PRM_M_R_ZLength/2.0 - 0.00001)).first == GlobalParams.MPI_Rank) {
					quality = waveguide.evaluate_for_z(GlobalParams.PRM_M_R_ZLength/2.0 - 0.00001);
				}

				quality = Utilities::MPI::max(quality, MPI_COMM_WORLD);

				pout << "Total quality: " << quality * 100.0<< "%"<< std::endl;
				if(GlobalParams.MPI_Rank == 0){
					for( int k = 0; k < residuals_count-1; k++) {
						double res = 1- (abs(waveguide.qualities[k])/reference);
						D[k][j]= (res - r[k])/step;
					}
					double res = 1- (abs(waveguide.qualities[residuals_count-1])/reference);
					D[residuals_count-1][j]= (res - r[residuals_count-1])/step;
				}
				structure->set_dof(j, old , true);
				MPI_Barrier(GlobalParams.MPI_Communicator);
			}
			pout << "All gradient steps done. Executing Gauss-Newton" << std::endl;

			if (GlobalParams.MPI_Rank == 0) {
				pout << "Matrix D:"<<std::endl;
				D.print(std::cout);
				D.Tvmult(rt_1, r, false);
				pout << "D^t * r:"<<std::endl;
				rt_1.print(std::cout);
				D.Tmmult(Prod, D, false);
				pout << "D^t * D:"<<std::endl;
				Prod.print(std::cout);
				Dinv.invert(Prod);
				pout << "(D^t * D)^(-1):"<<std::endl;
				Dinv.print(std::cout);
				Dinv.vmult(rt_2, rt_1,false);
				pout << "-a:"<<std::endl;
				rt_2.print(std::cout);
				a.add(-1.0,rt_2);
				pout << "Norm of the step: " << rt_2.l2_norm() <<std::endl;
			}
			double * arr = new double[freedofs];
			if (GlobalParams.MPI_Rank == 0) {
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

}

template<typename Matrix, typename Vector>
Optimization<Matrix, Vector>::~Optimization() {

}
