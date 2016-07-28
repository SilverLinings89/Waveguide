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
	double alpha = 0.1;
	bool abort_condition = false;
	dealii::Vector<double> r (residuals_count);
	dealii::FullMatrix<double> D(residuals_count, freedofs);
	dealii::FullMatrix<double> Prod(freedofs, freedofs);
	dealii::FullMatrix<double> Dinv(freedofs, freedofs);
	dealii::Vector<double> rt_1 (freedofs);
	dealii::Vector<double> rt_2 (freedofs);


	double ct1 = 1;
	double ct2 = 1;
	double ct3 = 1;

	if(GlobalParams.MPI_Rank == 0) {
		ct1 = residuals_count-1;
		ct2 = freedofs;
		ct3 = GlobalParams.PRM_Op_MaxCases;
	}
	dealii::Vector<double> optimization_history (ct3);
	dealii::FullMatrix<double> params_history(ct3, ct2);
	dealii::FullMatrix<double> residuals_history(ct3, ct1);

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
			pout <<", Evaluating ..." << std::endl;
			MPI_Barrier(GlobalParams.MPI_Communicator);

			int eval = 1;
			double quality = 0;


			double reference = 1.0;
			bool cont = true;
			while (eval == 1) {
				if((structure->Z_to_Layer(GlobalParams.PRM_M_R_ZLength/2.0 - 0.00001)) == GlobalParams.MPI_Rank) {
					quality = waveguide.evaluate_for_z(GlobalParams.PRM_M_R_ZLength/2.0 - 0.00001);
				}

				quality = Utilities::MPI::max(quality, MPI_COMM_WORLD);
				eval = 0;
				if(GlobalParams.MPI_Rank == 0){

					reference = waveguide.evaluate_for_z(- GlobalParams.PRM_M_R_ZLength / 2.0);
					if ( reference < 0.00001) {
						cont = false;
					}
					optimization_history[i] = 100.0 * quality/reference ;
					if (i > 0) {
						if(optimization_history[i] < optimization_history[i-1]) {
							a.add(1.0,rt_2);
							alpha /= 4.0;
							rt_2 *= 0.25;
							a.add(-1.0,rt_2);
							eval = 1;
							pout << "Reducing step width because of loss of quality in last step. Undoing the step and rerunning... "<<std::endl;
						}
						if(alpha < 0.00000001) {
							pout << "The step has become too small. Finishing." << std::endl;
							abort_condition = true;
							MPI_Abort(MPI_COMM_WORLD,0);
						}
					}
				}
				eval = Utilities::MPI::max(eval, MPI_COMM_WORLD);
				if (eval == 1) {
					double * arr = new double[freedofs];

					if (GlobalParams.MPI_Rank == 0) {
						for (int j = 0; j < freedofs; j++) {
							arr[j] = a(j);
							params_history.set(i,j,a(j));
						}
					}
					MPI_Bcast(arr, freedofs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
					for(int j = 0; j<freedofs; j++) {
						a(j) = arr[j];
					}

					for(int j = 0; j<freedofs; j++) {
						structure->set_dof(j,a(j), true);
					}
					MPI_Barrier(GlobalParams.MPI_Communicator);
					waveguide.rerun();
				}

				alpha = Utilities::MPI::min(alpha, MPI_COMM_WORLD);
			}
			if(!cont) {
				pout << "No residual on incoming side - No signal."<< std::endl;
				exit(0);
			}
			pout<< "Residuals: ";
			for (int j = 0; j < residuals_count-1; j++ ){
				r[j] = abs(1.0 - (waveguide.qualities[j]/reference));
				pout << r[j] << " ,";
			}
			r[residuals_count-1] = ((double)i / 10.0) * abs(1.0 - quality/reference);
			pout << r[residuals_count-1]<<std::endl;


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
				quality = 0;
				if((structure->Z_to_Layer(GlobalParams.PRM_M_R_ZLength/2.0 - 0.00001)) == GlobalParams.MPI_Rank) {
					quality = waveguide.evaluate_for_z(GlobalParams.PRM_M_R_ZLength/2.0 - 0.00001);
				}

				quality = Utilities::MPI::max(quality, MPI_COMM_WORLD);

				pout << "Total quality: " << (quality/reference) * 100.0<< "%"<< std::endl;
				if(GlobalParams.MPI_Rank == 0){
					for( int k = 0; k < residuals_count-1; k++) {
						double res = abs(1.0- (waveguide.qualities[k]/reference));
						residuals_history.set(i,k,res);
						D[k][j]= -(res - r[k])/step;
					}
					double res = ((double)i / 10.0) * abs(1.0- (quality/reference));
					D[residuals_count-1][j]= -(res - r[residuals_count-1])/step;
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
				pout << "- step:"<<std::endl;
				rt_2.print(std::cout);
				rt_2 *= alpha;
				a.add(-1.0,rt_2);
				pout << "Norm of the step: " << rt_2.l2_norm() <<std::endl;
			}
			double * arr = new double[freedofs];

			if (GlobalParams.MPI_Rank == 0) {
				for (int j = 0; j < freedofs; j++) {
					arr[j] = a(j);
					params_history.set(i,j,a(j));
				}
			}
			MPI_Bcast(arr, freedofs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			for(int j = 0; j<freedofs; j++) {
				a(j) = arr[j];
			}

			for(int j = 0; j<freedofs; j++) {
				structure->set_dof(j,a(j), true);
			}
			MPI_Barrier(GlobalParams.MPI_Communicator);

			if (GlobalParams.MPI_Rank == 0) {
				std::cout << " New configuration: ";
				for (int j = 0; j < freedofs; j++) {
					std::cout << a(j) << ", ";
				}
				std::cout << std::endl;
				std::cout << "Optimization History: "<<std::endl;

				for(int l = 0; l< i+1; l++){
					std:: cout << optimization_history(l) << std::endl;
				}
				std::cout << "Residual History: "<<std::endl;
				for(int l = 0; l< i+1; l++){
					for (int k = 0; k < residuals_count -1; k++) {
						std:: cout << residuals_history[l][k] << "  ";
					}
					std::cout << std::endl;
				}
				// residuals_history.print(std::cout);
				std::cout << "Parameters History: "<<std::endl;
				// params_history.print(std::cout);
				for(int l = 0; l< i+1; l++){
					for (int k = 0; k < freedofs; k++) {
						std:: cout << params_history[l][k] << "  ";
					}
					std::cout << std::endl;
				}
			}

			if(abort_condition) {
				pout << "A solution has been found. Terminanting." <<std::endl;
				MPI_Abort(MPI_COMM_WORLD,0);
			}
			MPI_Barrier(GlobalParams.MPI_Communicator);

		}
	}

}

template<typename Matrix, typename Vector>
Optimization<Matrix, Vector>::~Optimization() {

}
