#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_vector_base.h>
#include <deal.II/lac/solver.h>

#include "PreconditionerSweeping.h"

using namespace dealii;

PreconditionerSweeping::~PreconditionerSweeping (){
	delete solver;
}
PreconditionerSweeping::PreconditionerSweeping (  int in_own, int in_others, int bandwidth,  IndexSet locally_owned, int in_upper):

		matrix(in_own+in_others, in_own+in_others, bandwidth)
{
		upper = in_upper;
		own = in_own;
		others = in_others;
		IndexSet elements (own+others);
		elements.add_range(0,own+others);
		solver = new TrilinosWrappers::SolverDirect(s, TrilinosWrappers::SolverDirect::AdditionalData(false, GlobalParams.PRM_S_Preconditioner));
		indices = new int[locally_owned.n_elements()];
		for(int i = 0; i < own; i++){
			indices[i] = locally_owned.nth_index_in_set(i);
		}
   }


void PreconditionerSweeping::vmult (TrilinosWrappers::MPI::Vector       &dst,
			const TrilinosWrappers::MPI::Vector &src)const
{
	// line 1
	dealii::Vector<double> input(own);
	for (int i = 0; i < own; i++) {
		input[i] = src[indices[i]];
	}

	if (GlobalParams.MPI_Rank == 0) {

		dealii::Vector<double> outputb(own);

		solver->solve( matrix , outputb, input);

		double *  trans1 = new double[own];

		for(int i = 0; i < own; i++) {
			input[i] = outputb[i];
			trans1[i] = outputb[i];
		}

		MPI_Send(trans1, own, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

	} else {

		double * trans2 = new double[others];
		MPI_Recv(trans2, others, MPI_DOUBLE, GlobalParams.MPI_Rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
		dealii::Vector<double> temp2 (others);
		for (int i = 0; i < others; i++) {
			temp2[i] = trans2[i];
		}

		std::cout << "A" << GlobalParams.MPI_Rank << " " << temp2.l2_norm() <<  std::endl;

		dealii::Vector<double> temp3 (own);

		LowerProduct(temp2, temp3);

		std::cout << "D" << GlobalParams.MPI_Rank << " " << temp3.l2_norm() << std::endl;


		//Line 4
		input -= temp3;
		
		std::cout << "E" << GlobalParams.MPI_Rank << " " << input.l2_norm() << std::endl;


		if(GlobalParams.MPI_Rank != GlobalParams.MPI_Size -1) {
			dealii::Vector<double> temp4 (own);

			Hinv(input, temp4);

			std::cout << "F" << GlobalParams.MPI_Rank << " " << temp4.l2_norm() << std::endl;

			double * trans3 = new double [own];
			for(int i = 0; i < own; i++) {
				trans3[i] = temp4[i];
			}

			MPI_Send(trans3, own, MPI_DOUBLE, GlobalParams.MPI_Rank + 1, 0, MPI_COMM_WORLD);
		}
	}
        
    MPI_Barrier(MPI_COMM_WORLD);
        
    if (GlobalParams.MPI_Rank == 0) {
        std::cout << "S1 done ...";
    }
        
    // Line 8
                
    if(GlobalParams.MPI_Rank != 0) {
        dealii::Vector<double> temp (own);
        for(int i =0; i < own; i++) {
            temp(i) = input[i];
        }
        Hinv(temp, input);
    }

    if (GlobalParams.MPI_Rank == 0) {
        std::cout << "P done ...";
    }
        
    MPI_Barrier(MPI_COMM_WORLD);
    // Line 11
    if ( GlobalParams.MPI_Rank == GlobalParams.MPI_Size -1) {
            dealii::Vector<double> back_sweep (others);
            double * trans4 = new double [others];
            UpperProduct(input, back_sweep);
            for (int i = 0; i < others; i++) {
                    trans4[i] = back_sweep(i);
            }
            MPI_Send(trans4, others, MPI_DOUBLE, GlobalParams.MPI_Rank - 1, 0, MPI_COMM_WORLD);
    } else {
            double * trans4 = new double [own];
            MPI_Recv(trans4, own, MPI_DOUBLE, GlobalParams.MPI_Rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            dealii::Vector<double> back_sweep (own);
            dealii::Vector<double> temp_calc (own);
            for (int i = 0; i < own; i++) {
                    temp_calc(i) = trans4[i];
            }
			std::cout << "B" << GlobalParams.MPI_Rank << " " << temp_calc.l2_norm() << std::endl;
            Hinv(temp_calc, back_sweep);
                
            input -= back_sweep;
                
            if(GlobalParams.MPI_Rank >0) {
                    dealii::Vector<double> back_sweep2 (others);
                    double * trans5 = new double [others];
                    UpperProduct(input, back_sweep2);
                    for (int i = 0; i < others; i++) {
                            trans5[i] = back_sweep2(i);
                    }
                    MPI_Send(trans5, others, MPI_DOUBLE, GlobalParams.MPI_Rank - 1, 0, MPI_COMM_WORLD);
            }
    }
        
    if (GlobalParams.MPI_Rank == 0) {
        std::cout << "S2 done ...";
    }
    
	std::cout << "C" << GlobalParams.MPI_Rank << " " << input.l2_norm() << std::endl;

    for(int i = 0; i < own; i++ ){
		dst[indices[i]] = input[i];
    }

    

}

void PreconditionerSweeping::Hinv(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {
	TrilinosWrappers::Vector inputb(own + others);

	for(int i = 0; i < own; i++) {
		inputb[i + others] = src(i);
	}

	TrilinosWrappers::Vector outputb(own + others);

	solver->solve( matrix , outputb, inputb);

	for(int i = 0; i < own; i++) {
		dst[i] = outputb[others + i];
	}
}

void PreconditionerSweeping::LowerProduct(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {

	dealii::Vector<double> in_temp (own+others);
	for (int i = 0; i < others; i++) {
		in_temp[i] = src[i];
	}
	std::cout << "LA" << GlobalParams.MPI_Rank << " " << in_temp.l2_norm() << std::endl;

	dealii::Vector<double> out_temp (own+others);
	matrix.vmult(out_temp, in_temp);

	std::cout << "LB" << GlobalParams.MPI_Rank << " " << out_temp.l2_norm() << matrix.l1_norm() << std::endl;
	
	for(int i = 0; i < own; i++) {
		dst[i] = out_temp[others + i];
		if (out_temp[others + i] != 0.0) {
			std::cout << "--"
		}
		std::cout << std::endl;
	}

	std::cout << "LC" << GlobalParams.MPI_Rank << " " << dst.l2_norm() << std::endl;

}

void PreconditionerSweeping::UpperProduct(const dealii::Vector<double> & src, dealii::Vector<double> & dst) const {

	dealii::Vector<double> in_temp (own+others);
	for (int i = 0; i < own; i++) {
		in_temp[others + i] = src[i];
	}

	std::cout << "UA" << GlobalParams.MPI_Rank << " " << in_temp.l2_norm() << std::endl;

	dealii::Vector<double> out_temp (own+others);
	matrix.vmult(out_temp, in_temp);

	std::cout << "UB" << GlobalParams.MPI_Rank << " " << out_temp.l2_norm() << std::endl;
	for(int i = 0; i < others; i++) {
		dst[i] = out_temp[ i];
	}


}


