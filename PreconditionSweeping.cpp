/*
 * This file contains the implementation of the Sweeping Preconditioner.
 *
 *  \author Pascal Kraft
 *  \date 10.12.2015
 */


#ifndef SweepingPreconditionerCppFlag
#define SweepingPreconditionerCppFlag

#include <deal.II/lac/block_matrix_array.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/base/tensor.h>
#include "PreconditionSweeping.h"

using namespace dealii;

template<typename MatrixType, typename VectorType  >
PreconditionSweeping<MatrixType, VectorType>::AdditionalData (const double in_alpha , QGauss<3> & in_qf, FESystem<3> & in_fe, DoFHandler<3> & in_dofh, WaveguideStructure & in_structure): alpha(in_alpha) {
	quadrature_formula = in_qf;
	fe = in_fe;

}

template<typename MatrixType, typename VectorType  >
PreconditionSweeping<MatrixType, VectorType>::AdditionalData ( QGauss<3> & in_qf, FESystem<3> & in_fe, DoFHandler<3> & in_dofh, WaveguideStructure & in_structure): alpha(1.0) {

}

template<typename MatrixType, typename VectorType  >
bool PreconditionSweeping<MatrixType, VectorType>::PML_in_X(Point<3> &p) {
	double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	return p(0) < -(pmlboundary) ||p(0) > (pmlboundary);
}

template<typename MatrixType, typename VectorType>
bool PreconditionSweeping<MatrixType, VectorType>::PML_in_Y(Point<3> &p) {
	double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	return p(1) < -(pmlboundary) ||p(1) > (pmlboundary);
}

template<typename MatrixType, typename VectorType>
bool PreconditionSweeping<MatrixType, VectorType>::PML_in_Z(Point<3> &p, unsigned int block) {

	bool up =    (( p(2) + GlobalParams.PRM_M_R_ZLength/2.0 + GlobalParams.PRM_M_BC_XYin  ) - (block+1) * l + width) > 0;
	bool down = -(( p(2) + GlobalParams.PRM_M_R_ZLength/2.0 + GlobalParams.PRM_M_BC_XYin  ) - (block-1) * l - width) > 0;

	return up || down;
}

template<typename MatrixType, typename VectorType >
double PreconditionSweeping<MatrixType, VectorType>::PML_X_Distance(Point<3> &p){
	double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	if(p(0) >0){
		return p(0) - (pmlboundary) ;
	} else {
		return -p(0) - (pmlboundary);
	}
}

template<typename MatrixType, typename VectorType >
double PreconditionSweeping<MatrixType, VectorType>::PML_Y_Distance(Point<3> &p){
	double pmlboundary = (((GlobalParams.PRM_M_C_RadiusIn + GlobalParams.PRM_M_C_RadiusOut) / 2.0 ) * 15.5 / 4.35) * ((100.0 - GlobalParams.PRM_M_BC_Mantle)/100.0);
	if(p(1) >0){
		return p(1) - (pmlboundary);
	} else {
		return -p(1) - (pmlboundary);
	}
}

template<typename MatrixType, typename VectorType >
double PreconditionSweeping<MatrixType, VectorType>::PML_Z_Distance(Point<3> &p, unsigned int block ){
	double l = (double)(GlobalParams.PRM_M_R_ZLength + GlobalParams.PRM_M_BC_XYin + GlobalParams.PRM_M_BC_XYout) / (GlobalParams.PRM_M_W_Sectors);
	double width = l * 0.1;
	if( ( p(2) +GlobalParams.PRM_M_R_ZLength/2.0 + GlobalParams.PRM_M_BC_XYin  )-  block * l < 0){
		return -(( p(2) + GlobalParams.PRM_M_R_ZLength/2.0 + GlobalParams.PRM_M_BC_XYin  ) - (block-1) * l - width);
	} else {
		return  (( p(2) + GlobalParams.PRM_M_R_ZLength/2.0 + GlobalParams.PRM_M_BC_XYin  ) - (block+1) * l + width);
	}
}


template<typename MatrixType, typename VectorType >
Tensor<2,3, std::complex<double>> PreconditionSweeping<MatrixType, VectorType>::get_Tensor(Point<3> & position, bool inverse , bool epsilon, int block) {

	double omegaepsilon0 = GlobalParams.PRM_C_omega * ((System_Coordinate_in_Waveguide(position))?GlobalParams.PRM_M_W_EpsilonIn : GlobalParams.PRM_M_W_EpsilonOut);
	std::complex<double> sx(1.0, 0.0),sy(1.0,0.0), sz(1.0,0.0);
	if(PML_in_X(position)){
		double r,d, sigmax;
		r = PML_X_Distance(position);
		d = GlobalParams.PRM_M_R_XLength * 1.0 * GlobalParams.PRM_M_BC_Mantle/100.0;
		sigmax = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaXMax;
		sx.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaXMax);
		sx.imag( sigmax / ( omegaepsilon0));
	}
	if(PML_in_Y(position)){
		double r,d, sigmay;
		r = PML_Y_Distance(position);
		d = GlobalParams.PRM_M_R_YLength * 1.0 * GlobalParams.PRM_M_BC_Mantle/100.0;
		sigmay = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaYMax;
		sy.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaYMax);
		sy.imag( sigmay / ( omegaepsilon0));
	}
	if(PML_in_Z(position, block)){
		double r,d, sigmaz;
		r = PML_Z_Distance(position, block);
		d = (position(2)<0)? GlobalParams.PRM_M_BC_XYin : GlobalParams.PRM_M_BC_XYout;
		sigmaz = pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_SigmaZMax;
		sz.real( 1 + pow(r/d , GlobalParams.PRM_M_BC_M) * GlobalParams.PRM_M_BC_KappaZMax);
		sz.imag( sigmaz / omegaepsilon0 );
	}

	Tensor<2,3, double> transformation = data.structure.TransformationTensor(position[0], position[1], position[2]);

	Tensor<2,3, std::complex<double>> ret;
	for(int l = 0; l<3; l++) {
		for(int k = 0; k<3; k++) {
			std::complex<double> temp(transformation[l][k],0.0);
			ret[l][k] = temp;
		}
	}

	if(epsilon) {
		if(System_Coordinate_in_Waveguide(position) ) {
			ret *= GlobalParams.PRM_M_W_EpsilonIn;
		} else {
			ret *= GlobalParams.PRM_M_W_EpsilonOut;
		}
		ret *= GlobalParams.PRM_C_Eps0;
	} else {
		ret *= GlobalParams.PRM_C_Mu0;
	}

	ret[0][0] *= sy * sz / sx ;
	ret[0][1] *= sz ;
	ret[0][2] *= sy ;

	ret[1][0] *= sz ;
	ret[1][1] *= sx * sz / sy ;
	ret[1][2] *= sx ;

	ret[2][0] *= sy ;
	ret[2][1] *= sx ;
	ret[2][2] *= sx * sy / sz ;

	if(inverse) return invert(ret) ;

	else return ret;
}

template<typename MatrixType, typename VectorType >
Tensor<1,3, std::complex<double>> PreconditionSweeping<MatrixType, VectorType>::Conjugate_Vector(Tensor<1,3, std::complex<double>> input) {
	Tensor<1,3, std::complex<double>> ret ;
	for(int i= 0; i< 3; i++){
		ret[i].real(input[i].real());
		ret[i].imag( - input[i].imag());

	}
	return ret;
}

template <>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::initialize( dealii::BlockSparseMatrix<double> &System_Matrix, const PreconditionSweeping<dealii::BlockSparseMatrix<double>,dealii::BlockVector<double> >::AdditionalData &data):
	l((double)(GlobalParams.PRM_M_R_ZLength + GlobalParams.PRM_M_BC_XYin + GlobalParams.PRM_M_BC_XYout) / (GlobalParams.PRM_M_W_Sectors)),
	width(l * 0.1)
{

	// First Block. Prepare Solver directly.
	SparseDirectUMFPACK solver;
	solver.initialize(System_Matrix.block(0,0));
	inverse_blocks.push_back(solver);

	// Build local blocks
	if(System_Matrix.m() != System_Matrix.n()) {
		std::cout << "Critical Error in the Preconditioner. System Matrix block count mismatch!" << std::endl;
	}
	for(unsigned int block = 1; block <System_Matrix.m(); block++ ) {
		dealii::BlockMatrixArray<double, dealii::BlockVector<double> > temp;
		temp.initialize(2,2);



		FEValues<3>  	fe_values (data.fe, data.quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points );
		std::vector<Point<3> > quadrature_points;
		const unsigned int   					dofs_per_cell	= data.fe.dofs_per_cell;
		const unsigned int   					n_q_points		= data.quadrature_formula.size();

		FullMatrix<double>						cell_matrix_real (dofs_per_cell, dofs_per_cell);
		Vector<double>							cell_rhs (dofs_per_cell);
		cell_rhs = 0;
		Tensor<2,3, std::complex<double>> 		epsilon, mu;
		std::vector<types::global_dof_index> 	local_dof_indices (dofs_per_cell);
		const FEValuesExtractors::Vector 		real(0), imag(3);
		DoFHandler<3>::active_cell_iterator 	cell, endc;
		int test = 0;
		cell = data.dof_handler.begin_active(),
		endc = data.dof_handler.end();

		for (; cell!=endc; ++cell)
		{
			if(cell->subdomain_id() == block-1 || cell->subdomain_id() == block) {
				fe_values.reinit (cell);
				quadrature_points = fe_values.get_quadrature_points();
				cell_matrix_real = 0;

				for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
				{
					epsilon = get_Tensor(quadrature_points[q_index],  false, true, block);
					mu = get_Tensor(quadrature_points[q_index], true, false, block);
					const double JxW = fe_values.JxW(q_index);
					for (unsigned int i=0; i<dofs_per_cell; i++){
						Tensor<1,3, std::complex<double>> I_Curl;
						Tensor<1,3, std::complex<double>> I_Val;
						for(int k = 0; k<3; k++){
							I_Curl[k].imag(fe_values[imag].curl(i, q_index)[k]);
							I_Curl[k].real(fe_values[real].curl(i, q_index)[k]);
							I_Val[k].imag(fe_values[imag].value(i, q_index)[k]);
							I_Val[k].real(fe_values[real].value(i, q_index)[k]);
						}

						for (unsigned int j=0; j<dofs_per_cell; j++){
							Tensor<1,3, std::complex<double>> J_Curl;
							Tensor<1,3, std::complex<double>> J_Val;
							for(int k = 0; k<3; k++){
								J_Curl[k].imag(fe_values[imag].curl(j, q_index)[k]);
								J_Curl[k].real(fe_values[real].curl(j, q_index)[k]);
								J_Val[k].imag(fe_values[imag].value(j, q_index)[k]);
								J_Val[k].real(fe_values[real].value(j, q_index)[k]);
							}
							test ++;

							std::complex<double> x = (mu * I_Curl) * Conjugate_Vector(J_Curl) * JxW - ( ( epsilon * I_Val ) * Conjugate_Vector(J_Val))*JxW*GlobalParams.PRM_C_omega*GlobalParams.PRM_C_omega;

							cell_matrix_real[i][j] += x.real();

						}
					}
				}
				cell->get_dof_indices (local_dof_indices);

				cm.distribute_local_to_global(cell_matrix_real, cell_rhs, local_dof_indices,system_matrix, system_rhs, false);

			}
		}





		SparseDirectUMFPACK solver;
		solver.initialize(temp);
		inverse_blocks.push_back(solver);
	}

	alpha = data.alpha;



}

template<>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::vmult (  dealii::BlockVector<double> &  out_vec , const dealii::BlockVector<double> & in_vec ) {
	unsigned int Blocks = in_vec.n_blocks();

	inverse_blocks[0].vmult(out_vec.block(0) , in_vec.block(0));

	for ( unsigned int i = 1; i<Blocks; i++) {
		dealii::BlockVector<double> temp;
		dealii::BlockVector<double> temp2;
		temp.reinit(2);
		temp.block(0).reinit(in_vec.block(i-1).size());
		temp.block(1).reinit(in_vec.block(i).size());
		temp.block(1) = in_vec.block(i);
		temp2.reinit(2);
		temp2.block(0).reinit(in_vec.block(i-1).size());
		temp2.block(1).reinit(in_vec.block(i).size());
		inverse_blocks[i].vmult(temp2, temp);
		out_vec.block(i) = temp2.block(1);
	}
}

template<>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::Tvmult (  dealii::BlockVector<double> &  out_vec , const dealii::BlockVector<double> & in_vec ) {
	vmult (  out_vec ,  in_vec );
}

#endif
