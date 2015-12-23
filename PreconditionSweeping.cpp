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
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include "PreconditionSweeping.h"

using namespace dealii;

template<typename MatrixType, typename VectorType  >
PreconditionSweeping<MatrixType, VectorType>::PreconditionSweeping( PreconditionSweeping<MatrixType, VectorType>::AdditionalData & in_data): data(in_data.alpha, in_data.nonzero)
{
	Sectors = GlobalParams.PRM_M_W_Sectors;
}


template<typename MatrixType, typename VectorType  >
PreconditionSweeping<MatrixType, VectorType>::AdditionalData::AdditionalData (  double in_alpha, int in_nonzero):
alpha(in_alpha),
nonzero(in_nonzero)
{}





template <>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::initialize( dealii::BlockSparseMatrix<double> &System_Matrix, dealii::BlockSparseMatrix<double> &Preconditioner_Matrix1, dealii::BlockSparseMatrix<double> &Preconditioner_Matrix2)
{

	// First Block. Prepare Solver directly.
	SparseDirectUMFPACK solver;
	solver.initialize(System_Matrix.block(0,0));
	inverse_blocks.push_back(solver);

	// Build local blocks
	if(System_Matrix.m() != System_Matrix.n()) {
		std::cout << "Critical Error in the Preconditioner. System Matrix block count mismatch!" << std::endl;
	}
	for(unsigned int block = 2; block <System_Matrix.m(); block++ ) {
		dealii::BlockSparseMatrix<double> temp;
		BlockSparsityPattern tsp;
		tsp.reinit(2,2);

		tsp.block(0,0).reinit(Preconditioner_Matrix2.block(block-1, block-1).m(),Preconditioner_Matrix2.block(block-1, block-1).n(), data.nonzero );
		tsp.block(1,0).reinit(Preconditioner_Matrix2.block(block  , block-1).m(),Preconditioner_Matrix2.block(block  , block-1).n(), data.nonzero );
		tsp.block(0,1).reinit(Preconditioner_Matrix2.block(block-1, block  ).m(),Preconditioner_Matrix2.block(block-1, block  ).n(), data.nonzero );
		tsp.block(1,1).reinit(Preconditioner_Matrix2.block(block  , block  ).m(),Preconditioner_Matrix2.block(block  , block  ).n(), data.nonzero );
		deallog << "Sizes:"<<std::endl;
		deallog << " (" << Preconditioner_Matrix2.block(block-1, block-1).m()<< "," << Preconditioner_Matrix2.block(block-1, block-1).n() << ") "<< std::endl;
		deallog << " (" << Preconditioner_Matrix2.block(block, block-1).m()<< "," << Preconditioner_Matrix2.block(block, block-1).n() << ") "<< std::endl;
		deallog << " (" << Preconditioner_Matrix2.block(block-1, block).m()<< "," << Preconditioner_Matrix2.block(block-1, block).n() << ") "<< std::endl;
		deallog << " (" << Preconditioner_Matrix2.block(block, block).m()<< "," << Preconditioner_Matrix2.block(block, block).n() << ") "<< std::endl;
		tsp.collect_sizes();
		tsp.compress();

		//deallog << "Does it happen here?" << std::endl;
		temp.reinit(tsp);
		//deallog << "Nope!" << std::endl;

		if((block-1)%2 == 0) {
			deallog << 1 << std::endl;
			temp.block(0,0).reinit(Preconditioner_Matrix2.block(block-1, block-1).get_sparsity_pattern());
			temp.block(0,0).copy_from(Preconditioner_Matrix2.block(block-1, block-1));
			temp.block(1,0).reinit(Preconditioner_Matrix2.block(block  , block-1).get_sparsity_pattern());
			temp.block(1,0).copy_from(Preconditioner_Matrix2.block(block  , block-1));
			temp.block(0,1).reinit(Preconditioner_Matrix2.block(block-1, block  ).get_sparsity_pattern());
			temp.block(0,1).copy_from(Preconditioner_Matrix2.block(block-1, block  ));
			temp.block(1,1).reinit(Preconditioner_Matrix2.block(block  , block  ).get_sparsity_pattern());
			temp.block(1,1).copy_from(Preconditioner_Matrix2.block(block  , block  ));
			deallog << 2 << std::endl;
		} else {
			deallog << 3 << std::endl;
			temp.block(0,0).reinit(Preconditioner_Matrix1.block(block-1, block-1).get_sparsity_pattern());
			temp.block(0,0).copy_from(Preconditioner_Matrix1.block(block-1, block-1));
			temp.block(1,0).reinit(Preconditioner_Matrix1.block(block  , block-1).get_sparsity_pattern());
			temp.block(1,0).copy_from(Preconditioner_Matrix1.block(block  , block-1));
			temp.block(0,1).reinit(Preconditioner_Matrix1.block(block-1, block  ).get_sparsity_pattern());
			temp.block(0,1).copy_from(Preconditioner_Matrix1.block(block-1, block  ));
			temp.block(1,1).reinit(Preconditioner_Matrix1.block(block  , block  ).get_sparsity_pattern());
			temp.block(1,1).copy_from(Preconditioner_Matrix1.block(block  , block  ));
			deallog << 4 << std::endl;
		}

		deallog << Preconditioner_Matrix2.block(block-1, block-1).get_sparsity_pattern().n_cols();
		deallog << " " << Preconditioner_Matrix2.block(block-1, block-1).get_sparsity_pattern().n_rows();
		deallog << " " << temp.block(0,0).n_nonzero_elements() << std::endl;

		temp.compress(dealii::VectorOperation::insert);
		SparseDirectUMFPACK solver;
		deallog << "Point 1" << std::endl;
		solver.initialize(temp);
		deallog << "Point 2" << std::endl;
		inverse_blocks.push_back(solver);
		deallog << "Finished block" << block<< std::endl;
	}

}



template<>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::vmult (  dealii::BlockVector<double> &  out_vec , dealii::BlockVector<double> & in_vec ) const {
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
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::Tvmult (  dealii::BlockVector<double> &  out_vec ,dealii::BlockVector<double> & in_vec ) const {
	vmult (  out_vec ,  in_vec );
}

#endif
