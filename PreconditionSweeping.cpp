/*
 * This file contains the implementation of the Sweeping Preconditioner.
 *
 *  \author Pascal Kraft
 *  \date 10.12.2015
 */


#ifndef SweepingPreconditionerCppFlag
#define SweepingPreconditionerCppFlag
/**
#include <deal.II/lac/block_matrix_array.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include "PreconditionSweeping.h"
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_block_vector.h>

using namespace dealii;

template<typename MatrixType, typename VectorType  >
PreconditionSweeping<MatrixType, VectorType>::PreconditionSweeping( PreconditionSweeping<MatrixType, VectorType>::AdditionalData & in_data): data(in_data.alpha, in_data.nonzero)
{
	Sectors = GlobalParams.PRM_M_W_Sectors;
}

template<typename MatrixType, typename VectorType  >
PreconditionSweeping<MatrixType, VectorType>::AdditionalData::AdditionalData (  double in_alpha, unsigned int in_nonzero):
alpha(in_alpha),
nonzero(in_nonzero)
{}

template<typename MatrixType, typename VectorType  >
void PreconditionSweeping<MatrixType, VectorType>::AdditionalData::SetNonZero(unsigned int in_nonzero)
{

	nonzero = in_nonzero;

}
**/
/**
template<>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::Hinv(unsigned int block, dealii::BlockVector<double> &out_vec, dealii::BlockVector<double> &in_vec ) const {
	dealii::BlockVector<double> temp;
	temp.reinit(2);
	temp.block(0).reinit(in_vec.block(block-1).size());
	temp.block(1).reinit(in_vec.block(block).size());
	temp.collect_sizes();
	temp.block(0) = in_vec.block(block-1);
	temp.block(1) = in_vec.block(block);
	// deallog << "Input-vector prepared." << std::endl;

	out_vec.reinit(2);
	out_vec.block(0).reinit(in_vec.block(block-1).size());
	out_vec.block(1).reinit(in_vec.block(block).size());
	out_vec.collect_sizes();
	// deallog << "Output-vector prepared." << std::endl;

	inverse_blocks[block].vmult(out_vec, temp);
}
**/

/**
template <>
void PreconditionSweeping<PETScWrappers::MPI::SparseMatrix, dealii::PETScWrappers::MPI::Vector>::initialize( PETScWrappers::SparseMatrix & local_matrix) {


	// First Block. Prepare Solver directly.
	if(GlobalParams.MPI_Rank == 0) {
		inverse_block.solve(Preconditioner_Matrix1.block(0,0), PETScWrappers::Vector(Preconditioner_Matrix1.block(0,0).m()), PETScWrappers::Vector(Preconditioner_Matrix1.block(0,0).m()));

	}

	// Build local blocks
	if(System_Matrix->m() != System_Matrix->n()) {
		std::cout << "Critical Error in the Preconditioner. System Matrix block count mismatch!" << std::endl;
	}
	for(unsigned int block = 1; block <System_Matrix->n_block_cols(); block++ ) {
		dealii::BlockSparseMatrix<double> temp;
		BlockDynamicSparsityPattern tsp;
		tsp.reinit(2,2);

		if((block-1)%2 == 0) {
			tsp.block(0,0).reinit(Preconditioner_Matrix2.block(block-1, block-1).m(),Preconditioner_Matrix2.block(block-1, block-1).n() );
			SparseMatrix<double>::iterator it = Preconditioner_Matrix2.block(block-1, block-1).begin(),
			end = Preconditioner_Matrix2.block(block-1, block-1).end();
			for (; it!=end; ++it)
			{
				tsp.block(0,0).add(it->row(), it->column());
			}

			tsp.block(1,0).reinit(Preconditioner_Matrix2.block(block  , block-1).m(),Preconditioner_Matrix2.block(block  , block-1).n() );
			it = Preconditioner_Matrix2.block(block, block-1).begin();
			end = Preconditioner_Matrix2.block(block, block-1).end();
			for (; it!=end; ++it)
			{
				tsp.block(1,0).add(it->row(), it->column());
			}

			tsp.block(0,1).reinit(Preconditioner_Matrix2.block(block-1, block ).m(),Preconditioner_Matrix2.block(block-1, block  ).n() );
			it = Preconditioner_Matrix2.block(block-1, block).begin();
			end = Preconditioner_Matrix2.block(block-1, block).end();
			for (; it!=end; ++it)
			{
				tsp.block(0,1).add(it->row(), it->column());
			}

			tsp.block(1,1).reinit(Preconditioner_Matrix2.block(block  , block  ).m(),Preconditioner_Matrix2.block(block  , block  ).n() );
			it = Preconditioner_Matrix2.block(block, block).begin();
			end = Preconditioner_Matrix2.block(block, block).end();
			for (; it!=end; ++it)
			{
				tsp.block(1,1).add(it->row(), it->column());
			}

		} else {
			tsp.block(0,0).reinit(Preconditioner_Matrix1.block(block-1, block-1).m(),Preconditioner_Matrix1.block(block-1, block-1).n() );
			SparseMatrix<double>::iterator it = Preconditioner_Matrix1.block(block-1, block-1).begin(),
			end = Preconditioner_Matrix1.block(block-1, block-1).end();
			for (; it!=end; ++it)
			{
				tsp.block(0,0).add(it->row(), it->column());
			}

			tsp.block(1,0).reinit(Preconditioner_Matrix1.block(block  , block-1).m(),Preconditioner_Matrix1.block(block  , block-1).n() );
			it = Preconditioner_Matrix1.block(block, block-1).begin();
			end = Preconditioner_Matrix1.block(block, block-1).end();
			for (; it!=end; ++it)
			{
				tsp.block(1,0).add(it->row(), it->column());
			}

			tsp.block(0,1).reinit(Preconditioner_Matrix1.block(block-1, block  ).m(),Preconditioner_Matrix1.block(block-1, block  ).n() );
			it = Preconditioner_Matrix1.block(block-1, block).begin();
			end = Preconditioner_Matrix1.block(block-1, block).end();
			for (; it!=end; ++it)
			{
				tsp.block(0,1).add(it->row(), it->column());
			}

			tsp.block(1,1).reinit(Preconditioner_Matrix1.block(block  , block  ).m(),Preconditioner_Matrix1.block(block  , block  ).n() );
			it = Preconditioner_Matrix1.block(block, block).begin();
			end = Preconditioner_Matrix1.block(block, block).end();
			for (; it!=end; ++it)
			{
				tsp.block(1,1).add(it->row(), it->column());
			}
		}

		tsp.collect_sizes();
		BlockSparsityPattern dsp;
		dsp.copy_from(tsp);

		temp.reinit(dsp);

		if((block-1)%2 == 0) {

			temp.block(0,0) = 0;
			SparseMatrix<double>::iterator it = Preconditioner_Matrix2.block(block-1, block-1).begin(),
			end = Preconditioner_Matrix2.block(block-1, block-1).end();
			for (; it!=end; ++it)
			{
				temp.block(0,0).set(it->row(), it->column(), it->value());
			}

			temp.block(1,0) = 0;
			it = Preconditioner_Matrix2.block(block, block-1).begin();
			end = Preconditioner_Matrix2.block(block, block-1).end();
			for (; it!=end; ++it)
			{
				temp.block(1,0).set(it->row(), it->column(), it->value());
			}

			temp.block(0,1) = 0;
			it = Preconditioner_Matrix2.block(block-1, block).begin();
			end = Preconditioner_Matrix2.block(block-1, block).end();
			for (; it!=end; ++it)
			{
				temp.block(0,1).set(it->row(), it->column(), it->value());
			}

			temp.block(1,1) = 0;
			it = Preconditioner_Matrix2.block(block, block).begin();
			end = Preconditioner_Matrix2.block(block, block).end();
			for (; it!=end; ++it)
			{
				temp.block(1,1).set(it->row(), it->column(), it->value());
			}


		} else {

			temp.block(0,0) = 0;
			SparseMatrix<double>::iterator it = Preconditioner_Matrix1.block(block-1, block-1).begin(),
			end = Preconditioner_Matrix1.block(block-1, block-1).end();
			for (; it!=end; ++it)
			{
				temp.block(0,0).set(it->row(), it->column(), it->value());
			}


			temp.block(1,0) = 0;
			it = Preconditioner_Matrix1.block(block, block-1).begin();
			end = Preconditioner_Matrix1.block(block, block-1).end();
			for (; it!=end; ++it)
			{
				temp.block(1,0).set(it->row(), it->column(), it->value());
			}

			temp.block(0,1) = 0;
			it = Preconditioner_Matrix1.block(block-1, block).begin();
			end = Preconditioner_Matrix1.block(block-1, block).end();
			for (; it!=end; ++it)
			{
				temp.block(0,1).set(it->row(), it->column(), it->value());
			}

			temp.block(1,1) = 0;
			it = Preconditioner_Matrix1.block(block, block).begin();
			end = Preconditioner_Matrix1.block(block, block).end();
			for (; it!=end; ++it)
			{
				temp.block(1,1).set(it->row(), it->column(), it->value());
			}

		}

		inverse_blocks[block].initialize(temp);
		deallog << "Done with block " << block +1 <<  std::endl;
		temp.clear();

	}

	deallog << "All preconditioner-blocks have been constructed." << std::endl;

}
**/
/**
template <>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::initialize( dealii::BlockSparseMatrix<double> * System_Matrix, dealii::BlockSparseMatrix<double> &Preconditioner_Matrix1, dealii::BlockSparseMatrix<double> &Preconditioner_Matrix2)
{
	SystemMatrix = System_Matrix;
	inverse_blocks.clear();
	for(unsigned int block = 0; block <System_Matrix->n_block_cols(); block++ ) {
		SparseDirectUMFPACK solver;
		inverse_blocks.push_back(solver);
	}
	// First Block. Prepare Solver directly.
	SparseDirectUMFPACK solver;
	inverse_blocks[0].initialize(Preconditioner_Matrix1.block(0,0));
	deallog << "Done with block " << 1 <<  std::endl;


	// Build local blocks
	if(System_Matrix->m() != System_Matrix->n()) {
		std::cout << "Critical Error in the Preconditioner. System Matrix block count mismatch!" << std::endl;
	}
	for(unsigned int block = 1; block <System_Matrix->n_block_cols(); block++ ) {
		dealii::BlockSparseMatrix<double> temp;
		BlockDynamicSparsityPattern tsp;
		tsp.reinit(2,2);

		if((block-1)%2 == 0) {
			tsp.block(0,0).reinit(Preconditioner_Matrix2.block(block-1, block-1).m(),Preconditioner_Matrix2.block(block-1, block-1).n() );
			SparseMatrix<double>::iterator it = Preconditioner_Matrix2.block(block-1, block-1).begin(),
			end = Preconditioner_Matrix2.block(block-1, block-1).end();
			for (; it!=end; ++it)
			{
				tsp.block(0,0).add(it->row(), it->column());
			}

			tsp.block(1,0).reinit(Preconditioner_Matrix2.block(block  , block-1).m(),Preconditioner_Matrix2.block(block  , block-1).n() );
			it = Preconditioner_Matrix2.block(block, block-1).begin();
			end = Preconditioner_Matrix2.block(block, block-1).end();
			for (; it!=end; ++it)
			{
				tsp.block(1,0).add(it->row(), it->column());
			}

			tsp.block(0,1).reinit(Preconditioner_Matrix2.block(block-1, block ).m(),Preconditioner_Matrix2.block(block-1, block  ).n() );
			it = Preconditioner_Matrix2.block(block-1, block).begin();
			end = Preconditioner_Matrix2.block(block-1, block).end();
			for (; it!=end; ++it)
			{
				tsp.block(0,1).add(it->row(), it->column());
			}

			tsp.block(1,1).reinit(Preconditioner_Matrix2.block(block  , block  ).m(),Preconditioner_Matrix2.block(block  , block  ).n() );
			it = Preconditioner_Matrix2.block(block, block).begin();
			end = Preconditioner_Matrix2.block(block, block).end();
			for (; it!=end; ++it)
			{
				tsp.block(1,1).add(it->row(), it->column());
			}

		} else {
			tsp.block(0,0).reinit(Preconditioner_Matrix1.block(block-1, block-1).m(),Preconditioner_Matrix1.block(block-1, block-1).n() );
			SparseMatrix<double>::iterator it = Preconditioner_Matrix1.block(block-1, block-1).begin(),
			end = Preconditioner_Matrix1.block(block-1, block-1).end();
			for (; it!=end; ++it)
			{
				tsp.block(0,0).add(it->row(), it->column());
			}

			tsp.block(1,0).reinit(Preconditioner_Matrix1.block(block  , block-1).m(),Preconditioner_Matrix1.block(block  , block-1).n() );
			it = Preconditioner_Matrix1.block(block, block-1).begin();
			end = Preconditioner_Matrix1.block(block, block-1).end();
			for (; it!=end; ++it)
			{
				tsp.block(1,0).add(it->row(), it->column());
			}

			tsp.block(0,1).reinit(Preconditioner_Matrix1.block(block-1, block  ).m(),Preconditioner_Matrix1.block(block-1, block  ).n() );
			it = Preconditioner_Matrix1.block(block-1, block).begin();
			end = Preconditioner_Matrix1.block(block-1, block).end();
			for (; it!=end; ++it)
			{
				tsp.block(0,1).add(it->row(), it->column());
			}

			tsp.block(1,1).reinit(Preconditioner_Matrix1.block(block  , block  ).m(),Preconditioner_Matrix1.block(block  , block  ).n() );
			it = Preconditioner_Matrix1.block(block, block).begin();
			end = Preconditioner_Matrix1.block(block, block).end();
			for (; it!=end; ++it)
			{
				tsp.block(1,1).add(it->row(), it->column());
			}
		}

		tsp.collect_sizes();
		BlockSparsityPattern dsp;
		dsp.copy_from(tsp);

		temp.reinit(dsp);

		if((block-1)%2 == 0) {

			temp.block(0,0) = 0;
			SparseMatrix<double>::iterator it = Preconditioner_Matrix2.block(block-1, block-1).begin(),
			end = Preconditioner_Matrix2.block(block-1, block-1).end();
			for (; it!=end; ++it)
			{
				temp.block(0,0).set(it->row(), it->column(), it->value());
			}

			temp.block(1,0) = 0;
			it = Preconditioner_Matrix2.block(block, block-1).begin();
			end = Preconditioner_Matrix2.block(block, block-1).end();
			for (; it!=end; ++it)
			{
				temp.block(1,0).set(it->row(), it->column(), it->value());
			}

			temp.block(0,1) = 0;
			it = Preconditioner_Matrix2.block(block-1, block).begin();
			end = Preconditioner_Matrix2.block(block-1, block).end();
			for (; it!=end; ++it)
			{
				temp.block(0,1).set(it->row(), it->column(), it->value());
			}

			temp.block(1,1) = 0;
			it = Preconditioner_Matrix2.block(block, block).begin();
			end = Preconditioner_Matrix2.block(block, block).end();
			for (; it!=end; ++it)
			{
				temp.block(1,1).set(it->row(), it->column(), it->value());
			}


		} else {

			temp.block(0,0) = 0;
			SparseMatrix<double>::iterator it = Preconditioner_Matrix1.block(block-1, block-1).begin(),
			end = Preconditioner_Matrix1.block(block-1, block-1).end();
			for (; it!=end; ++it)
			{
				temp.block(0,0).set(it->row(), it->column(), it->value());
			}


			temp.block(1,0) = 0;
			it = Preconditioner_Matrix1.block(block, block-1).begin();
			end = Preconditioner_Matrix1.block(block, block-1).end();
			for (; it!=end; ++it)
			{
				temp.block(1,0).set(it->row(), it->column(), it->value());
			}

			temp.block(0,1) = 0;
			it = Preconditioner_Matrix1.block(block-1, block).begin();
			end = Preconditioner_Matrix1.block(block-1, block).end();
			for (; it!=end; ++it)
			{
				temp.block(0,1).set(it->row(), it->column(), it->value());
			}

			temp.block(1,1) = 0;
			it = Preconditioner_Matrix1.block(block, block).begin();
			end = Preconditioner_Matrix1.block(block, block).end();
			for (; it!=end; ++it)
			{
				temp.block(1,1).set(it->row(), it->column(), it->value());
			}

		}

		inverse_blocks[block].initialize(temp);
		deallog << "Done with block " << block +1 <<  std::endl;
		temp.clear();

	}

	deallog << "All preconditioner-blocks have been constructed." << std::endl;
}
**/
/**
template<>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::vmult (  dealii::BlockVector<double> &  out_vec , dealii::BlockVector<double> & in_vec ) const {
	// deallog << "Starting VMULT" << std::endl;
	unsigned int Blocks = in_vec.n_blocks();
	// deallog << "Block 1 done." << std::endl;
	inverse_blocks[0].vmult(out_vec.block(0) , in_vec.block(0));

	for ( unsigned int i = 1; i<Blocks; i++) {
		// deallog << "Starting Block " << i+1 << std::endl;
		dealii::BlockVector<double> temp;
		temp.reinit(2);
		temp.block(0).reinit(in_vec.block(i-1).size());
		temp.block(1).reinit(in_vec.block(i).size());
		temp.collect_sizes();
		temp.block(0) = in_vec.block(i-1);
		temp.block(1) = in_vec.block(i);
		// deallog << "Input-vector prepared." << std::endl;

		dealii::BlockVector<double> temp2;
		temp2.reinit(2);
		temp2.block(0).reinit(in_vec.block(i-1).size());
		temp2.block(1).reinit(in_vec.block(i).size());
		temp2.collect_sizes();
		// deallog << "Output-vector prepared." << std::endl;

		inverse_blocks[i].vmult(temp2, temp);
		// deallog << "Solution calculated." << std::endl;
		out_vec.block(i) = temp2.block(1);

		// deallog << "Block " << i+1 << " done." << std::endl;
	}
}
**/
/**
template<>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::vmult (  dealii::BlockVector<double> &  out_vec , dealii::BlockVector<double> & in_vec ) const {
	// deallog << "Starting VMULT" << std::endl;

	unsigned int Blocks = in_vec.n_blocks();
	// deallog << "Block 1 done." << std::endl;

	out_vec = in_vec;
	dealii::Vector<double> temp;
	inverse_blocks[0].vmult(out_vec.block(0), in_vec.block(0));
	temp.reinit(in_vec.block(1).size());

	SystemMatrix->block(1,0).vmult(temp, out_vec.block(0));
	out_vec.block(1).add(-1.0, temp);

	for(unsigned int i = 1; i< Blocks-1; i++) {
		dealii::BlockVector<double> temp;
		Hinv(i,temp, out_vec);
		dealii::Vector<double> temp2;
		temp2.reinit(in_vec.block(i+1).size());
		SystemMatrix->block(i+1, i).vmult(temp2,temp.block(1));
		out_vec.block(i+1).add(-1.0, temp2);
	}

	for(unsigned int i = 1; i< Blocks; i++) {
		dealii::BlockVector<double> temp2;
		Hinv(i, temp2, out_vec);
		out_vec.block(i) = temp2.block(1);
	}

	for(unsigned int i = Blocks - 2; i> 0; i--) {
		dealii::BlockVector<double> temp3;
		temp3.reinit(Blocks);
		for(unsigned int j = 0; j < Blocks; j++) {
			temp3.block(j).reinit(in_vec.block(j).size());
		}
		temp3.collect_sizes();

		for(unsigned int j = 0; j < Blocks; j++) {
			temp3.block(j) = 0;
		}

		SystemMatrix->block(i, i+1).vmult(temp3.block(i),out_vec.block(i+1));

		dealii::BlockVector<double> temp;
		Hinv(i,temp, temp3);

		out_vec.block(i).add(-1.0, temp.block(1));
	}

	dealii::Vector<double> temp3;
	temp3.reinit(out_vec.block(0).size());
	SystemMatrix->block(0, 1).vmult(temp3,out_vec.block(1));
	dealii::Vector<double> temp4;
	temp4.reinit(out_vec.block(0).size());
	inverse_blocks[0].vmult(temp4 , temp3);
	out_vec.block(0).add(-1.0, temp4);

}

template<>
void PreconditionSweeping<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double>>::Tvmult (  dealii::BlockVector<double> &  out_vec ,dealii::BlockVector<double> & in_vec ) const {
	vmult (  out_vec ,  in_vec );
}

**/
#endif
