#ifndef PreconditionFlag
#define PreconditionFlag

#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>

using namespace dealii;

/**
 * This Class encapsulates the Sweeping Preconditioner functionality as prposed by Tsuji, Engquist and Ying in the paper "A sweeping preconditioner for time-harmonic Maxwell's equations" (See <a href="http://www.sciencedirect.com/science/article/pii/S0021999112000460">here</a>).
 * It is currently under development and *not yet* functional.
 * \author Pascal Kraft
 * \date 9.12.2015
 */
template< typename MatrixType, typename VectorType>
class PreconditionSweeping : public Subscriptor
   {

	public:
		/**
		 * This member stores the index type which is needed to write loops that generate nor warnings etc.
		 */
		typedef types::global_dof_index size_type;

		/**
		 * This class as usual for Dealii Preconditioners, stores data which will be used to finetune the Algorithm. This is mainly the parameter \f$\alpha\f$ in the paper which is a weighing parameter.
		 * \author Pascal Kraft
		 * \date 9.12.2015
		 */
		class AdditionalData
		{
			public:

				/**
				 * This constructor is the same as the other but it doesn't take a value for \f$\alpha\f$. Instead \f$\alpha\f$ is set to 1.0.
				 */
				AdditionalData (  double in_alpha, unsigned int in_nonzero);
				/**
				 * This member stores the value of alpha as set in the constructor call. See page 3777 in the afore mentioned paper about this preconditioner for a more detailed description.
				 */
				void SetNonZero (unsigned int nonzero);

				double 				alpha;

				unsigned int 				nonzero;
		};

		/**
		 * This is the normal constructor for preconditioners. It uses an object of the internal AdditionalData type specifiying the data it requires.
		 * \param data This parameter is a reference to an object of type PreconditionSweeping<MatrixType, VectorType>::AdditionalData. It holds two members alpha and nonzero which give the dof_handlers result for max_couplings and a preconditioner property for this special preconditioner which is a kind of weighting and is currently always set to 1.0.
		 */
		PreconditionSweeping( PreconditionSweeping<MatrixType, VectorType>::AdditionalData & data);


		/**
		 * This function generates all the inverse matrices for the blocks. It remains to be seen if this is really necessary for all the purely internal blocks which have similar structure.
		 * \param matrix This parameter contains a reference to the Systemmatrix which gives all the information for the matrices. Maybe it will become necessary to also handle a separate matrix with the additional blocks i.e. the Information about the neighboring blocks being filled with the sweeping PML.
		 * \param PML_1 This matrix holds the first half of the blocks needed for the construction of the preconditioner.
		 * \param PML_2 This matrix holds the second half of the blocks needed for the construction of the preconditioner.
		 */

		void initialize ( MatrixType * matrix, MatrixType & PML_1, MatrixType & PML_2  ) ;

		/**
		 * Need to figure out
		 */
		void vmult (VectorType &, VectorType &) const;

		void Hinv(unsigned int block, VectorType &out_vec, VectorType &in_vec ) const;

		/**
		 * Need to figure out
		 */

		void Tvmult (VectorType &,  VectorType &) const;

		/**
		 * Need to figure out
		 */
		void vmult_add (VectorType &, const VectorType &) const;

		/**
		 * Need to figure out
		 */
		void Tvmult_add (VectorType &, const VectorType &) const;

		/**
		 * Need to figure out
		 */
		void clear () {}

		/**
		 * Need to figure out
		 */
		size_type m () const;

		/**
		 * Need to figure out
		 */
		size_type n () const;


	private:

		unsigned int Sectors;

		std::vector<dealii::SparseDirectUMFPACK> inverse_blocks;

		PreconditionSweeping<MatrixType, VectorType>::AdditionalData data;

		MatrixType *SystemMatrix;
};

#endif
