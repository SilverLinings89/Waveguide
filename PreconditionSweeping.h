#ifndef PreconditionFlag
#define PreconditionFlag

#include <deal.II/base/subscriptor.h>

using namespace dealii;

/**
 * This Class encapsulates the Sweeping Preconditioner functionality as prposed by Tsuji, Engquist and Ying in the paper "A sweeping preconditioner for time-harmonic Maxwell's equations" (See <a href="http://www.sciencedirect.com/science/article/pii/S0021999112000460">here</a>).
 * It is currently under development and *not yet* functional.
 * \author Pascal Kraft
 * \date 9.12.2015
 */
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
			 * unless specified differently we want complete preconditioner weighing which means \f$\alpha = 1.0\f$
			 */
				AdditionalData (const double alpha = 1.);

				/**
				 * This member stores the value of alpha as set in the constructor call. See page 3777 in the afore mentioned paper about this preconditioner for a more detailed description.
				 */
				double alpha;
		};

		PreconditionSweeping();


		/**
		 * This function generates all the inverse matrices for the blocks. It remains to be seen if this is really necessary for all the purely internal blocks which have similar structure.
		 * \param matrix This parameter contains a reference to the Systemmatrix which gives all the information for the matrices. Maybe it will become necessary to also handle a separate matrix with the additional blocks i.e. the Information about the neighboring blocks being filled with the sweeping PML.
		 * \param parameters This argument can be used to set the value of \f$\alpha\f$ to be used later in the computation.
		 */
		template <typename MatrixType>
		void initialize (const MatrixType &matrix, const AdditionalData &parameters);

		/**
		 * Need to figure out
		 */
		template<class VectorType>
		void vmult (VectorType &, const VectorType &) const;

		/**
		 * Need to figure out
		 */
		template<class VectorType>
		void Tvmult (VectorType &, const VectorType &) const;

		/**
		 * Need to figure out
		 */
		template<class VectorType>
		void vmult_add (VectorType &, const VectorType &) const;

		/**
		 * Need to figure out
		 */
		template<class VectorType>
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
		double alpha;

		size_type n_rows;

		size_type n_columns;
};





