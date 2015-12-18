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
				 * unless specified differently we want complete preconditioner weighing which means \f$\alpha = 1.0\f$
				 */
				AdditionalData (const double in_alpha , QGauss<3> & in_qf, FESystem<3> & in_fe, DoFHandler<3> & in_dofh, WaveguideStructure & in_structure);

				/**
				 * This constructor is the same as the other but it doesn't take a value for \f$\alpha\f$. Instead \f$\alpha\f$ is set to 1.0.
				 */
				AdditionalData ( QGauss<3> & in_qf, FESystem<3> & in_fe, DoFHandler<3> & in_dofh, WaveguideStructure & in_structure);
				/**
				 * This member stores the value of alpha as set in the constructor call. See page 3777 in the afore mentioned paper about this preconditioner for a more detailed description.
				 */

				double 				alpha;
				/**
				 * This is the same quadrature formula as used in the Waveguide class and it is required since similar matrices have to be assembled.
				 */
				QGauss<3> 			quadrature_formula;

				/**
				 * This member stores the same finite element as the identical member in the Waveguide class. The reasoning is the same as for the quadrature formula.
				 */
				FESystem<3>			fe;

				/**
				 * This member stores the same DOF-handler as the identical member in the Waveguide class. The reasoning is the same as for the quadrature formula.
				 */
				DoFHandler<3>		dof_handler;

				/**
				 * In order to calculate transformation tensors in the assembly method it is important to have a handle to the structure of the Waveguide.
				 */
				WaveguideStructure	structure;
		};

		PreconditionSweeping();


		/**
		 * This function generates all the inverse matrices for the blocks. It remains to be seen if this is really necessary for all the purely internal blocks which have similar structure.
		 * \param matrix This parameter contains a reference to the Systemmatrix which gives all the information for the matrices. Maybe it will become necessary to also handle a separate matrix with the additional blocks i.e. the Information about the neighboring blocks being filled with the sweeping PML.
		 * \param parameters This argument can be used to set the value of \f$\alpha\f$ to be used later in the computation.
		 */

		void initialize ( MatrixType & matrix,  const AdditionalData & parameters);

		/**
		 * Need to figure out
		 */
		void vmult (VectorType &, const VectorType &);

		/**
		 * Need to figure out
		 */

		void Tvmult (VectorType &, const VectorType &);

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

		/**
		 * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the x-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
		 * \param position Stores the position in which to test for presence of a PML-Material.
		 */
		bool	PML_in_X(Point<3> & position);
		/**
		 * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the y-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
		 * \param position Stores the position in which to test for presence of a PML-Material.
		 */
		bool	PML_in_Y(Point<3> & position);
		/**
		 * This function is used to determine, if a system-coordinate belongs to a PML-region for the PML that limits the computational domain along the z-axis. Since there are 3 blocks of PML-type material, there are 3 functions.
		 * \param position Stores the position in which to test for presence of a PML-Material.
		 * \param block The preconditioner builds several matrices which differ for each Sector. In order to compute such a matrix it regards the sector itself and the sector before with the exception of the first sector (for this case only the first sector is regarded. I.e. to build the fifth preconditioner block, the assembly function regards the sectors 4 and 5. In the block \f$ i \f$ we have PML for the lower ( \f$ z \f$ small) end of sector \f$ i-1 \f$ and the higher end of sector \f$ i \f$.
		 */
		bool	PML_in_Z(Point<3> & position, unsigned int block);

		/**
		 * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
		 * \param position Stores the position from which to calculate the distance to the PML-surface.
		 */
		double 	PML_X_Distance(Point<3> & position);
		/**
		 * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
		 * \param position Stores the position from which to calculate the distance to the PML-surface.
		 */
		double 	PML_Y_Distance(Point<3> & position);
		/**
		 * This function calculates for a given point, its distance to a PML-boundary limiting the computational domain. This function is used merely to make code more readable. There is a function for every one of the dimensions since the normal vectors of PML-regions in this implementation are the coordinate-axis. This value is set to zero outside the PML and positive inside both PML-domains (only one for the z-direction).
		 * \param position Stores the position from which to calculate the distance to the PML-surface.
		 * \param block The preconditioner builds several matrices which differ for each Sector. In order to compute such a matrix it regards the sector itself and the sector before with the exception of the first sector (for this case only the first sector is regarded. I.e. to build the fifth preconditioner block, the assembly function regards the sectors 4 and 5. In the block \f$ i \f$ we have PML for the lower ( \f$ z \f$ small) end of sector \f$ i-1 \f$ and the higher end of sector \f$ i \f$.
		 */
		double 	PML_Z_Distance(Point<3> & position, unsigned int block );

		/**
		 * This function calculates the complex conjugate of every vector entry and returns the result in a copy. This function does not operate in place - it operates on a copy and hence returns a new object.
		 */
		Tensor<1,3, std::complex<double>> Conjugate_Vector(Tensor<1,3, std::complex<double>> input);

		/**
		 * In order to keep the code readable, this function is introduced to encapsulate all the calls to transformation-optics- and PML-related functions. In the assembly of the preconditioner matrices this function is called to get the complex 3x3 matrix \f$\boldsymbol{\mu}\f$ and \f$\boldsymbol{\epsilon}\f$.
		 * \param point Since both the PML and the transformation-tensor are position-dependent, this has to be specified.
		 * \param inverse In Maxwell's equations (written as a second order PDE) we need the inverse of one of the Tensors (either \f$\boldsymbol{\mu}\f$ or \f$\boldsymbol{\epsilon}\f$). This can be achieved by setting this flag.
		 * \param epsilon The general computation of both \f$\boldsymbol{\mu}\f$ and \f$\boldsymbol{\epsilon}\f$ is the same so we use the same function. If this parameter is set to true, \f$\boldsymbol{\epsilon}\f$ will be returned.
		 * \param block Considering the PML it makes a difference for which block a material tensor is supposed to be used. The same coordinate can be in a PML region for one calculation and outside of it for another resulting in different tensors. For this reason the block under investigation has to be passed as an argument to this function.
		 */
		Tensor<2,3, std::complex<double>> get_Tensor(Point<3> & point, bool inverse, bool epsilon, int block);

	private:

		const double width;

		const double l;

		double alpha;

		size_type n_rows;

		size_type n_columns;

		const unsigned int Sectors;

		std::vector<dealii::SparseDirectUMFPACK> inverse_blocks;

		PreconditionSweeping<MatrixType, VectorType>::AdditionalData data;
};

#endif
