#ifndef PRECONDITIONERSWEEPING_H_
#define PRECONDITIONERSWEEPING_H_

using namespace dealii;
#include <deal.II/base/config.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/vector.h>

static SolverControl s(10,1.e-10, false, false);
dealii::TrilinosWrappers::SolverDirect * solver;
// dealii::TrilinosWrappers::SolverDirect * solver;


/**
 * \class PreconditionerSweeping
 * \brief This class implements the DealII preconditioner interface and offers a sweeping preconditioning mechanism.
 *
 * Details can be found in the paper <a href="http://www.sciencedirect.com/science/article/pii/S0021999112000460">A sweeping preconditioner for time-harmonic Maxwell’s equations with finite elements</a>. The general idea is as follows:
 * Let \f$\Omega\f$ be the computational domain (internally) truncated by an absorbing boundary condition. This domain can be split into  layers along a direction (in our case \f$ z \f$  and triangulated. We therefore have a triangulation spread across multiple processes. We chose the splitting such, that the degrees of freedom are ordered process-wise. Let \f$ K \f$ be the number of Layers and \f$ T_i \quad \imath \in \{1,\ldots,K\} \f$ the parts of the triangulation.
 * For a PML function
 * \f[
 * \sigma_i (\xi) = \begin{cases} \theta \left(\frac{-1 +(i-1)l - \xi}{l} \right)^2 \quad &\xi \in [-1 + (i-2)l, -1 +(i-1)l] \\ 0,& \xi \in [-1 + (i-1)l, 1-l] \\ \Theta\left(\frac{\xi -1 +l}{l}\right)^2, & \xi \in [1-l,1]\end{cases}
 * \f]
 * we now regard the problem
 * \f[
 * \nabla \times \tilde{\mu}_{r,i}^{-1}\nabla \times \boldsymbol{E} - \kappa^2\tilde{\epsilon}_{r,i}\boldsymbol{E} = 0 \quad \text{ in } \mathrm{int} (\mathrm{T}_{i-1}\cup\mathrm{T}_i
 * \f]
 * where
 *
 * Described in words: We put a PML into the neighboring block of the block we want to precondition and setup the system matrix for this smaller problem. This matrix we can then invert and name the Operator \f$H_i^{-1}\f$. We then define the operator
 * \f[
 * S(\boldsymbol{v}) = P_{0,n_i}H_i^{-1} (\boldsymbol{v} , \boldsymbol{0}))
 * \f]
 * where \f$ P_{0,n_i} \f$ describes the extraction of the first \f$ n_i \f$ components. For the one block which has no neighbor the inverse of the block of the system matrix can be used. The inversion does not have to be performed numerically - a decomposition (performed by UMFPACK or MUMPS) is sufficient.
 *
 *  \date 28.11.2016
 *  \author Pascal Kraft
 **/
class PreconditionerSweeping : TrilinosWrappers::PreconditionBase
  {

  using dealii::TrilinosWrappers::PreconditionBase::vmult;

  public:

  /**
   * This constructor is the only one that should be used at this time.
   * \param in_own This is the number of degrees of freedom that the current process has to deal with (owned).
   * \param in_others This is the number of degrees of freedom that the process below has. Every process has to deal with one other process. The other neighbor only contacts it for a multiplication with its own matrix block - in this case, no objects of unknown size are concerned. However: for the one process that does require more contact needs a vector tp be initialized. This vectors size is this int.
   * \param bandwidth The number of dofs per line on average is required for the construction of matrices.
   * \param locally_owned The degrees of freedom associated with the current process. Required for vector and matrix construction.
   */
	PreconditionerSweeping ( int in_own, int in_others, int bandwidth, IndexSet locally_owned);

    ~PreconditionerSweeping ();

    /**
     * For the application of the preconditioner we require the application of the inverse of \f$H\f$. This is implemented in this function. (The mathematical usage is included in lines 2, 6 and 13 and indirectly in every use of the Operator \f$S\f$.
     * \param src This is the vector to be multiplied by \f$H_i^{-1}\f$.
     * \param dst This is the vector to store the result in.
     */
    void Hinv(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;
        
    /**
     * Cases in which we require multiplications with \f$A(E_{i+1}, E_i)\f$, are where this function is used. See algorithm lines 2 and 4.
     * \param src This is the vector to be multiplied by  \f$A(E_{i+1}, E_i)\f$.
     * \param dst This is the vector to store the result in.
     */
    void LowerProduct(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;

    /**
	 * Cases in which we require multiplications with \f$A(E_i, E_{i+1})\f$, are where this function is used. See algorithm lines 11 and 13.
	 * \param src This is the vector to be multiplied by  \f$A(E_i, E_{i+1})\f$.
	 * \param dst This is the vector to store the result in.
	 */
	void UpperProduct(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;

	/**
	 * In order to be called by the iterative solver, this function has to be overloaded. It gets called from GMRES and is the core function which contains the implementation. For a description of the interface, see the implementation in the base class.
	 * \param dst The vector to store the result in.
	 * \param src The vector to be multiplied by the approximate inverse.
	 */
	virtual void vmult (TrilinosWrappers::MPI::BlockVector       &dst,      const TrilinosWrappers::MPI::BlockVector &src) const;

	TrilinosWrappers::SparseMatrix * matrix;
	TrilinosWrappers::SparseMatrix * prec_matrix_lower, * prec_matrix_upper;

	void Prepare(TrilinosWrappers::MPI::BlockVector &src);

	void init(SolverControl in_sc);

  private:
	int * indices;
	int own, others;
	TrilinosWrappers::MPI::Vector itmp, otmp;
	Vector<double> boundary;
	unsigned int sweepable;
	IndexSet locally_owned_dofs;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
