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


  public:
	PreconditionerSweeping ( int in_own, int in_others, int bandwidth, IndexSet locally_owned,  int in_upper, ConstraintMatrix * in_cm);

    ~PreconditionerSweeping ();

    void Hinv(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;
        
    void LowerProduct(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;

    void UpperProduct(const dealii::Vector<double> &src, dealii::Vector<double> &dst) const ;

	virtual void vmult (TrilinosWrappers::MPI::BlockVector       &dst,      const TrilinosWrappers::MPI::BlockVector &src) const;

	TrilinosWrappers::SparseMatrix matrix, prec_matrix_lower, prec_matrix_upper;

	void Prepare(TrilinosWrappers::MPI::BlockVector &src);

  private:
	int * indices;
	int own, others, upper;
	TrilinosWrappers::MPI::Vector itmp, otmp;
	ConstraintMatrix * cm;
	Vector<double> boundary;
	unsigned int sweepable;
	IndexSet locally_owned_dofs;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
