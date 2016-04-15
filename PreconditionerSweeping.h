/*
 * PreconditionerSweepingPetscParallel.h
 *
 *  Created on: 19.02.2016
 *      Author: kraft
 */

#ifndef PRECONDITIONERSWEEPING_H_
#define PRECONDITIONERSWEEPING_H_

using namespace dealii;
#include <deal.II/base/config.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/petsc_precondition.h>
#include <petscpc.h>



class PreconditionerSweeping : public PETScWrappers::PreconditionerBase
  {
  public:
    /**
     * Standardized data struct to pipe additional flags to the
     * preconditioner.
     */
    struct AdditionalData
    {
    	Mat matrix;
    	unsigned int sub;
    	unsigned int self;
    	int total_rows;
    	int total_cols;
    	int entries;
    	int lowest;
    	int highest;
    	int * positions;
    	Vec	src;
    	Vec	dst;
    	KSP ksp;
    };

    /**
     * Empty Constructor. You need to call initialize() before using this
     * object.
     */
    PreconditionerSweeping ();

    /**
     * Constructor. Take the matrix which is used to form the preconditioner,
     * and additional flags if there are any.
     */
    PreconditionerSweeping (const dealii::PETScWrappers::MatrixBase     &matrix, int in_lowest, int in_highest);

    /**
     * Same as above but without setting a matrix to form the preconditioner.
     * Intended to be used with SLEPc objects.
     */
    PreconditionerSweeping (const MPI_Comm communicator);


    /**
     * Initializes the preconditioner object and calculate all data that is
     * necessary for applying it in a solver. This function is automatically
     * called when calling the constructor with the same arguments and is only
     * used if you create the preconditioner without arguments.
     */
    void initialize (const dealii::PETScWrappers::MatrixBase     &matrix);

    void vmult (PETScWrappers::VectorBase       &dst,
                    const PETScWrappers::VectorBase &src) const;

    const PC &get_pc () const;



  protected:
    /**
     * Store a copy of the flags for this particular preconditioner.
     */
    AdditionalData additional_data;

    PC pc;

    void create_pc ();
    /**
     * Initializes the preconditioner object without knowing a particular
     * matrix. This function sets up appropriate parameters to the underlying
     * PETSc object after it has been created.
     */

    KSP ksp;
  };

#endif /* PRECONDITIONERSWEEPING_H_ */
