#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_solver.h>
#include "PreconditionerSweeping.h"

using namespace dealii;


PreconditionerSweeping::PreconditionerSweeping (const TrilinosWrappers::SparseMatrix  &S, int in_own, int in_others)
      :
      preconditioner_matrix     (&S),
	  input(2),
	  output(2)
    {
		own = in_own;
		others = in_others;
		itmp = TrilinosWrappers::MPI::Vector(complete_index_set(own + others));
		otmp = TrilinosWrappers::MPI::Vector(complete_index_set(own + others));
		input.block(0).reinit(complete_index_set(others), MPI_COMM_SELF);
		input.block(1).reinit(complete_index_set(own), MPI_COMM_SELF);
		output.block(0).reinit(complete_index_set(others), MPI_COMM_SELF);
		output.block(1).reinit(complete_index_set(own), MPI_COMM_SELF);
	}

void PreconditionerSweeping::vmult (TrilinosWrappers::MPI::Vector       &dst,
			const TrilinosWrappers::MPI::Vector &src) const
{
	for(int i = 0; i < own; i++) {
		input.block(1).se[i] = src[i];
	}
	const TrilinosWrappers::MPI::Vector inp(input);
	SolverControl solver_control(5000, 1e-6 * src.l2_norm());
	TrilinosWrappers::SolverDirect solver(solver_control, TrilinosWrappers::SolverDirect::AdditionalData(true, "Amesos_Umfpack"));
	solver.solve(*preconditioner_matrix, output, input);

	dst.reinit(output.block(1));

}



















/**
PetscErrorCode SampleShellPCApply(PC pc,Vec x,Vec y)
  {
	std::cout << "Beginning Application of Precondtioner:" << std::endl;
	PreconditionerSweeping::AdditionalData  *shell;
    PetscErrorCode ierr;
    ierr = PCShellGetContext(pc,(void**)&shell);CHKERRQ(ierr);
    VecCreate(PETSC_COMM_SELF, & shell->src);
    VecSetType(shell->src, VECSEQ);
    VecSetSizes( shell->src, shell->total_rows, shell->total_rows);
    VecSetSizes( shell->dst, shell->total_rows, shell->total_rows);
    VecSet(shell->src, 0.0 );
    VecSet(shell->dst, 0.0 );
    std::cout << GlobalParams.MPI_Rank <<": c" <<std::endl;
	PetscScalar * vals;
	vals = new PetscScalar[shell->entries];

	VecLockPop(x);
	VecLockPop(x);
	VecGetArray( x, & vals );

	for( int i = 0; i < shell->entries; i++) {
		VecSetValue( shell->src, shell->total_rows - shell->entries + i, vals[ i], INSERT_VALUES);
	}

	VecRestoreArray(x, &vals);
	std::cout << GlobalParams.MPI_Rank <<": d" <<std::endl;
	VecAssemblyBegin(shell->src);
	VecAssemblyEnd(shell->src);

	std::cout << "This is process number " << GlobalParams.MPI_Rank << ". I have " << shell->entries << " Entries from " << shell->lowest << " to " << shell->highest <<". Local from " << shell->total_rows-shell->entries << " until " << shell->total_rows <<std::endl;

	int src_len, dst_len;
	VecGetSize(shell->dst, & dst_len);
	VecGetSize(shell->src, & src_len);
	std::cout << "This is process number " << GlobalParams.MPI_Rank << ". My vectors have sizes: " << src_len << " and " << dst_len<<std::endl;

	VecGetSize(x, & dst_len);
	VecGetSize(y, & src_len);
	std::cout << "This is process number " << GlobalParams.MPI_Rank << ". Global vectors have sizes: " << src_len << " and " << dst_len<<std::endl;


	MatGetSize(shell->matrix, & src_len, & dst_len );
	std::cout << "This is process number " << GlobalParams.MPI_Rank << ". My matrix sizes: " << src_len << " and " << dst_len<<std::endl;

	ierr = KSPSolve(shell->ksp, shell->src, shell->dst);
	std::cout << "e" <<std::endl;

	VecLockPush(x);
	VecLockPush(x);
	std::cout << GlobalParams.MPI_Rank <<": f" <<std::endl;
	VecAssemblyBegin(shell->dst);
	VecAssemblyEnd(shell->dst);
	PetscScalar * target;
	VecGetArray( shell->dst , &target);

	for(int i = 0; i < shell->entries; i++) {
		VecSetValue(y, shell->lowest + i, target[shell->total_rows - shell->entries + i], INSERT_VALUES);
	}

	VecRestoreArray(shell->dst, &target);

	std::cout << GlobalParams.MPI_Rank <<": done"<<std::endl;

    return 0;
  }

  PetscErrorCode SampleShellPCDestroy(PC pc)
  {
	  PreconditionerSweeping::AdditionalData  *shell;
    PetscErrorCode ierr;

    ierr = PCShellGetContext(pc,(void**)&shell);CHKERRQ(ierr);
    ierr = VecDestroy(&shell->src);CHKERRQ(ierr);
    ierr = VecDestroy(&shell->dst);CHKERRQ(ierr);

    return 0;
  }

PreconditionerSweeping::PreconditionerSweeping (const MPI_Comm comm)
  {
    additional_data = {};

    int ierr = PCCreate(comm, &pc);
    AssertThrow (ierr == 0, ExcPETScError(ierr));


  }


PreconditionerSweeping::PreconditionerSweeping ()
  {}


PreconditionerSweeping::PreconditionerSweeping (const dealii::PETScWrappers::MatrixBase     &matrix, int in_sub_lowest, int in_lowest, int in_highest)
  {
	additional_data = {};
	additional_data.sub_lowest = in_sub_lowest;
	additional_data.lowest = in_lowest;
	additional_data.highest = in_highest;
    initialize(matrix);
  }

  void PreconditionerSweeping::vmult(PETScWrappers::VectorBase &dst, const PETScWrappers::VectorBase &src)const {
	  std::cout << "AALSDFJAHSDF" << std::endl;
	 AssertThrow (ksp != NULL, StandardExceptions::ExcInvalidState ());
     int ierr;
	 ierr = KSPSolve(ksp, src, dst);
	 AssertThrow (ierr == 0, ExcPETScError(ierr));
  }

  void PreconditionerSweeping::create_pc() {

	  PetscErrorCode ierr = 0;

	  ierr = PCCreate(PETSC_COMM_WORLD, &pc);

	  ierr = PCSetType(pc,PCSHELL);

	  ierr = PCShellSetApply(pc,SampleShellPCApply);

	  ierr = PCShellSetContext(pc,& additional_data);




  }

  const PC & PreconditionerSweeping::get_pc() {
	  std::cout << "get pc" << std::endl;
	  return pc;
  }

void  PreconditionerSweeping::initialize (const PETScWrappers::MatrixBase     &matrix_)
  {

	std::cout << "Initializing Preconditioner..." <<std::endl;
	Mat matrix = static_cast<Mat>(matrix_);
    int       *indices;
    IS             is;

    indices = new int[additional_data.highest-additional_data.sub_lowest+1];
    for(int i = 0; i < additional_data.highest-additional_data.sub_lowest +1; i++) {
    	indices[i] = additional_data.sub_lowest + i;
    }
    MatSetOption(matrix, MAT_SYMMETRIC, PETSC_FALSE);

    ISCreateGeneral(PETSC_COMM_SELF,additional_data.highest-additional_data.sub_lowest +1,indices,PETSC_COPY_VALUES,&is);
    MatAssemblyBegin(matrix , MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix , MAT_FINAL_ASSEMBLY);
    MatCreate(MPI_COMM_SELF, &additional_data.matrix);
    MatSetSizes(additional_data.matrix, additional_data.highest-additional_data.sub_lowest +1, additional_data.highest-additional_data.sub_lowest +1, additional_data.highest-additional_data.sub_lowest +1, additional_data.highest-additional_data.sub_lowest +1);
    MatSetType(additional_data.matrix, MATSEQMAIJ);
    MatGetSubMatrix(matrix, is , is, MAT_INITIAL_MATRIX, &additional_data.matrix);
    MatAssemblyBegin(additional_data.matrix , MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(additional_data.matrix , MAT_FINAL_ASSEMBLY);

    PetscViewer    viewer;
    PetscViewerDrawOpen(PETSC_COMM_SELF,NULL,NULL,0,0,1000,1000,&viewer);
    PetscObjectSetName((PetscObject)viewer,"Matrix Structure");
    PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_BASIC);
    MatView(additional_data.matrix,viewer);

    VecCreate(PETSC_COMM_SELF, & additional_data.src);
    VecCreate(PETSC_COMM_SELF, & additional_data.dst);
    VecSetType(additional_data.src, VECMPI);
    VecSetType(additional_data.dst, VECMPI);
    MatGetSize(additional_data.matrix, &additional_data.total_rows, &additional_data.total_cols );
    VecSetSizes( additional_data.src, additional_data.total_rows, additional_data.total_rows);
    VecSetSizes( additional_data.dst, additional_data.total_rows, additional_data.total_rows);
    VecSet( additional_data.src, 0.0);
    VecSet( additional_data.dst, 0.0);

    KSPCreate(PETSC_COMM_SELF,& additional_data.ksp);
    KSPSetOperators(additional_data.ksp,additional_data.matrix,additional_data.matrix);
    KSPSetType(additional_data.ksp,KSPPREONLY);

    KSPGetPC(additional_data.ksp,&Temppc);
    PCSetType(Temppc,PCLU);
    PCFactorSetMatSolverPackage(Temppc,MATSOLVERUMFPACK);
    PCFactorSetUpMatSolverPackage(Temppc);
    additional_data.entries = additional_data.highest - additional_data.lowest +1;
    additional_data.positions = new int[additional_data.entries ];
   	for(unsigned int i = 0; i < additional_data.entries; i++) {
   		additional_data.positions[ i] = additional_data.lowest + i;
   	}
    create_pc();

    std::cout << "Result of Preparation: highest: " << additional_data.highest << " lowest:"<< additional_data.lowest<< " Number of Rows:" << additional_data.total_rows << " Number of cols:" << additional_data.total_cols<<std::endl;
  }


**/
