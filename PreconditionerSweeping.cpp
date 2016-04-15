#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/petsc_precondition.h>
#include "PreconditionerSweeping.h"
#include <petscksp.h>

using namespace dealii;

PetscErrorCode SampleShellPCApply(PC pc,Vec x,Vec y)
  {
	std::cout << "Beginning Application of Precondtioner:" << std::endl;
	PreconditionerSweeping::AdditionalData  *shell;
    PetscErrorCode ierr;

    ierr = PCShellGetContext(pc,(void**)&shell);CHKERRQ(ierr);
    if(shell->lowest == -1 || shell->highest == -1) {
    	std::cout << "Determine Size" << std::endl;
    	VecGetOwnershipRange(x, &(shell->lowest), &(shell->highest));
    	shell->entries = shell->highest - shell->lowest;
    	shell->positions = new int[shell->entries ];

    	for(unsigned int i = 0; i < shell->entries; i++) {
    		shell->positions[ i] = shell->total_rows - shell->entries + i;
    	}

    }
    std::cout << "Entries:" << shell->entries << std::endl;
    VecSet(shell->src, 0.0 );

	PetscScalar * vals;
	VecGetArray( x, &vals);

	VecSetValues(shell->src, shell->entries, shell->positions, vals, ADD_VALUES);

	ierr = KSPSolve(shell->ksp, shell->src, shell->dst);


	PetscScalar * target;
	VecGetArray( y, &target);
	for(int i = 0; i < shell->entries; i++) {
		target[i] = vals[shell->total_rows - shell->entries + i];
	}

	std::cout << "done"<<std::endl;

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


PreconditionerSweeping::PreconditionerSweeping (const dealii::PETScWrappers::MatrixBase     &matrix, int in_lowest, int in_highest)
  {
	additional_data = {};
	additional_data.lowest = in_lowest;
	additional_data.highest = in_highest;
    initialize(matrix);
  }

  void PreconditionerSweeping::vmult(PETScWrappers::VectorBase &dst, const PETScWrappers::VectorBase &src)const {
	 AssertThrow (ksp != NULL, StandardExceptions::ExcInvalidState ());
     int ierr;
	 ierr = KSPSolve(ksp, src, dst);
	 AssertThrow (ierr == 0, ExcPETScError(ierr));
  }

  void PreconditionerSweeping::create_pc() {

	  MPI_Comm comm;
	  PetscErrorCode ierr;
	  PCCreate(PETSC_COMM_SELF, &pc);
	  PCSetType(pc,PCSHELL);

	  ierr = PCShellSetApply(pc,SampleShellPCApply);
	  ierr = PCShellSetContext(pc,& additional_data);

	  /* (Optional) Set user-defined function to free objects used by custom preconditioner */
	  ierr = PCShellSetDestroy(pc,SampleShellPCDestroy);

	  /* (Optional) Set a name for the preconditioner, used for PCView() */
	  ierr = PCShellSetName(pc,"Sweeping Preconditioner");


  }

const PC & PreconditionerSweeping::get_pc () const
{
	std::cout<< "Attempt" <<std::endl;
	return pc;
}

void  PreconditionerSweeping::initialize (const PETScWrappers::MatrixBase     &matrix_)
  {

	std::cout << "Initializing Preconditioner..." <<std::endl;
    additional_data.matrix = static_cast<Mat>(matrix_);
    PetscViewer    viewer;
    PetscViewerDrawOpen(PETSC_COMM_SELF,NULL,NULL,0,0,1000,1000,&viewer);
    PetscObjectSetName((PetscObject)viewer,"Matrix Structure");
    PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG);
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
    PC Temppc;
    KSPGetPC(additional_data.ksp,&Temppc);
    PCSetType(Temppc,PCLU);
    PCFactorSetMatSolverPackage(Temppc,MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(Temppc);

    additional_data.entries = additional_data.highest - additional_data.lowest;
    additional_data.positions = new int[additional_data.entries ];
   	for(unsigned int i = 0; i < additional_data.entries; i++) {
   		additional_data.positions[ i] = additional_data.total_rows - additional_data.entries + i;
   	}
    create_pc();

    std::cout << "Result of Preparation: highest: " << additional_data.highest << " lowest:"<< additional_data.lowest<< " Number of Rows:" << additional_data.total_rows << " Number of cols:" << additional_data.total_cols<<std::endl;
  }


