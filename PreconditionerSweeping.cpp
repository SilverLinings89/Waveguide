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
    std::cout << "a" <<std::endl;
    ierr = PCShellGetContext(pc,(void**)&shell);CHKERRQ(ierr);
    if(shell->lowest == -1 || shell->highest == -1) {
    	std::cout << "Determine Size" << std::endl;
    	VecGetOwnershipRange(x, &(shell->lowest), &(shell->highest));
    	shell->entries = shell->highest - shell->lowest;
    	shell->positions = new int[shell->entries ];

    	for(unsigned int i = 0; i < shell->entries; i++) {
    		shell->positions[ i] = shell->total_rows - shell->entries + i;
    	}

    }std::cout << "b" <<std::endl;
    std::cout << "Entries:" << shell->entries << std::endl;
    VecSet(shell->src, 0.0 );

	PetscScalar * vals;
	VecGetArray( x, &vals);
	std::cout << "c" <<std::endl;
	VecSetValues(shell->src, shell->entries, shell->positions, vals, ADD_VALUES);

	ierr = KSPSolve(shell->ksp, shell->src, shell->dst);

	std::cout << "d" <<std::endl;
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
	  if ( ierr != 0) {
		  std::cout << "Create PC Failed at 1"<<std::endl;
	  }
	  ierr = PCSetType(pc,PCSHELL);
	  if ( ierr != 0) {
	  std::cout << "Create PC Failed at 2"<<std::endl;
	  }
	  ierr = PCShellSetApply(pc,SampleShellPCApply);
	  if ( ierr != 0) {
	  std::cout << "Create PC Failed at 3"<<std::endl;
	  }
	  ierr = PCShellSetContext(pc,& additional_data);
	  if ( ierr != 0) {
	  std::cout << "Create PC Failed at 4"<<std::endl;
	  }
	  /**
	  ierr = PCShellSetDestroy(pc,SampleShellPCDestroy);

	  ierr = PCShellSetName(pc,"SweepingPreconditioner");
	   **/

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

    indices = new int[additional_data.highest-additional_data.sub_lowest];
    for(int i = 0; i < additional_data.highest-additional_data.sub_lowest ; i++) {
    	indices[i] = additional_data.sub_lowest + i;
    }
    ISCreateGeneral(PETSC_COMM_SELF,additional_data.highest-additional_data.sub_lowest,indices,PETSC_COPY_VALUES,&is);
    MatAssemblyBegin(matrix , MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix , MAT_FINAL_ASSEMBLY);
    MatGetSubMatrix(matrix, is , is, MAT_INITIAL_MATRIX, &additional_data.matrix);
    MatAssemblyBegin(additional_data.matrix , MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(additional_data.matrix , MAT_FINAL_ASSEMBLY);
    std::cout << "1" <<std::endl;
    /**
    PetscViewer    viewer;
    PetscViewerDrawOpen(PETSC_COMM_SELF,NULL,NULL,0,0,1000,1000,&viewer);
    PetscObjectSetName((PetscObject)viewer,"Matrix Structure");
    PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG);
    MatView(additional_data.matrix,viewer);
    **/
    VecCreate(PETSC_COMM_SELF, & additional_data.src);
    VecCreate(PETSC_COMM_SELF, & additional_data.dst);
    VecSetType(additional_data.src, VECMPI);
    VecSetType(additional_data.dst, VECMPI);
    std::cout << "2" <<std::endl;
    MatGetSize(additional_data.matrix, &additional_data.total_rows, &additional_data.total_cols );
    VecSetSizes( additional_data.src, additional_data.total_rows, additional_data.total_rows);
    VecSetSizes( additional_data.dst, additional_data.total_rows, additional_data.total_rows);
    std::cout << "3" <<std::endl;
    VecSet( additional_data.src, 0.0);
    VecSet( additional_data.dst, 0.0);

    std::cout << "4" <<std::endl;
    KSPCreate(PETSC_COMM_SELF,& additional_data.ksp);
    KSPSetOperators(additional_data.ksp,additional_data.matrix,additional_data.matrix);
    std::cout << "5" <<std::endl;
    KSPSetType(additional_data.ksp,KSPPREONLY);

    std::cout << "6" <<std::endl;
    KSPGetPC(additional_data.ksp,&Temppc);
    PCSetType(Temppc,PCLU);
    std::cout << "7" <<std::endl;
    PCFactorSetMatSolverPackage(Temppc,MATSOLVERMUMPS);
    //PCFactorSetUpMatSolverPackage(Temppc);
    std::cout << "8" <<std::endl;
    additional_data.entries = additional_data.highest - additional_data.lowest;
    additional_data.positions = new int[additional_data.entries ];
   	for(unsigned int i = 0; i < additional_data.entries; i++) {
   		additional_data.positions[ i] = additional_data.total_rows - additional_data.entries + i;
   	}
    create_pc();

    std::cout << "Result of Preparation: highest: " << additional_data.highest << " lowest:"<< additional_data.lowest<< " Number of Rows:" << additional_data.total_rows << " Number of cols:" << additional_data.total_cols<<std::endl;
  }


