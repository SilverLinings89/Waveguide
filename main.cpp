#ifndef MainCppFlag
#define MainCppFlag

#include "Waveguide.h"
#include "WaveguideStructure.h"
#include "staticfunctions.cpp"
#include "Optimization.h"
#include "Parameters.h"
#include <deal.II/base/parameter_handler.h>
#include "ParameterReader.h"
#include "ParameterReader.cpp"
#include "FileLogger.cpp"
#include "FileLoggerData.cpp"
#include "Optimization.cpp"
#include "ParameterReader.cpp"
#include "Parameters.cpp"
#include "RightHandSide.cpp"
#include "Sector.cpp"
#include "PreconditionSweeping.cpp"
#include "Waveguide.cpp"
#include "WaveguideStructure.cpp"
#include "ExactSolution.cpp"
#include "SolutionWeight.cpp"

using namespace dealii;




int main (int argc, char *argv[])
{

	GlobalParams = GetParameters();
	WaveguideStructure structure(GlobalParams);
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, GlobalParams.PRM_S_MPITasks);
	if(GlobalParams.PRM_S_Library == "DealII" ) {
		Waveguide<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double> > waveguide(GlobalParams, structure);
		Optimization<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double> > opt(GlobalParams, waveguide, structure);
		opt.run();
	}
	if(GlobalParams.PRM_S_Library == "Trilinos") {
		Waveguide<dealii::TrilinosWrappers::BlockSparseMatrix, dealii::TrilinosWrappers::MPI::BlockVector  > waveguide(GlobalParams, structure);
		Optimization<dealii::TrilinosWrappers::BlockSparseMatrix, dealii::TrilinosWrappers::MPI::BlockVector  > opt(GlobalParams, waveguide, structure);
		opt.run();
	}
	if(GlobalParams.PRM_S_Library == "PETSc") {
			Waveguide<PETScWrappers::MPI::BlockSparseMatrix, dealii::PETScWrappers::MPI::BlockVector > waveguide(GlobalParams, structure);
			Optimization<PETScWrappers::MPI::BlockSparseMatrix, dealii::PETScWrappers::MPI::BlockVector> opt(GlobalParams, waveguide, structure);
			opt.run();
		}
	return 0;
}

#endif
