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
#include "GradientTable.cpp"

using namespace dealii;



int main (int argc, char *argv[])
{
	GlobalParams = GetParameters();
	structure = new WaveguideStructure(GlobalParams);

	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, GlobalParams.PRM_S_MPITasks);
	if(GlobalParams.PRM_S_Library == "DealII" ) {
		Waveguide<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double> > waveguide(GlobalParams);
		Optimization<dealii::BlockSparseMatrix<double>, dealii::BlockVector<double> > opt(GlobalParams, waveguide);
		opt.run();
	}
	if(GlobalParams.PRM_S_Library == "Trilinos") {
		Waveguide<dealii::TrilinosWrappers::BlockSparseMatrix, dealii::TrilinosWrappers::MPI::BlockVector  > waveguide(GlobalParams);
		Optimization<dealii::TrilinosWrappers::BlockSparseMatrix, dealii::TrilinosWrappers::MPI::BlockVector  > opt(GlobalParams, waveguide);
		opt.run();
	}
	if(GlobalParams.PRM_S_Library == "PETSc") {
			Waveguide<PETScWrappers::MPI::BlockSparseMatrix, dealii::PETScWrappers::MPI::BlockVector > waveguide(GlobalParams);
			Optimization<PETScWrappers::MPI::BlockSparseMatrix, dealii::PETScWrappers::MPI::BlockVector> opt(GlobalParams, waveguide);
			opt.run();
		}
	return 0;
}

#endif
