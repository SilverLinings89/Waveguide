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
	//Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, GlobalParams.PRM_S_MPITasks);

	 Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	GlobalParams.MPI_Communicator = MPI_COMM_WORLD;
	GlobalParams.MPI_Rank =Utilities::MPI::this_mpi_process(GlobalParams.MPI_Communicator);
	std::cout << GlobalParams.MPI_Rank << std::endl;
	std::cout << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)<< std::endl;
	structure = new WaveguideStructure(GlobalParams);


	Waveguide<TrilinosWrappers::SparseMatrix, dealii::TrilinosWrappers::MPI::Vector > waveguide(GlobalParams);
	Optimization<TrilinosWrappers::SparseMatrix, dealii::TrilinosWrappers::MPI::Vector> opt(GlobalParams, waveguide);
	opt.run();

	return 0;
}

#endif
