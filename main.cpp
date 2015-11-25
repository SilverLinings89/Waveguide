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
#include "Waveguide.cpp"
#include "WaveguideStructure.cpp"
#include "ExactSolution.cpp"
#include "SolutionWeight.cpp"

using namespace dealii;




int main (int argc, char *argv[])
{

	GlobalParams = GetParameters();
	WaveguideStructure structure(GlobalParams);
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	if(GlobalParams.PRM_S_Library == "DealII" ) {
		Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> > waveguide(GlobalParams, structure);
		Optimization<dealii::SparseMatrix<double>, dealii::Vector<double> > opt(GlobalParams, waveguide, structure);
		opt.run();
	}
	if(GlobalParams.PRM_S_Library == "Trilinos") {
		Waveguide<dealii::TrilinosWrappers::SparseMatrix, dealii::TrilinosWrappers::Vector > waveguide(GlobalParams, structure);
		Optimization<dealii::TrilinosWrappers::SparseMatrix, dealii::TrilinosWrappers::Vector > opt(GlobalParams, waveguide, structure);
		opt.run();
	}
	if(GlobalParams.PRM_S_Library == "PETSc") {
			Waveguide<dealii::PETScWrappers::SparseMatrix, dealii::PETScWrappers::Vector > waveguide(GlobalParams, structure);
			Optimization<dealii::PETScWrappers::SparseMatrix, dealii::PETScWrappers::Vector > opt(GlobalParams, waveguide, structure);
			opt.run();
		}
	return 0;
}

#endif
