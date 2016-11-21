#ifndef MainCppFlag
#define MainCppFlag

#include "Code/Core/Waveguide.h"
#include "Code/Core/WaveguideStructure.h"
#include "Code/Helpers/staticfunctions.cpp"
#include "Code/Helpers/Parameters.h"
#include "Code/Helpers/ParameterReader.cpp"
#include "Code/OptimizationStrategies/Optimization.cpp"
#include "Code/Helpers/ParameterReader.cpp"
#include "Code/Helpers/Parameters.cpp"
#include "Code/Core/Sector.cpp"
#include "Code/Core/Waveguide.cpp"
#include "Code/Core/WaveguideStructure.cpp"
#include "Code/Helpers/ExactSolution.cpp"
#include "Code/Core/SolutionWeight.cpp"
#include "Code/OutputGenerators/Console/GradientTable.cpp"

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	GlobalParams = GetParameters();

	structure = new WaveguideStructure(GlobalParams);

	Waveguide waveguide(GlobalParams);

	Optimization opt(GlobalParams, waveguide);
	opt.run();

	return 0;
}

#endif
