#ifndef MainCppFlag
#define MainCppFlag

#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <sstream>

#include <mpi.h>

#include "Code/Core/Waveguide.h"
#include "Code/Core/WaveguideStructure.h"
#include "Code/Helpers/Parameters.h"
#include "Code/Helpers/ParameterReader.cpp"
#include "Code/Helpers/staticfunctions.cpp"
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

	MPI_Comm * mpi_primal, * mpi_dual;

	MeshGenerator * mg;
	if(GlobalParams.PRM_M_C_TypeIn == "circle"){
	  mg = new RoundMeshGenerator();
	} else {
	  mg = new SquareMeshGenerator();
	}

	SpaceTransformation * st;

	if(GlobalParams.PRM_M_C_TypeIn == "circle"){
    if(GlobalParams.PRM_M_BC_Homog == "true") {
      st = new HomogenousTransformationCircular();
    } else {
      st = new InhomogenousTransformationCircular();
    }
	} else {
	  if(GlobalParams.PRM_M_BC_Homog == "true") {
      st = new HomogenousTransformationRectangular();
    } else {
      st = new InhomogenousTransformationRectangular();
    }
	}

	SpaceTransformation * dst;

	dst = new DualProblemTransformationWrapper(st);

	if(GlobalParams.PRM_OptimizationStrategy == "adjoint" ) {
	  // adjoint based
	  int primal_rank = GlobalParams.MPI_Rank;
	  int dual_rank = GlobalParams.MPI_Size - 1 - GlobalParams.MPI_Rank;
	  if(dual_rank < 0) {
	    dual_rank += GlobalParams.MPI_Size;
	  }
	  MPI_Comm_split(MPI_COMM_WORLD, 1, primal_rank, mpi_primal);
	  MPI_Comm_split(MPI_COMM_WORLD, 1, dual_rank, mpi_dual);
	} else {
	  // fd based
    int primal_rank = GlobalParams.MPI_Rank;
	  MPI_Comm_split(MPI_COMM_WORLD, 1, primal_rank, mpi_primal);
	}

	// structure = new WaveguideStructure(GlobalParams);

	Waveguide * primal_waveguide;
	primal_waveguide = new Waveguide(mpi_primal, mg, st);

	Waveguide * dual_waveguide;

	if(GlobalParams.PRM_OptimizationStrategy == "adjoint") {
	  dual_waveguide = new Waveguide(mpi_dual, mg, dst);
	}

	Optimization * opt;

	if(GlobalParams.PRM_OptimizationStategy == "adjoint") {
	  opt = new AdjointOptimization();
	} else {
	  opt = new FDOptimization();
	}


	return 0;
}

#endif
