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

#include "Code/MeshGenerators/MeshGenerator.cpp"
#include "Code/MeshGenerators/RoundMeshGenerator.cpp"
#include "Code/MeshGenerators/SquareMeshGenerator.cpp"

#include "Code/SpaceTransformations/SpaceTransformation.cpp"
#include "Code/SpaceTransformations/InhomogenousTransformationRectangular.h"
#include "Code/SpaceTransformations/InhomogenousTransformationCircular.h"
#include "Code/SpaceTransformations/HomogenousTransformationCircular.cpp"
#include "Code/SpaceTransformations/HomogenousTransformationRectangular.h"
#include "Code/SpaceTransformations/DualProblemTransformationWrapper.h"

#include "Code/OptimizationStrategies/Optimization.cpp"
#include "Code/OptimizationStrategies/AdjointOptimization.cpp"
#include "Code/OptimizationStrategies/FDOptimization.h"

#include <deal.II/base/parameter_handler.h>
#include "Code/OptimizationAlgorithm/OptimizationAlgorithm.h"
#include "Code/OptimizationAlgorithm/OptimizationCG.h"
#include "Code/OptimizationAlgorithm/OptimizationSteepestDescent.h"


using namespace dealii;

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	GlobalParams = GetParameters();

	MPI_Comm * mpi_primal, * mpi_dual;

	SpaceTransformation * st;

  if(GlobalParams.M_C_Shape == ConnectorType::Circle){
    if(GlobalParams.Sc_Homogeneity) {
      st = new HomogenousTransformationCircular();
    } else {
      // st = new InhomogenousTransformationRectangular();
    }
  } else {
    if(GlobalParams.Sc_Homogeneity) {
     // st = new HomogenousTransformationRectangular();
    } else {
     // st = new InhomogenousTransformationRectangular();
    }
  }

	MeshGenerator * mg;
	if(GlobalParams.M_C_Shape == ConnectorType::Circle ){
	  mg = new RoundMeshGenerator(st );
	} else {
	  mg = new SquareMeshGenerator(st);
	}



	SpaceTransformation * dst;

	dst = new DualProblemTransformationWrapper(st);

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint ) {
	  // adjoint based
	  int primal_rank = GlobalParams.MPI_Rank;
	  int dual_rank = GlobalParams.NumberProcesses - 1 - GlobalParams.MPI_Rank;
	  if(dual_rank < 0) {
	    dual_rank += GlobalParams.NumberProcesses;
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

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	  dual_waveguide = new Waveguide(mpi_dual, mg, dst);
	}

	Optimization * opt;

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	  opt = new AdjointOptimization(primal_waveguide, dual_waveguide, mg, st, dst);
	} else {
	  opt = new FDOptimization(primal_waveguide, mg, st);
	}


	return 0;
}

#endif
