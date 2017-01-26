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
#include "Code/Helpers/Parameters.h"
#include "Code/Helpers/ParameterReader.cpp"
#include "Code/Helpers/staticfunctions.cpp"
#include "Code/OptimizationStrategies/Optimization.cpp"
#include "Code/Helpers/ParameterReader.cpp"
#include "Code/Helpers/Parameters.cpp"
#include "Code/Core/Sector.cpp"
#include "Code/Core/Waveguide.cpp"
#include "Code/Helpers/ExactSolution.cpp"
#include "Code/Core/SolutionWeight.cpp"
#include "Code/OutputGenerators/Console/GradientTable.cpp"

#include "Code/SpaceTransformations/SpaceTransformation.cpp"
#include "Code/SpaceTransformations/InhomogenousTransformationRectangular.cpp"
#include "Code/SpaceTransformations/InhomogenousTransformationCircular.cpp"
#include "Code/SpaceTransformations/HomogenousTransformationCircular.cpp"
#include "Code/SpaceTransformations/HomogenousTransformationRectangular.cpp"
#include "Code/SpaceTransformations/DualProblemTransformationWrapper.cpp"

#include "Code/MeshGenerators/MeshGenerator.cpp"
#include "Code/MeshGenerators/RoundMeshGenerator.cpp"
#include "Code/MeshGenerators/SquareMeshGenerator.cpp"

#include "Code/OptimizationStrategies/Optimization.cpp"
#include "Code/OptimizationStrategies/AdjointOptimization.cpp"
#include "Code/OptimizationStrategies/FDOptimization.cpp"

#include <deal.II/base/parameter_handler.h>
#include "Code/OptimizationAlgorithm/OptimizationAlgorithm.cpp"
#include "Code/OptimizationAlgorithm/OptimizationCG.cpp"
#include "Code/OptimizationAlgorithm/OptimizationSteepestDescent.cpp"
#include "Code/OptimizationAlgorithm/Optimization1D.cpp"


using namespace dealii;

int main (int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	GlobalParams = GetParameters();

	MPI_Comm  mpi_primal, mpi_dual;

	mpi_primal = MPI_COMM_WORLD;
	mpi_dual = MPI_COMM_WORLD;

	SpaceTransformation * st;

	if(GlobalParams.M_C_Shape == ConnectorType::Circle){
    if(GlobalParams.Sc_Homogeneity) {
      st = new HomogenousTransformationCircular();
    } else {
      st = new InhomogenousTransformationCircular();
    }
  } else {
    if(GlobalParams.Sc_Homogeneity) {
     st = new HomogenousTransformationRectangular();
    } else {
     st = new InhomogenousTransformationRectangular();
    }
  }

	MeshGenerator * mg;
	if(GlobalParams.M_C_Shape == ConnectorType::Circle ){
	  mg = new RoundMeshGenerator(st );
	} else {
	  mg = new SquareMeshGenerator(st);
	}

	SpaceTransformation * dst;
	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint ) {
	  dst = new DualProblemTransformationWrapper(st);
	}

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint ) {
	  // adjoint based
	  int primal_rank = GlobalParams.MPI_Rank;
	  int dual_rank = GlobalParams.NumberProcesses - 1 - GlobalParams.MPI_Rank;
	  if(dual_rank < 0) {
	    dual_rank += GlobalParams.NumberProcesses;
	  }
	  MPI_Comm_split(MPI_COMM_WORLD, 1, primal_rank, &mpi_primal);
	  // MPI_Comm_split(MPI_COMM_WORLD, 0, primal_rank, &mpi_primal);
	  MPI_Comm_split(MPI_COMM_WORLD, 1, dual_rank, &mpi_dual);
	  //MPI_Comm_split(MPI_COMM_WORLD, 0, dual_rank, &mpi_dual);
	} else {
	  // fd based
    int primal_rank = GlobalParams.MPI_Rank;
	  MPI_Comm_split(MPI_COMM_WORLD, 0, primal_rank, &mpi_primal);
	}

	Waveguide * primal_waveguide;
	primal_waveguide = new Waveguide(mpi_primal, mg, st);

	Waveguide * dual_waveguide;

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	  dual_waveguide = new Waveguide(mpi_dual, mg, dst);
	}

	Optimization * opt;

	OptimizationAlgorithm<double> * Oa_d;
	OptimizationAlgorithm<std::complex<double>> * Oa_c;

	if(GlobalParams.Sc_SteppingMethod == SteppingMethod::CG) {
	  // Oa = new OptimizationCG();
	} else if (GlobalParams.Sc_SteppingMethod == SteppingMethod::Steepest){
	  Oa_d = new OptimizationSteepestDescent();
	} else {
	  Oa_c = new Optimization1D();
	}

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	  opt = new AdjointOptimization(primal_waveguide, dual_waveguide, mg, st, dst, Oa_c);
	} else {
	  opt = new FDOptimization(primal_waveguide, mg, st, Oa_d);
	}

	opt->run();

	return 0;
}

#endif
