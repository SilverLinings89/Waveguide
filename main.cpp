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

  deallog.depth_console(5);

	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  PrepareStreams();

  deallog.push("Main");

  deallog << "Streams prepared. Loading Parameters..." <<std::endl;

	GlobalParams = GetParameters();

	deallog << "Parameters loaded. Preparing Space Transformations..." <<std::endl;

	MPI_Comm  mpi_primal, mpi_dual;

	mpi_primal = MPI_COMM_WORLD;
	mpi_dual = MPI_COMM_WORLD;

  int primal_rank = GlobalParams.MPI_Rank;
  int dual_rank = GlobalParams.NumberProcesses - 1 - GlobalParams.MPI_Rank;

	SpaceTransformation * st;

	if(GlobalParams.M_C_Shape == ConnectorType::Circle){
    if(GlobalParams.Sc_Homogeneity) {
      st = new HomogenousTransformationCircular(primal_rank);
    } else {
      st = new InhomogenousTransformationCircular(primal_rank);
    }
  } else {
    if(GlobalParams.Sc_Homogeneity) {
     st = new HomogenousTransformationRectangular(primal_rank);
    } else {
     st = new InhomogenousTransformationRectangular(primal_rank);
    }
  }

	st->estimate_and_initialize();

	deallog << "Done. Preparing Mesh Generators..." <<std::endl;

	MeshGenerator * mg;
	if(GlobalParams.M_C_Shape == ConnectorType::Circle ){
	  mg = new RoundMeshGenerator(st );
	} else {
	  mg = new SquareMeshGenerator(st);
	}

	SpaceTransformation * dst;
	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint ) {
	  dst = new DualProblemTransformationWrapper(st, dual_rank, primal_rank);
	  dst->estimate_and_initialize();
	}

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint ) {
	  // adjoint based

	  MPI_Comm_split(MPI_COMM_WORLD, 1, primal_rank, &mpi_primal);
	  // MPI_Comm_split(MPI_COMM_WORLD, 0, primal_rank, &mpi_primal);
	  MPI_Comm_split(MPI_COMM_WORLD, 1, dual_rank, &mpi_dual);
	  //MPI_Comm_split(MPI_COMM_WORLD, 0, dual_rank, &mpi_dual);
	} else {
	  // fd based
    MPI_Comm_split(MPI_COMM_WORLD, 0, primal_rank, &mpi_primal);
	}

	deallog << "Done. Building Waveguides." <<std::endl;

	Waveguide * primal_waveguide;
	std::string prefix = "";
	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	  prefix = "primal";
	} else {
	  prefix = ".";
	}

	primal_waveguide = new Waveguide(mpi_primal, mg, st, "primal");

	Waveguide * dual_waveguide;

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	  // TODO Wieder auf dual setzen.
	  dual_waveguide = new Waveguide(mpi_dual, mg, st, "dual");
	}

  deallog << "Done. Loading Schema..." <<std::endl;

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

  deallog << "Done." <<std::endl;

  deallog.push("Configuration");
	if(GlobalParams.MPI_Rank == 0) {
	  deallog << "Prepared for the following setup: " <<std::endl;
	  deallog << "Mesh Generator:";
	  if(GlobalParams.M_C_Shape == ConnectorType::Circle ){
	    deallog << "Round Mesh Generator";
	  } else {
	    deallog << "Rectangular Mesh Generator";
	  }

	  deallog << std::endl <<"Space Transformation: ";

	  if(GlobalParams.M_C_Shape == ConnectorType::Circle){
	      if(GlobalParams.Sc_Homogeneity) {
	        deallog << "Homogenous Transformation Circular" ;
	      } else {
	        deallog << "Inhomogenous Transformation Circular" ;
	      }
	    } else {
	      if(GlobalParams.Sc_Homogeneity) {
	        deallog << "Homogenous Transformation Rectangular" ;
	      } else {
	        deallog << "Inhomogenous Transformation Rectangular" ;
	      }
	    }

	  deallog << std::endl << "Optimization Schema:" ;

	  if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	    deallog << "Adjoint Schema" ;
	  } else {
	    deallog << "Finite Differences" ;
	  }

	  deallog << std::endl << "Stepping Method:" ;

	  if(GlobalParams.Sc_SteppingMethod == SteppingMethod::CG) {
	    deallog << "Conjugate  Gradient";
	  } else if (GlobalParams.Sc_SteppingMethod == SteppingMethod::Steepest){
	    deallog << "Steepest Descend";
	  } else {
	    deallog << "1D search";
	  }
	  deallog << std::endl;
	}

	deallog.pop();

	deallog << "Starting optimization run..." << std::endl;

	deallog.push("Run");

	opt->run();

	deallog.pop();

	deallog.pop();
	return 0;
}

#endif
