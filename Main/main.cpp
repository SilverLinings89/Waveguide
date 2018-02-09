#ifndef MainCppFlag
#define MainCppFlag

#include <sys/types.h>
#include <sys/stat.h>
#include <deal.II/base/parameter_handler.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "../Code/Core/Waveguide.h"
#include "../Code/Helpers/Parameters.h"
#include "../Code/Helpers/ParameterReader.h"
#include "../Code/Helpers/staticfunctions.h"
#include "../Code/OptimizationStrategies/Optimization.h"
#include "../Code/Helpers/ModeManager.h"
#include "../Code/Core/Sector.h"
#include "../Code/Helpers/PointVal.h"
#include "../Code/Helpers/ExactSolution.h"
#include "../Code/Core/SolutionWeight.h"
#include "../Code/OutputGenerators/Console/GradientTable.h"

#include "../Code/SpaceTransformations/SpaceTransformation.h"
#include "../Code/SpaceTransformations/InhomogenousTransformationRectangular.h"
#include "../Code/SpaceTransformations/InhomogenousTransformationCircular.h"
#include "../Code/SpaceTransformations/HomogenousTransformationCircular.h"
#include "../Code/SpaceTransformations/HomogenousTransformationRectangular.h"
#include "../Code/SpaceTransformations/DualProblemTransformationWrapper.h"

#include "../Code/MeshGenerators/MeshGenerator.h"
#include "../Code/MeshGenerators/RoundMeshGenerator.h"
#include "../Code/MeshGenerators/SquareMeshGenerator.h"

#include "../Code/OptimizationStrategies/AdjointOptimization.h"
#include "../Code/OptimizationStrategies/FDOptimization.h"

#include "../Code/OptimizationAlgorithm/OptimizationAlgorithm.h"
#include "../Code/OptimizationAlgorithm/OptimizationCG.h"
#include "../Code/OptimizationAlgorithm/OptimizationSteepestDescent.h"
#include "../Code/OptimizationAlgorithm/Optimization1D.h"

#include "../Code/Helpers/ShapeDescription.h"

int main (int argc, char *argv[])
{

  if(argc > 1) {
    input_file_name = argv[1];
  } else {
    input_file_name = "../Parameters/Parameters.xml";
  }

  deallog.depth_console(5);

	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  PrepareStreams();

  deallog.push("Main");

  deallog << "Streams prepared. Loading Parameters..." <<std::endl;

	GlobalParams = GetParameters();

	if(argc ==3) {
	  GlobalParams.StepWidth = std::atof(argv[2]);
	}

	ModeMan.load();

	deallog << "Parameters loaded. Preparing Space Transformations..." <<std::endl;

	SpaceTransformation * st;

	if(GlobalParams.M_C_Shape == ConnectorType::Circle){
    if(GlobalParams.Sc_Homogeneity) {
      st = new HomogenousTransformationCircular(GlobalParams.MPI_Rank);
    } else {
      st = new InhomogenousTransformationCircular(GlobalParams.MPI_Rank);
    }
  } else {
    if(GlobalParams.Sc_Homogeneity) {
     st = new HomogenousTransformationRectangular(GlobalParams.MPI_Rank);
    } else {
     st = new InhomogenousTransformationRectangular(GlobalParams.MPI_Rank);
    }
  }

	st->estimate_and_initialize();

	deallog << "Done. Preparing Mesh Generators..." <<std::endl;

	MeshGenerator * mg;
	if(GlobalParams.M_C_Shape == ConnectorType::Circle ){
	  mg = new RoundMeshGenerator(st);
	} else {
	  mg = new SquareMeshGenerator(st);
	}

	SpaceTransformation * dst;
	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint ) {
	  dst = new DualProblemTransformationWrapper(st,  GlobalParams.MPI_Rank);
	  dst->estimate_and_initialize();
	} else {
		// should be different? not sure.
		dst = new DualProblemTransformationWrapper(st,  GlobalParams.MPI_Rank);
		dst->estimate_and_initialize();
	}

	deallog << "Done. Building Waveguides." <<std::endl;

	Waveguide * waveguide;
	std::string prefix = "";
	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	  prefix = "primal";
	} else {
	  prefix = ".";
	}

	waveguide = new Waveguide(MPI_COMM_WORLD, mg, st);

	deallog << "Done. Loading Schema..." <<std::endl;

	Optimization * opt;

	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
		OptimizationAlgorithm<std::complex<double>> * Oa_c ;
		Oa_c = new Optimization1D();
		opt = new AdjointOptimization(waveguide, mg, st, dst, Oa_c);
	} else {
		OptimizationAlgorithm<double> * Oa_d;
		Oa_d = new OptimizationSteepestDescent();
		opt = new FDOptimization(waveguide, mg, st, Oa_d);
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
	  deallog << " with step width " << GlobalParams.StepWidth << std::endl;
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
