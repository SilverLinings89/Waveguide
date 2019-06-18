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
#include "../Code/SpaceTransformations/HomogenousTransformationRectangular.h"
#include "../Code/SpaceTransformations/DualProblemTransformationWrapper.h"

#include "../Code/MeshGenerators/MeshGenerator.h"
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

	SpaceTransformation * st;

	if(GlobalParams.Sc_Homogeneity) {
		st = new HomogenousTransformationRectangular(GlobalParams.MPI_Rank);
	} else {
		st = new InhomogenousTransformationRectangular(GlobalParams.MPI_Rank);
	}

	st->estimate_and_initialize();

	MeshGenerator * mg;
	mg = new SquareMeshGenerator(st);
	
	SpaceTransformation * dst;
	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint ) {
	  dst = new DualProblemTransformationWrapper(st,  GlobalParams.MPI_Rank);
	  dst->estimate_and_initialize();
	} else {
		// should be different? not sure.
		dst = new DualProblemTransformationWrapper(st,  GlobalParams.MPI_Rank);
		dst->estimate_and_initialize();
	}

	Waveguide * waveguide;
	std::string prefix = "";
	if(GlobalParams.Sc_Schema == OptimizationSchema::Adjoint) {
	  prefix = "primal";
	} else {
	  prefix = ".";
	}

	waveguide = new Waveguide(MPI_COMM_WORLD, mg, st);

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

  
	deallog.pop();

	deallog.push("Run");

	opt->run();


	deallog.pop();
	return 0;
}

#endif
