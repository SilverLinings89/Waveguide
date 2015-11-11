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


int main ()
{
	GlobalParams = GetParameters();
	WaveguideStructure structure(GlobalParams);
	Waveguide<dealii::SparseMatrix<double>, dealii::Vector<double> > waveguide(GlobalParams, structure);
	Optimization opt(GlobalParams, waveguide, structure);
	opt.run();
	return 0;
}

#endif
