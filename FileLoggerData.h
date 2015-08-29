#ifndef FileLoggerDataFlag
#define FileLoggerDataFlag

#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <deal.II/base/timer.h>

using namespace dealii;

class FileLoggerData {
public:
	FileLoggerData();
	std::string solver, preconditioner;
	int 		XLength, YLength, ZLength, ParamSteps, Dofs, Precondition_BlockSize;
	double		PML_in, PML_out, PML_mantle, Solver_Precision, Precondition_weight;
};

#endif
