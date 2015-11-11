/**
 * Die Klasse FileLoggerData
 * Dies ist eine Hilfsklasse, die Daten für einen Logger speichert. Sie dient ausschließlich der Kapselung von Daten und hat außer ihrem Konstruktor keine Funktionen
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */

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
