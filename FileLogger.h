/**
 * Die Klasse File-Logger
 * Dies Klasse schreibt logfiles. Beim initialisieren kann angegeben werden, welche Werte in der Log-Tabelle auftauchen sollen. Außerdem erfolgt eine Zeitmessung. Ein Logger hat deshalb eine start() und stop() Methode.
 *
 * Funktion start():
 * Startet die Zeitmessung.
 *
 * Funktion stop():
 * Stoppt die Zeitmessung und schreibt alle angegebenen Parameter mit der Zeit ins Logfile auf der Festplatte.
 *
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */

#ifndef FileLoggerFlag
#define FileLoggerFlag

#include <deal.II/base/timer.h>
#include "FileLoggerData.h"

using namespace dealii;

class FileLogger {
	private:
		Timer t;
		std::string filename;
		std::ofstream file;
		std::string pre;
		std::string post;
		FileLoggerData fld;

	public:

		FileLogger(std::string, const FileLoggerData& );
		FileLogger();
		bool solver, preconditioner, XLength, YLength, ZLength, ParamSteps, Dofs, Precondition_BlockSize, PML_in, PML_out, PML_mantle, Solver_Precision, Precondition_weight, walltime, cputime;
		void start();
		void stop() ;


};

#endif
