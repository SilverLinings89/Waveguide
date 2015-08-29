/** The FileLogger
 * File Logger is supposed to write specific logfiles. It has a constructor taking 2 Arguments: Filename and 2 Parameterlists. The Parameterlists are a list of specific values appropriate for the Logger.
 * The represent a text written before and after the timer value. (Stuff like Solver, Preconditioner, Settings, Steps etc.
 * @author: Pascal Kraft
 * @date: 22.5.2015
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
