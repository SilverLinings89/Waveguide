#ifndef FileLoggerFlag
#define FileLoggerFlag

#include <deal.II/base/timer.h>
#include "FileLoggerData.h"

using namespace dealii;
/**
 * \class FileLogger
 * Objects of this type are used to hide the persistence layer when logging progress. This means, that they write messages into files that stem from the calculation. The provide the comfort of not having to deal with IO-code in the actual mathematical scheme.
 * The object will write a table to the file. In this table certain values can be written. This values can be picked by setting the appropriate flags (setting the according values in the data structure to true).
 * This class has been marked for removal in version 1.4. It will be replaced by the internal logging functionality in Deal.II.
 * \author Pascal Kraft
 * \date 23.11.2015
 */
class FileLogger {
	private:
		Timer t;
		std::string filename;
		std::ofstream file;
		std::string pre;
		std::string post;
		FileLoggerData fld;

	public:
		/**
		 * This constructor requires a filename and a FileLoggerData structure reference specifying what this logger is supposed to keep a log of.
		 * \param fname This is a string containing the file that should be written to.
		 * \param data This structure contains detailed information on the kind of information to store in the file.
		 */
		FileLogger(std::string fname, const FileLoggerData&  data);

		/**
		 * This constructor should not be used anymore.
		 */
		FileLogger();

		/**
		 * This flag signals, that the type of the Solver in use should be logged.
		 */
		bool solver;
		/**
		 * This flag signals, that the type of the Preconditioner in use should be logged.
		 */
		bool preconditioner;
		/**
		 * This flag will include the Length along the \f$x\f$-axis in the table.
		 */
		bool XLength;
		/**
		 * This flag will include the Length along the \f$y\f$-axis in the table.
		 */
		bool YLength;
		/**
		 * This flag will include the Length along the \f$z\f$-axis in the table.
		 */
		bool ZLength;
		/**
		 * This flag will include the the maximal amount of steps in the table.
		 */
		bool ParamSteps;
		/**
		 * If this is set to true, the number of degrees of freedom will be included in the table.
		 */
		bool Dofs;
		/**
		 * This flag triggers the recording of the block-size in the table.
		 */
		bool Precondition_BlockSize;
		/**
		 * Marking this property leads to an entry listing the width of the PML on the input-side.
		 */
		bool PML_in;
		/**
		 * Marking this property leads to an entry listing the width of the PML on the output-side.
		 */
		bool PML_out;
		/**
		 * Marking this property leads to an entry listing the width of the PML on the mantle.
		 */
		bool PML_mantle;
		/**
		 * This entry will include the Solvers abort-precision (assumed convergence) in the table.
		 */
		bool Solver_Precision;
		/**
		 * Some preconditioners require a weight to be specified. Should this be true und this variable set to true also, this value will be listed in the output table.
		 */
		bool Precondition_weight;
		/**
		 * The walltime is the time as the user experiences. It is in a way a very macroscopic property. Considering short runtime of an application, this is the value you are trying to reduce.
		 */
		bool walltime;
		/**
		 * The CPU-time is a property, that aligns much more with the actual computational effort than the walltime. It measures the time consumed by the process on every individual core and adds them up. Considering to consume as little computational power as possible, this is the value you are looking for.
		 */
		bool cputime;

		/**
		 * This will start the internal timer. Logging will occur as soon as stop() gets called on this object.
		 */
		void start();

		/**
		 * When this function gets called, all information is available. The process, that should be measured is assumed to have terminated and the timer can be stopped. Upon stopping the timer and calculating walltime and cputime, a line is written to the output-file containing all the demanded information. Possible processes to measure are for example
		 * -# mesh generation,
		 * -# system-matrix assemblation,
		 * -# time required for the preconditioning to terminate,
		 * -# time required to solve the system-matrix.
		 */
		void stop() ;


};

#endif
