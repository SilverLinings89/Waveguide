/**
 * The ParameterReader contains all data required for the reading and exposing of parameter values.
 * It encapsulates a ParameterHandler object for reading the file and some more functionality for ease of use.
 * It is used to load a file a runtime an insert constant values.
 * This removes the necessity of recompiling upon changing parameter values.
 */

#ifndef ParameterReaderFlag
#define ParameterReaderFlag

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

class ParameterReader : public Subscriptor
{
public:

	ParameterReader			(ParameterHandler &prmhandler);
	void read_parameters	(const std::string inputfile);
	void declare_parameters	();

private:
	ParameterHandler &prm;

};

#endif
