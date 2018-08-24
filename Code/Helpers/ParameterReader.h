#ifndef ParameterReaderFlag
#define ParameterReaderFlag

#include <deal.II/base/parameter_handler.h>
#include "../Core/Waveguide.h"
using namespace dealii;

/**
 * \class ParameterReader
 * \brief This class is used to gather all the information from the input file
 *and store it in a static object available to all processes.
 *
 * The ParameterReader is a very useful tool. It uses a deal-function to read a
 *xml-file and parse the contents to specific variables. These variables have
 *default values used in their declaration. The members of this class do two
 *things:
 * -# declare the variables. This includes setting a data-type for them and a
 *default value should none be provided in the input file. Furthermore there can
 *be restrictions like maximum or minimum values etc.
 * -# call an external function to parse an input-file.
 *
 * After creating an object of this type and calling both declare() and read(),
 *this object contains all the information from the input file and can be used
 *in the code without dealing with persistence.
 *
 *\author Pascal Kraft
 *\date 23.11.2015
 */
class ParameterReader : public Subscriptor {
 public:
  /**
   * Deal Offers the ParameterHandler object wich contains all of the
   * parsing-functionality. An object of that type is included in this one. This
   * constructor simply uses a copy-constructor to initialize it.
   */
  ParameterReader(ParameterHandler &prmhandler);

  /**
   * This member calls the read_input_from_xml()-function of the contained
   * ParameterHandler and this replaces the default values with the values in
   * the input file.
   */
  void read_parameters(const std::string inputfile);

  /**
   * In this function, we add all values descriptions to the parameter-handler.
   * This includes
   * -# a default value,
   * -# a data-type,
   * -# possible restrictions (greater than zero etc.),
   * -# a description, which is displayed in deals ParameterGUI-tool,
   * -# a hierarchical structure to order the variables.
   *
   * Deals Parameter-GUI can be installed at build-time of the library and
   * offers a great and easy way to edit the input file. It displays appropriate
   * input-methods depending on the type, so, for example, in case of a
   * selection from three different values (i.e. the name of a solver that has
   * to either be GMRES, MINRES or UMFPACK) it displays a dropdown containing
   * all the options.
   */
  void declare_parameters();

 private:
  ParameterHandler &prm;
};

#endif
