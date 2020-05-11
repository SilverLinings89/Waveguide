/*
 * Simulation.h
 * This class handles all the heavy lifting for the order of important steps and
 * "owns" most of the important objects. It facilitates the run by loading the
 * parameters, identifying the geometry and initializing and starting the
 * numeric problem objects.
 * \date Jun 24, 2019
 * \author Pascal Kraft
 */

#ifndef CODE_CORE_SIMULATION_H_
#define CODE_CORE_SIMULATION_H_

#include "../Helpers/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"

class Simulation {
  NonLocalProblem *mainProblem;

 public:
  Simulation();

  virtual ~Simulation();

  void prepare();

  void run();

  void overwrite_parameters_from_console();

  void prepare_transformed_geometry();

  void create_output_directory();
};

#endif /* CODE_CORE_SIMULATION_H_ */
