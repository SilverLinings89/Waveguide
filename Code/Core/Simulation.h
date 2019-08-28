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

class Simulation {
private:
    GeometryManager geometry;
    Parameters parameters;

public:
    Simulation();

    virtual ~Simulation();

    void Run();

    void LoadParameters();

    void LoadParametersFromFile();

    void OverwriteParametersFromConsole();

    void PrepareGeometry();

    void PrepareTransformedGeometry();

    void InitializeMainProblem();

    void InitializeAuxiliaryProblem();

    void CreateOutputDirectory();
};

#endif /* CODE_CORE_SIMULATION_H_ */
