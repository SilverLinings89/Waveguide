#pragma once
/**
 * @file ParameterSweep.h
 * @author Pascal Kraft
 * @brief Contains the parameter sweep runner which is somewhat deprecated.
 * @version 0.1
 * @date 2022-04-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "./Simulation.h"
#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"

/**
 * @brief The Parameter run performs multiple forward runs for a sweep across a parameter value, i.e multiple computations for different domain sizes or similar.
 * This is not really required anymore because there is now an implementation of parameter overrides which does the same but is parallelizable.
 * The class is not documented for this reason but the code is simple.
 */
class ParameterSweep: public Simulation {
  LocalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  ParameterSweep();

  ~ParameterSweep();

  void prepare() override;

  void run() override;

  void prepare_transformed_geometry() override;

};
