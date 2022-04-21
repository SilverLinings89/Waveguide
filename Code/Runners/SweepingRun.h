#pragma once
/**
 * @file SweepingRun.h
 * @author Pascal Kraft
 * @brief  Default Runner for sweeping preconditioner runs.
 * @version 0.1
 * @date 2022-04-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include "../ModalComputations/RectangularMode.h"

/**
 * @brief This runner constructs a single non-local problem and solves it. 
 * This is mainly used for work on the sweeping preconditioner since it enables a single run and result output.
 * 
 */
class SweepingRun: public Simulation {
  NonLocalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  SweepingRun();

  virtual ~SweepingRun();

  /**
   * @brief Prepare the solver hierarchy for the parameters provided in the input fields.
   * 
   */
  void prepare() override;

  /**
   * @brief Solve the non-local problem.
   * 
   */
  void run() override;

  /**
   * @brief Not required / Not implemented.
   * 
   */
  void prepare_transformed_geometry() override;

};
