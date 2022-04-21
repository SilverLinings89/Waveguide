#pragma once
/**
 * @file SingleCoreRun.h
 * @author Pascal Kraft
 * @brief This is deprecated. It is supposed to be used for minature examples that rely on only a Local Problem instead of an object hierarchy.
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
 * @brief In cases in which a single core is enough to solve the problem, this runner can be used.
 * It is the only one that constructs the mainProblem member to be a Local instead of a NonLocal problem.
 */
class SingleCoreRun: public Simulation {
  LocalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  SingleCoreRun();

  virtual ~SingleCoreRun();

  /**
   * @brief Prepares the mainProblem, which in this case is cheap because it is completely local.
   * 
   */
  void prepare() override;

  /**
   * @brief Computes the solution.
   * 
   */
  void run() override;

  /**
   * @brief Not required / not implemented.
   * 
   */
  void prepare_transformed_geometry() override;
};
