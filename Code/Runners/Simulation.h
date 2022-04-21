#pragma once
/**
 * @file Simulation.h
 * @author Pascal Kraft
 * @brief Base class of the simulation runners.
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
 * @brief This base class is very important and abstract.
 * While the HierarchicalProblem types perform the computation of an E-field solution to a problem, these classes are the reason why we do so.
 * The derived classes handle default experiments for the sweeping preconditioners, convergence studies or shape optimization.
 * 
 */
class Simulation {
  HierarchicalProblem *mainProblem;
  RectangularMode * rmProblem;

 public:
  Simulation();

  virtual ~Simulation();

  /**
   * @brief In derived classes, this function sets up all that is required to perform the core functionality, i.e. construct problems types.
   * 
   */
  virtual void prepare() = 0;

  /**
   * @brief Run the core computation.
   * 
   */
  virtual void run() = 0;

  /**
   * @brief If a representation of the solution in the physical coordinates is required, this function provides it.
   * 
   */
  virtual void prepare_transformed_geometry() = 0;

  /**
   * @brief Create a output directory to store the computational results in.
   * 
   */
  void create_output_directory();
};
