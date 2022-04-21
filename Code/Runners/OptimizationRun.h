#pragma once
/**
 * @file OptimizationRun.h
 * @author Pascal Kraft
 * @brief Contains the Optimization Runner which performs shape optimization type computations.
 * @version 0.1
 * @date 2022-04-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "../GlobalObjects/GeometryManager.h"
#include "../Helpers/Parameters.h"
#include "../Hierarchy/NonLocalProblem.h"
#include <functional>

/**
 * @brief This runner performs a shape optimization run based on adjoint based shape optimization.
 * It is therefore one of the runner types that solves multiple forward problems.
 * 
 */
class OptimizationRun: public Simulation {
  static NonLocalProblem *mainProblem;
  static std::vector<std::vector<double>> shape_dofs;
  static std::vector<std::vector<double>> shape_gradients;
  static unsigned int step_counter;
  std::function< double(const dealii::Vector<double> &x, dealii::Vector<double> &g)> function_pointer;
  const unsigned int n_free_dofs;
  static double loss_functional_evaluation;
  
 
 public:
 /**
  * @brief Computes the number of free shape dofs for this configuration.
  * Also inits the step counter to 0.
  * 
  */
  OptimizationRun();

  virtual ~OptimizationRun();

  /**
   * @brief Prepares the object by constructing the solver hierarchy.
   * 
   */
  void prepare() override;

  /**
   * @brief Calls the BFGS solver and writes output.
   * First prepare the vector of shape parameters for the start configuration. Then we call the BFGS solver to perform the shape optimization and give it a handle to this object for the update handler.
   * 
   */
  void run() override;

  /**
   * @brief Not required / implemented for this runner.
   * 
   */
  void prepare_transformed_geometry() override;

  /**
   * @brief This function is called by the BFGS solver. It gives the next state and requests the shape gradient and the loss functional for that configuration in return.
   * 
   * In the function we set the provided values in x as the new shape parameter values. Then we solve the forward and adjoint state  and compute the shape gradient. We push the values into the input argument g which stores the gradient components and compute the loss functional which we return. Additionaly we increment the step counter.
   * @param x New shape configuration to compute.
   * @param g Return argument to write the gradient to.
   * @return double The evaluation of the loss functional for the given shape parametrization.
   */
  static double perform_step(const dealii::Vector<double> &x, dealii::Vector<double> &g);

  /**
   * @brief Assembles and solves forward and adjoint problem.
   * 
   */
  static void solve_main_problem();

  /**
   * @brief This function updates the stored shape configuration for a provided vector of dof values.
   * 
   * @param in_shape_dofs 
   */
  static void set_shape_dofs(const dealii::Vector<double> in_shape_dofs);
};
