#ifndef OptimizationAlgorithm_H_
#define OptimizationAlgorithm_H_

#include "../Core/NumericProblem.h"

/**
 * \class OptimizationAlgorithm
 * \brief This class is an interface for Optimization algorithms such as CG or
 * steepest descent.
 *
 * The derived classes take residuals and gradients, store them in a history and
 * compute the next configuration based on the data. This functionality is
 * encapsulated like this to enable easy exchange and comparison of convergence
 * rates. Later on the interface will be extended to make use of output
 * generators during runtime or at the end of the program to directly generate
 * convergence plots. This class will also be extended to allow for restrained
 * optimization which will become necessary at some point. \author Pascal Kraft
 * \date 29.11.2016
 */

template <typename datatype>
class OptimizationAlgorithm {
 public:
  std::vector<std::vector<datatype>> states;

  std::vector<datatype> residuals;

  OptimizationAlgorithm();

  ~OptimizationAlgorithm();

  /**
   * A function to pass a vector of values, computed by performing a single or
   * multiple steps with short step-width. "small" in the name references the
   * fact, that the step width is small. In gerneral this is done whenever an
   * accurat approximation of a gradient is saught based on linearization. This
   * computation (especially in finite difference based approaches) can be much
   * more costly than a big step. In a big step, one forward problem has to be
   * solved. A small step computation based on finite differences requires
   * \f$NDofs\f$ forward problems to be solved. This function has the purpose of
   * passing the result of such computations to the optimization algorithm which
   * will store it and use it to compute optimization steps in the future.
   * \param vec This parameter is a vector of changes of the target functional
   * based on a change in the individual component. The components belonging to
   * restrained degrees of freedom are set to zero.
   */
  virtual void pass_result_small_step(std::vector<datatype> vec);

  /**
   * Similar to pass_result_small_step but for the results of big steps.
   * Since for a big step we always only perform the solution of one forward
   * problem we also only get one change of the target functional. Therefore in
   * this case we only pass a value, not a vector of the like. \param input This
   * is the value describing how much the target functional has changed upon
   * performing the step last computed by this optimization algorithm.
   */
  virtual void pass_result_big_step(datatype input);

  /**
   * The optimization is mainly split into two kinds of steps: Full and small
   * steps. For FD based schemes, a small step is a computation of finite
   * differences for all degrees of freedom which entails a lot of computation.
   * Small here refers to the norm of the step width - not necessarily to the
   * amount of computation required. In general this function is supposed to
   * gather information about the target functional around the current state.
   * \param small_steps_before this number tells the scheme how many small steps
   * were performed before the current request. \return this is true, if the
   * Optimization Scheme requires more small steps before a big step can be
   * performed.
   */
  virtual bool perform_small_step_next(int small_steps_before) = 0;

  /**
   * For the optimization scheme to know, which step size is appropriate, this
   * function was included. \param small_steps_before similar to
   * perform_small_step_next this is the number of small steps before the
   * current one. \return double this is how much the values of the degrees of
   * freedom should be adapted.
   */
  virtual double get_small_step_step_width(int small_steps_before) = 0;

  /**
   * This functions returns true, if enough steps were performed to compute the
   * next state to compute a full solution on. \param small_steps_before number
   * of small steps performed before this call. \return true, if the next
   * computation should be a big step - otherwise false.
   */
  virtual bool perform_big_step_next(int small_steps_before) = 0;

  /**
   * This function computes the states that should be computed next. If the next
   * step will be a small step the update can be done by simply updating all
   * dofs with a step width (or only one depending on the pattern) so this
   * function is only used when a big step will be computed next and therefore
   * all dofs could change differently. \return This is a vector of degrees of
   * freedom which can be used by the Optimization Strategy to update the Space
   * Transformation.
   *
   */
  virtual std::vector<double> get_big_step_configuration() = 0;

  void WriteStepsOut(std::ofstream &);
};

#endif
