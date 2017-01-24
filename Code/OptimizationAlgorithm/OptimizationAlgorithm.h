#ifndef OptimizationAlgorithm_H_
#define OptimizationAlgorithm_H_

/**
 * \class OptimizationAlgorithm
 * \brief This class is an interface for Optimization algorithms such as CG or steepest descent.
 *
 * The derived classes take residuals and gradients, store them in a history and compute the next configuration based on the data. This functionality is encapsulated like this to enable easy exchange and comparison of convergence rates.
 * Later on the interface will be extended to make use of output generators during runtime or at the end of the program to directly generate convergence plots. This class will also be extended to allow for restrained optimization which will become necessary at some point.
 * \author Pascal Kraft
 * \date 29.11.2016
 */

template <typename datatype>
class OptimizationAlgorithm {

public:

  std::vector<std::vector<datatype>> states;

  std::vector<datatype> residuals;

  OptimizationAlgorithm();

  virtual ~OptimizationAlgorithm();

  /**
   * This function can be used by the optimization strategy to pass a residual into the history of the algorithm. History aware methods such as GMRES profit from its storage.
   * \param in_residual The double valued residual to be stored.
   */

  virtual void pass_result_small_step(std::vector<datatype>);


  virtual void pass_result_big_step(datatype);

  /**
   * This function returns the next configuration based on the currently stored values of residual and vectors and the latest shape gradient.
   */
  virtual std::vector<double> get_configuration() = 0;

  virtual bool perform_small_step_next( int small_steps_before ) = 0;

  virtual double get_small_step_step_width( int small_steps_before ) = 0;

  virtual bool perform_small_big_next( int small_steps_before ) =0 ;

  virtual std::vector<double> get_big_step_configuration() =0;

};

#endif
