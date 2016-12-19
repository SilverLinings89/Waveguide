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
class OptimizationAlgorithm {

public:
  OptimizationAlgorithm();

  virtual ~OptimizationAlgorithm();

  /**
   * This function can be used by the optimization strategy to pass a residual into the history of the algorithm. History aware methods such as GMRES profit from its storage.
   * \param in_residual The double valued residual to be stored.
   */
  virtual void pass_residual(double in_residual) = 0;

  /**
   * This function can be used to store a shape gradient inside the algorithm. Outside this object an underlying data-structure might need to be cleared before the next step so storing this makes it persistent. The generated amount of data does not have to be considered for storage constraints since it is minimal even for many (hundreds) dofs.
   * \param in_gradient The gradient vector to be stored. It is the most important functionality of the entire system to estimate this vector accurately in short time.
   */
  virtual void pass_gradient(std::vector<double> in_gradient) = 0;

  /**
   * This function returns the next configuration based on the currently stored values of residual and vectors and the latest shape gradient.
   */
  virtual std::vector<double> get_configuration() = 0;

};

#endif OptimizationAlgorithm_H_
