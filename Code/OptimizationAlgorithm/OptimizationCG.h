#ifndef OptimizationCG_H_
#define OptimizationCG_H_

/**
 * \class OptimizationCG
 * \brief This class implements the computation of an optimization step via a CG-method.
 *
 * Objects of the Type OptimizationAlgorithm are used by the class OptimizationStrategy to compute the next viable configuration based on former results. Its is encapsulated in it's own class to offer comparison and easy changing between differenct schemes.
 * \author Pascal Kraft
 * \date 29.11.2016
 */
class OptimizationCG : public OptimizationAlgorithm {

  OptimizationCG();

  ~OptimizationCG();

  virtual void pass_residual(double in_residual);

  virtual void pass_gradient(std::vector<double> in_gradient);

  virtual std::vector<double> get_configuration();

};

#endif OptimizationCG_H_
