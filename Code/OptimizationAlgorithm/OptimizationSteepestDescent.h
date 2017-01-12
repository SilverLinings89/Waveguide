#ifndef OptimizationSD_H_
#define OptimizationSD_H_

/**
 * \class OptimizationSteepestDescent
 * \brief This class implements the computation of an optimization step via a Steepest-Descent-method.
 *
 * Objects of the Type OptimizationAlgorithm are used by the class OptimizationStrategy to compute the next viable configuration based on former results. Its is encapsulated in it's own class to offer comparison and easy changing between different schemes.
 * \author Pascal Kraft
 * \date 29.11.2016
 */
class OptimizationSteepestDescent : public OptimizationAlgorithm {

public:
  OptimizationSteepestDescent();

  ~OptimizationSteepestDescent();

  virtual std::vector<double> get_configuration();

};

#endif
