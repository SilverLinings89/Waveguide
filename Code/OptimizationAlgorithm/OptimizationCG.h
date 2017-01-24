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
class OptimizationCG : public OptimizationAlgorithm<double> {

public:
  OptimizationCG();

  ~OptimizationCG();

  virtual std::vector<double> get_configuration();

  bool perform_small_step_next();

  double get_small_step_step_width();

  bool perform_small_big_next();

  std::vector<double> get_big_step_configuration();

};

#endif
