#ifndef Optimization1D_H_
#define Optimization1D_H_

/**
 * \class Optimization1D
 * \brief This class implements the computation of an optimization step by doing 1D optimization based on an adjoint scheme.
 *
 * Objects of the Type OptimizationAlgorithm are used by the class OptimizationStrategy to compute the next viable configuration based on former results. Its is encapsulated in it's own class to offer comparison and easy changing between differenct schemes.
 * \author Pascal Kraft
 * \date 9.1.2017
 */
class Optimization1D : public OptimizationAlgorithm {

public:
  Optimization1D( );

  ~Optimization1D();

  virtual void pass_residual(double in_residual);

  virtual void pass_gradient(std::vector<double> in_gradient);

  virtual std::vector<double> get_configuration();

};

#endif
