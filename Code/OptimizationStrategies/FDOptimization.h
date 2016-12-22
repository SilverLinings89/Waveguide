#ifndef FDOptimization_H_
#define FDOptimization_H_

using namespace dealii;

/**
 * \class FDOptimization
 * \brief Derived from the Optimization class, this class implements an Optimization-scheme based on finite differences.
 *
 * The idea here is to compute the solution of one forward problem per entry in the shape gradient. For \f$N\$ degrees of freedom available for the shape parametrization, this results in a total of \f$N+1\f$ forward problems to be solved. Sooner or later we aim at implementing an adjoint optimization scheme, which should come at a much lower cost.
 * \author Pascal Kraft
 * \date 29.11.2016
 */
class FDOptimization : public Optimization {

public:
  const int type = 0; // Allows callers to identify the exact type easily. 0 = FD, 1 = Adj.

  FDOptimization();

  FDOptimization(Waveguide * waveguide_primal, MeshGenerator * mg, SpaceTransformation * st_primal);

  ~FDOptimization();

  /**
   * The advantage of this formulation is the fact, that we don't need to differentiate between a 'normal' forward problem and it's dual which (in a parallel computation) holds some difficulties.
   * We can simply adapt the shape parameters to account for the change in one component and rerun the solver and assembly process.
   */
  virtual void run();

};

#endif
