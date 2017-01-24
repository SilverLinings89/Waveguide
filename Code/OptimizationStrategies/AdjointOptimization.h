#ifndef AdjointOptimization_H_
#define AdjointOptimization_H_

using namespace dealii;

/**
 * \class AdjointOptimization
 * \brief Derived from the Optimization class, this class implements an Optimization-scheme based on an adjoint method.
 *
 * This method should prove to be far superior to a finite difference approach as soon as the shape has more then 2 degrees of freedom since its effort is always a total of 2 forward problems to solve.
 * \author Pascal Kraft
 * \date 29.11.2016
 */
class AdjointOptimization : public Optimization {

public:
  const int type = 1; // Allows callers to identify the exact type easily. 0 = FD, 1 = Adj.

  AdjointOptimization(Waveguide * waveguide_primal, Waveguide * wavegudie_dual, MeshGenerator * mg, SpaceTransformation * st_primal, SpaceTransformation * st_dual, OptimizationAlgorithm<std::complex<double>> * Oa);

  ~AdjointOptimization();

  std::vector<std::complex<double>> compute_small_step(double step);

  void compute_big_step();

  void run();

  Waveguide * primal_waveguide;
  Waveguide * dual_waveguide;

  SpaceTransformation * primal_st;
  SpaceTransformation * dual_st;

  MeshGenerator * mg;

  OptimizationAlgorithm<std::complex<double>> * oa;

};

#endif
