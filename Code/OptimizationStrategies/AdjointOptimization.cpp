#ifndef AdjointOptimization_CPP_
#define AdjointOptimization_CPP_

using namespace dealii;

#include "AdjointOptimization.h"

AdjointOptimization::AdjointOptimization(Waveguide * in_waveguide_primal, Waveguide * in_waveguide_dual, MeshGenerator * in_mg, SpaceTransformation * in_st_primal, SpaceTransformation * in_st_dual, OptimizationAlgorithm * in_Oa) {
  primal_waveguide = in_waveguide_primal;
  primal_st = in_st_primal;
  dual_waveguide = in_waveguide_dual;
  dual_st = in_st_dual;
  mg = in_mg;
  oa = in_Oa;
}

AdjointOptimization::~AdjointOptimization() {

}

void AdjointOptimization::run() {


}

#endif


