#ifndef FDOptimization_CPP_
#define FDOptimization_CPP_

using namespace dealii;

#include "FDOptimization.h"

FDOptimization::FDOptimization(Waveguide * in_waveguide, MeshGenerator * in_mg, SpaceTransformation * in_st, OptimizationAlgorithm * in_Oa) {
  waveguide = in_waveguide;

  oa = in_Oa;

  mg = in_mg;

}

void FDOptimization::run() {


}

#endif


