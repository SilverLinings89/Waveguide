#ifndef FDOptimization_CPP_
#define FDOptimization_CPP_

using namespace dealii;

#include "FDOptimization.h"

FDOptimization::FDOptimization(Waveguide * in_waveguide, MeshGenerator * in_mg, SpaceTransformation * in_st, OptimizationAlgorithm * in_Oa) {
  waveguide = in_waveguide;

  st = in_st;

  oa = in_Oa;

  mg = in_mg;

}

FDOptimization::~FDOptimization() {

}

void FDOptimization::run() {


}

#endif


