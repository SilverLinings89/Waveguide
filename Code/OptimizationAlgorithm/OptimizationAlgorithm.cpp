#ifndef OPTIMIZATION_ALGORITHM_CPP
#define OPTIMIZATION_ALGORITHM_CPP

#include "OptimizationAlgorithm.h"

template<typename datatype>
OptimizationAlgorithm<datatype>::OptimizationAlgorithm (){

}

template<typename datatype>
OptimizationAlgorithm<datatype>::~OptimizationAlgorithm() {

}

template<typename datatype>
void OptimizationAlgorithm<datatype>::pass_result_small_step(std::vector<datatype> in_step_result){
  states.push_back(in_step_result);
}

template<typename datatype>
void OptimizationAlgorithm<datatype>::pass_result_big_step(datatype in_change){
  residuals.push_back(in_change);
}

#endif
