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

template<typename datatype>
void OptimizationAlgorithm<datatype>::increment_small_step_counter() {
  small_step_counter = small_step_counter +1;
}

template<typename datatype>
void OptimizationAlgorithm<datatype>::increment_big_step_counter() {
  big_step_counter = big_step_counter +1;
}

#endif
