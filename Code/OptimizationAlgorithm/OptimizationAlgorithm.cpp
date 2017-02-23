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
  deallog.push("Small Step Result");
  bool complex = std::is_same<datatype, std::complex<double>>::value;
  if(complex) {
    std::vector<double> values;
    values.resize(2*in_step_result.size());
    for(int i= 0; i < in_step_result.size(); i++) {
      values[2*i] = ((std::complex<double>)in_step_result[i]).real();
      values[2*i + 1] = ((std::complex<double>)in_step_result[i]).imag();
    }
    dealii::Utilities::MPI::sum(values, MPI_COMM_WORLD, values);
    for(int i = 0; i < in_step_result.size(); i++) {
      ((std::complex<double>)in_step_result[i]).real(values[2*i]);
      ((std::complex<double>)in_step_result[i]).imag(values[2*i+1]);
    }
  } else {
    /**
    std::vector<double> values;
    values.resize(in_step_result.size());
    for(int i = 0 ; i < in_step_result.size(); i++) {
      values[i]= (double)in_step_result[i];
    }
    dealii::Utilities::MPI::sum(values, MPI_COMM_WORLD, values);
    for(int i = 0 ; i < in_step_result.size(); i++) {
        (double)in_step_result[i]= values[i];
    }
    **/
  }

  for(int i = 0; i < in_step_result.size(); i++) {
    deallog<<in_step_result[i];
    if(i != in_step_result.size()-2) {
      deallog<< " , " ;
    }
  }
  deallog << std::endl;
  deallog.pop();
  states.push_back(in_step_result);
}

template<typename datatype>
void OptimizationAlgorithm<datatype>::pass_result_big_step(datatype in_change){
  deallog.push("Big Step Result passed");
  deallog << "Result:" << in_change <<std::endl;
  residuals.push_back(in_change);
}

#endif
