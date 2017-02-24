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
    for(unsigned int i= 0; i < in_step_result.size(); i++) {
      values[2*i] = ((std::complex<double>)in_step_result[i]).real();
      values[2*i + 1] = ((std::complex<double>)in_step_result[i]).imag();
    }
    dealii::Utilities::MPI::sum(values, MPI_COMM_WORLD, values);
    for(unsigned int i = 0; i < in_step_result.size(); i++) {
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

  for(unsigned int i = 0; i < in_step_result.size(); i++) {
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
  deallog.pop();
  residuals.push_back(in_change);
}

template<typename datatype>
void OptimizationAlgorithm<datatype>::WriteStepsOut(std::ofstream & result_file ) {
  std::string * names = new std::string[3] ;
  names[0] = "Radius ";
  names[1] = "Shift ";
  names[2] = "Angle ";
  bool complex = std::is_same<datatype, std::complex<double>>::value;
  for(unsigned int l = 0; l < states.size(); l++){
    for(unsigned int i =0; i < 3; i++) {
      for(unsigned int j = 0; j < states[0].size()/3; j++) {

        if(complex) {
          std::complex<double> temp = (std::complex<double>)states[l][3*j + i];
          Optimization_Steps.add_value(names[i] + std::to_string(j) + "r", temp.real());
          Optimization_Steps.add_value(names[i] + std::to_string(j) + "i", temp.imag());
        } else {
          //Optimization_Steps.add_value(names[i] + std::to_string(j) , (double)states[l][3*j + i]);
        }

      }
    }
    Optimization_Steps.add_value("Step Width", steps_widths[l%STEPS_PER_DOFS]);
  }
  for(unsigned int i =0; i < 3; i++) {
    for(unsigned int j = 0; j < states[0].size()/3; j++) {
      if(complex){
        Optimization_Steps.add_column_to_supercolumn(names[i] + std::to_string(j)+ "r", names[i]);
        Optimization_Steps.add_column_to_supercolumn(names[i] + std::to_string(j)+ "i", names[i]);
      } else {
        Optimization_Steps.add_column_to_supercolumn(names[i] + std::to_string(j), names[i]);
      }
    }
  }
  Optimization_Steps.write_tex(result_file);
}
#endif
