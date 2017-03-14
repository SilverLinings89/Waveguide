#ifndef FDOptimization_CPP_
#define FDOptimization_CPP_

using namespace dealii;

#include "FDOptimization.h"

FDOptimization::FDOptimization(Waveguide * in_waveguide, MeshGenerator * in_mg, SpaceTransformation * in_st, OptimizationAlgorithm<double> * in_Oa) {
  waveguide = in_waveguide;

  st = in_st;

  oa = in_Oa;

  mg = in_mg;

}

FDOptimization::~FDOptimization() {

}

double FDOptimization::evaluate() {
  double quality = 0;
  double q_in = std::abs(st->evaluate_for_z(- GlobalParams.M_R_ZLength/2.0, waveguide));
  double q_out = std::abs(st->evaluate_for_z(GlobalParams.M_R_ZLength/2.0, waveguide));
  quality = q_out / q_in;
  return quality;
}

std::vector<double> FDOptimization::compute_small_step(double step) {
  unsigned int ndofs = st->NDofs();
  std::vector<double> ret;
  ret.reserve(ndofs);
  double q_old = evaluate();
  for (unsigned int i =0; i < ndofs; i++) {
    double old_dof_value = st->get_dof(i);
    if(st->IsDofFree(i)) {
      st->set_dof(i, old_dof_value + step);
      waveguide->run();
      ret[i] = evaluate() - q_old;
    } else {
      ret[i] = 0.0;
    }

  }
  return ret;
}

double FDOptimization::compute_big_step(std::vector<double> step) {
  Vector<double> current_config = st->Dofs();
  for(unsigned int i = 0; i< step.size(); i++){
    st->set_dof(i, current_config[i] + step[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  waveguide->run();
  MPI_Barrier(MPI_COMM_WORLD);

  return evaluate();
}


void FDOptimization::run() {
  bool run = true;
  int counter = 0;
  double quality =0;

  while(run) {
    int small_steps = 0;
    while(oa->perform_small_step_next(small_steps)) {
      double temp_step_width = oa->get_small_step_step_width(small_steps);
      oa->pass_result_small_step(compute_small_step(temp_step_width));
      small_steps++;
    }

    if(oa->perform_big_step_next(small_steps)) {
      std::vector<double> step = oa->get_big_step_configuration();
      quality = compute_big_step(step);
      oa->pass_result_big_step(quality);
    }

    counter++;

    if(counter > GlobalParams.Sc_OptimizationSteps || quality > 1.0) {
      run = false;
      std::cout << "The optimization is shutting down after " << counter << " steps. Last quality: " << 100* quality <<"%" <<std::endl;
    }
  }

}

#endif


