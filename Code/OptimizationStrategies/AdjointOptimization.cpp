#ifndef AdjointOptimization_CPP_
#define AdjointOptimization_CPP_

using namespace dealii;

#include "../Core/Waveguide.h"
#include "AdjointOptimization.h"

AdjointOptimization::AdjointOptimization(Waveguide * in_waveguide_primal, Waveguide * in_waveguide_dual, MeshGenerator * in_mg, SpaceTransformation * in_st_primal, SpaceTransformation * in_st_dual, OptimizationAlgorithm<std::complex<double>> * in_Oa) {
  primal_waveguide = in_waveguide_primal;
  primal_st = in_st_primal;
  dual_waveguide = in_waveguide_dual;
  dual_st = in_st_dual;
  mg = in_mg;
  oa = in_Oa;

}

AdjointOptimization::~AdjointOptimization() {

}

std::vector<std::complex<double>> AdjointOptimization::compute_small_step(double step) {
  return primal_waveguide->assemble_adjoint_local_contribution(dual_waveguide, step);
}

double AdjointOptimization::compute_big_step(std::vector<double> step) {
  Vector<double> current_config = primal_st->Dofs();
	for(unsigned int i = 0; i< step.size(); i++){
		primal_st->set_dof(i, current_config[i] + step[i]);
		dual_st->set_dof(i, current_config[i] + step[i]);
	}
    MPI_Barrier(MPI_COMM_WORLD);
    primal_waveguide->run();
    MPI_Barrier(MPI_COMM_WORLD);
    dual_waveguide->run();
    MPI_Barrier(MPI_COMM_WORLD);

    double quality = 0;
    double q_in  = std::abs(primal_st->evaluate_for_z(- GlobalParams.M_R_ZLength/2.0, primal_waveguide));
    double q_out = std::abs(primal_st->evaluate_for_z(  GlobalParams.M_R_ZLength/2.0, primal_waveguide));
    quality = q_out / q_in;

    deallog.push("AO::compute_big_step");
    deallog << "Computed quality " << quality << std::endl;
    deallog.pop();
    return quality;
}

void AdjointOptimization::run() {
  bool run = true;
  int counter = 0;
  double quality =0;

  while(run) {
    int small_steps = 0;
    while(oa->perform_small_step_next(small_steps)) {
      deallog << "Performing a small step." << std::endl;
      double temp_step_width = oa->get_small_step_step_width(small_steps);
      oa->pass_result_small_step(compute_small_step(temp_step_width));
      small_steps++;
    }

    if(oa->perform_big_step_next(small_steps)) {
      deallog << "Performing a big step." << std::endl;
      std::vector<double> step = oa->get_big_step_configuration();
      quality = compute_big_step(step);
      oa->pass_result_big_step(primal_st->evaluate_for_z(GlobalParams.M_R_ZLength/2.0, primal_waveguide));
    }

    counter++;

    if(counter > GlobalParams.Sc_OptimizationSteps || quality > 1.0) {
      deallog << "The optimization is shutting down after " << counter << " steps. Last quality: " << 100*quality <<"%." << std::endl;
      run = false;
    }
  }

}

#endif


