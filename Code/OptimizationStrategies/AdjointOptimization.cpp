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

std::vector<std::complex<double>> AdjointOptimization::compute_small_step(double step) {
  return primal_waveguide->assemble_adjoint_local_contribution(dual_waveguide, step);
}

void AdjointOptimization::compute_big_step() {
    MPI_Barrier(MPI_COMM_WORLD);
    primal_waveguide->run();
    MPI_Barrier(MPI_COMM_WORLD);
    dual_waveguide->run();
    MPI_Barrier(MPI_COMM_WORLD);
}

void AdjointOptimization::run() {
  bool run = true;
  int counter = 0;

  while(run) {
    compute_big_step();
    double quality = 0;
    double q_in = std::abs(primal_st->evaluate_for_z(- GlobalParams.M_R_ZLength/2.0, primal_waveguide));
    double q_out = std::abs(primal_st->evaluate_for_z(GlobalParams.M_R_ZLength/2.0, primal_waveguide));
    quality = q_out / q_in;
    counter++;
    std::vector<std::complex<double>> qualities;

    // At this point both solutions are known and we now deal with the computation of shape gradients based on this knowledge.

    oa->pass_full_step(quality, primal_st->Dofs());

    oa->estimate();

    oa->get_configuration();


    const int steps = 5;
    double * steps_widths = new double[steps];
    steps_widths[0] = 0.00001;
    steps_widths[1] = 0.0001;
    steps_widths[2] = 0.001;
    steps_widths[3] = 0.01;
    steps_widths[4] = 0.1;

    qualities.reserve(steps * 2 * primal_st->NFreeDofs());

    for(int i = 0; i < steps; i++) {
      std::vector<std::complex<double>> temp_results;
      temp_results.reserve(primal_st->NFreeDofs());
      temp_results = primal_waveguide->assemble_adjoint_local_contribution(dual_waveguide, steps_widths[i]);
      for(int j = 0; j < primal_st->NFreeDofs(); j++) {
        qualities[i*primal_st->NFreeDofs() + j ]= temp_results[j];
      }
    }

    for(int i = 0; i < steps; i++) {
        std::vector<std::complex<double>> temp_results;
        temp_results.reserve(primal_st->NFreeDofs());
        temp_results = primal_waveguide->assemble_adjoint_local_contribution(dual_waveguide, -steps_widths[i]);
        for(int j = 0; j < primal_st->NFreeDofs(); j++) {
          qualities[(steps + i)*primal_st->NFreeDofs() + j ]= temp_results[j];
        }
    }

    if(counter > GlobalParams.Sc_OptimizationSteps || quality > 1.0) {
      run = false;
      pout << "The optimization is shutting down after " << counter << " steps. Last quality: " << 100* quality <<"%" <<std::endl;
    }
  }

}

#endif


