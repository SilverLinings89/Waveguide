#ifndef FDOptimization_CPP_
#define FDOptimization_CPP_

#include "FDOptimization.h"
#include <deal.II/base/point.h>
#include <complex>
#include <vector>
#include "../Core/NumericProblem.h"
#include "../MeshGenerators/SquareMeshGenerator.h"
#include "../OptimizationAlgorithm/OptimizationAlgorithm.h"
#include "../SpaceTransformations/SpaceTransformation.h"

using namespace dealii;

FDOptimization::FDOptimization(NumericProblem *in_waveguide,
                               SquareMeshGenerator *in_mg,
                               SpaceTransformation *in_st,
                               OptimizationAlgorithm<double> *in_Oa) {
  waveguide = in_waveguide;
  st = in_st;
  oa = in_Oa;
  mg = in_mg;
}

FDOptimization::~FDOptimization() {}

double FDOptimization::evaluate() {
  double quality = 0;
  double q_in = std::abs(st->evaluate_for_z_with_sum(
      -GlobalParams.M_R_ZLength / 2.0, Evaluation_Domain::CIRCLE_CLOSE,
      Evaluation_Metric::FUNDAMENTAL_MODE_EXCITATION, waveguide));
  double q_out = std::abs(st->evaluate_for_z_with_sum(
      GlobalParams.M_R_ZLength / 2.0, Evaluation_Domain::CIRCLE_CLOSE,
      Evaluation_Metric::FUNDAMENTAL_MODE_EXCITATION, waveguide));
  quality = q_out / q_in;
  return quality;
}

std::vector<double> FDOptimization::compute_small_step(double step) {
  unsigned int ndofs = st->NDofs();
  std::complex<double> global_a_out = st->evaluate_for_z_with_sum(
      GlobalParams.M_R_ZLength / 2.0, Evaluation_Domain::CIRCLE_CLOSE,
      Evaluation_Metric::FUNDAMENTAL_MODE_EXCITATION, waveguide);
  std::vector<double> ret;
  ret.resize(ndofs);
  double q_old = evaluate();
  for (unsigned int i = 0; i < ndofs; i++) {
    double old_dof_value = st->get_dof(i);
    if (st->IsDofFree(i)) {
      st->set_dof(i, old_dof_value + step);
      waveguide->run();
      ret[i] = (evaluate() - q_old) / step;

      std::complex<double> a_in = st->evaluate_for_z_with_sum(
          -GlobalParams.M_R_ZLength / 2.0, Evaluation_Domain::CIRCLE_CLOSE,
          Evaluation_Metric::FUNDAMENTAL_MODE_EXCITATION, waveguide);
      std::complex<double> a_out = st->evaluate_for_z_with_sum(
          GlobalParams.M_R_ZLength / 2.0, Evaluation_Domain::CIRCLE_CLOSE,
          Evaluation_Metric::FUNDAMENTAL_MODE_EXCITATION, waveguide);
      deallog << "Phase in: " << a_in;
      deallog << " Phase out: " << a_out;
      deallog << " Quality derivative: " << ret[i];
      deallog << " Step: " << step;
      deallog << " Phase derivative: " << (a_out - global_a_out) / step
              << std::endl;
      st->set_dof(i, old_dof_value);
    } else {
      ret[i] = 0.0;
    }
  }
  return ret;
}

double FDOptimization::compute_big_step(std::vector<double> step) {
  Vector<double> current_config = st->Dofs();
  for (unsigned int i = 0; i < step.size(); i++) {
    st->set_dof(i, current_config[i] + step[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  waveguide->switch_to_primal(st);
  waveguide->run();
  MPI_Barrier(MPI_COMM_WORLD);

  return evaluate();
}

void FDOptimization::run() {
  Convergence_Table.set_auto_fill_mode(true);

  bool run = true;
  int counter = 0;
  double quality = 0;

  while (run) {
    int small_steps = 0;
    while (oa->perform_small_step_next(small_steps)) {
      deallog << "Performing a small step." << std::endl;
      double temp_step_width = oa->get_small_step_step_width(small_steps);
      oa->pass_result_small_step(compute_small_step(temp_step_width));
      small_steps++;
    }

    if (oa->perform_big_step_next(small_steps)) {
      deallog << "Performing a big step." << std::endl;
      std::vector<double> step = oa->get_big_step_configuration();
      deallog << "Got the following big step configuration: ";
      for (unsigned int i = 0; i < step.size(); i++) {
        deallog << step[i] << " , ";
      }
      deallog << std::endl;
      quality = compute_big_step(step);
      oa->pass_result_big_step(quality);
    }

    counter++;

    if (counter >= GlobalParams.Sc_OptimizationSteps || quality > 1.0) {
      run = false;
      std::cout << "The optimization is shutting down after " << counter
                << " steps. Last quality: " << 100 * quality << "%"
                << std::endl;
    }

    if ((GlobalParams.O_C_D_ConvergenceFirst ||
         GlobalParams.O_C_D_ConvergenceAll) &&
        (GlobalParams.MPI_Rank == 0)) {
      std::ofstream result_file;
      result_file.open((solutionpath + "/convergence_rates.dat").c_str(),
                       std::ios_base::openmode::_S_trunc);

      Convergence_Table.write_text(
          result_file,
          dealii::TableHandler::TextOutputFormat::table_with_headers);
      result_file.close();
      result_file.open((solutionpath + "/convergence_rates.tex").c_str(),
                       std::ios_base::openmode::_S_trunc);
      Convergence_Table.write_tex(result_file);
      result_file.close();

      result_file.open((solutionpath + "/steps.dat").c_str(),
                       std::ios_base::openmode::_S_trunc);
      oa->WriteStepsOut(result_file);
      result_file.close();
    }
  }
}

#endif
