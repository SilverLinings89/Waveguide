#ifndef AdjointOptimization_CPP_
#define AdjointOptimization_CPP_



#include "AdjointOptimization.h"
#include <complex>
#include <vector>
#include <deal.II/base/point.h>
#include "../Core/Waveguide.h"
#include "../MeshGenerators/MeshGenerator.h"
#include "../SpaceTransformations/SpaceTransformation.h"
#include "../OptimizationAlgorithm/OptimizationAlgorithm.h"

using namespace dealii;

AdjointOptimization::AdjointOptimization(Waveguide * in_waveguide_primal, MeshGenerator * in_mg, SpaceTransformation * in_st_primal, SpaceTransformation * in_st_dual, OptimizationAlgorithm<std::complex<double>> * in_Oa) {
  waveguide = in_waveguide_primal;
  primal_st = in_st_primal;
  dual_st = in_st_dual;
  mg = in_mg;
  oa = in_Oa;

}

AdjointOptimization::~AdjointOptimization() {

}

std::vector<std::complex<double>> AdjointOptimization::compute_small_step(double step) {
  waveguide->switch_to_primal(primal_st);
  std::complex<double> global_a_out= primal_st->evaluate_for_z(  GlobalParams.M_R_ZLength/2.0 -0.0001 , waveguide);
  std::vector<std::complex<double>> grad = waveguide->assemble_adjoint_local_contribution(step);
  for(unsigned int i = 0; i < grad.size(); i++) {
    deallog << "Phase Derivative: " << grad[i]/step << " Step: " << step << "Quality derivative: " << std::abs(global_a_out + grad[i]) - std::abs(global_a_out)<< std::endl;
  }
  return grad;
}

double AdjointOptimization::compute_big_step(std::vector<double> step) {
  Vector<double> current_config = primal_st->Dofs();

	for(unsigned int i = 0; i< step.size(); i++){
		primal_st->set_dof(i, current_config[i] + step[i]);
		dual_st->set_dof(i, current_config[i] + step[i]);
	}

	deallog.push("Config");
	Vector<double> for_output = primal_st->Dofs();
	for(unsigned int i = 0; i< step.size(); i++){
    deallog << for_output[i] << " , ";
  }
	deallog<<std::endl;
	deallog.pop();

  MPI_Barrier(MPI_COMM_WORLD);
  waveguide->switch_to_primal(primal_st);
  waveguide->run();
  MPI_Barrier(MPI_COMM_WORLD);
  double quality = 0;
  std::complex<double> a_in = primal_st->evaluate_for_z(- GlobalParams.M_R_ZLength/2.0, waveguide);
  std::complex<double> a_out= primal_st->evaluate_for_z(  GlobalParams.M_R_ZLength/2.0 -0.0001 , waveguide);
  deallog<< "Phase in: " << a_in << std::endl;
  deallog<< "Phase out: " << a_out << std::endl;
  quality = std::abs(a_out) / std::abs(a_in);
  deallog << "Computed primal quality " << quality << std::endl;
  std::ofstream result_file;
  result_file.open((solutionpath + "/complex qualities"+ GlobalParams.MPI_Rank + ".dat").c_str(),std::ios_base::openmode::_S_trunc);
  double z = -GlobalParams.M_R_ZLength/2.0 +0.00001;
  result_file << "z \t re(f) \t im(f) \t |f|" <<std::endl;
  while(z < -GlobalParams.M_R_ZLength/2.0+ GlobalParams.SystemLength){
    std::complex<double> f = primal_st->evaluate_for_z(z, waveguide);
    result_file << z << "\t" << f.real() << "\t" << f.imag() << "\t" << sqrt(f.real()*f.real() + f.imag()*f.imag()) << std::endl;
    z+= 0.2;
  }
  result_file.close();
  MPI_Barrier(MPI_COMM_WORLD);
  waveguide->switch_to_dual(dual_st);
  waveguide->run();
  MPI_Barrier(MPI_COMM_WORLD);
  double dual_quality = 0;
  std::complex<double> d_a_in = primal_st->evaluate_for_z(- GlobalParams.M_R_ZLength/2.0, waveguide);
  std::complex<double> d_a_out= primal_st->evaluate_for_z(  GlobalParams.M_R_ZLength/2.0 -0.0001 , waveguide);
  dual_quality = std::abs(d_a_out) / std::abs(d_a_in);
  deallog<< "Phase in: " << d_a_in << std::endl;
  deallog<< "Phase out: " << d_a_out << std::endl;

  deallog << "Computed dual quality " << dual_quality << std::endl;
  waveguide->switch_to_primal(primal_st);
  return quality;
}

void AdjointOptimization::run() {
  Convergence_Table.set_auto_fill_mode(true);
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
      deallog << "Got the following big step configuration: ";
      for(unsigned int i = 0; i < step.size(); i++) {
        deallog << step[i] << " , ";
      }
      deallog << std::endl;
      quality = compute_big_step(step);
      oa->pass_result_big_step(primal_st->evaluate_for_z(GlobalParams.M_R_ZLength/2.0 -0.0001 , waveguide));
    }

    counter++;

    if(counter > GlobalParams.Sc_OptimizationSteps || quality > 1.0) {
      deallog << "The optimization is shutting down after " << counter << " steps. Last quality: " << 100*quality <<"%." << std::endl;
      run = false;
    }

    if((GlobalParams.O_C_D_ConvergenceFirst || GlobalParams.O_C_D_ConvergenceAll)&& (GlobalParams.MPI_Rank==0)) {
      std::ofstream result_file;
      result_file.open((solutionpath + "/convergence_rates.dat").c_str(),std::ios_base::openmode::_S_trunc);

      Convergence_Table.write_text(result_file, dealii::TableHandler::TextOutputFormat::table_with_headers);
      result_file.close();
      result_file.open((solutionpath + "/convergence_rates.tex").c_str(),std::ios_base::openmode::_S_trunc);
      Convergence_Table.write_tex(result_file);
      result_file.close();

      result_file.open((solutionpath + "/steps.dat").c_str(),std::ios_base::openmode::_S_trunc);
      oa->WriteStepsOut(result_file);
      result_file.close();
    }



  }

}

#endif


