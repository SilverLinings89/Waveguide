#ifndef AdjointOptimization_CPP_
#define AdjointOptimization_CPP_



#include "AdjointOptimization.h"
#include <complex>
#include <vector>
#include <string>
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
  // New starts here
  const double step_width = 0.05;
  unsigned int cnt_steps = 0;
  double z_temp = -GlobalParams.M_R_ZLength/2.0 +0.00001;
  while ( z_temp < -GlobalParams.M_R_ZLength/2.0+ GlobalParams.SystemLength){
    cnt_steps ++;
    z_temp += step_width;
  }
  deallog << "Start" << std::endl;
  bool* mine = new bool[cnt_steps];
  z_temp = -GlobalParams.M_R_ZLength/2.0 +0.00001;
  unsigned int own_cnt = 0;
  double lowest_own = -GlobalParams.SystemLength;
  int lowest_idx = 0;
  for(unsigned int i=0; i< cnt_steps; i++) {
    if(z_temp > this->waveguide->mg->z_min && z_temp < this->waveguide->mg->z_max) {
      mine[i] = true;
      if(own_cnt == 0) {
        lowest_own = z_temp;
        lowest_idx = i;
      }
      own_cnt ++;
    } else {
      mine[i] = false;
    }
    z_temp += step_width;
  }
  int * displs;
  if(GlobalParams.MPI_Rank == 0) {
    displs = new int[GlobalParams.NumberProcesses];
  }
  std::cout << "I am " << GlobalParams.MPI_Rank << ". I have lowest own " << lowest_own << " and own_cnt " << own_cnt << std::endl;
  MPI_Gather(&lowest_idx, 1, MPI_INTEGER, displs, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
  double* a_reals = new double[own_cnt];
  double* a_imags = new double[own_cnt];
  double* a_abolutes = new double[own_cnt];
  z_temp = lowest_own;
  for(unsigned int i = 0; i < own_cnt; i++) {
    std::complex<double> l_val(0,0);
    try {
      l_val = primal_st->gauss_product_2D_sphere(z_temp,10,(GlobalParams.M_C_Dim1In+GlobalParams.M_C_Dim1Out),0,0, waveguide);
    } catch(...) {
      std::cout << "In Process " << GlobalParams.MPI_Rank << ": Broke for " << z_temp << std::endl;
    }
    a_reals[i] = l_val.real();
    a_imags[i] = l_val.imag();
    a_abolutes[i] = std::sqrt(a_reals[i]*a_reals[i] + a_imags[i]*a_imags[i]);
    z_temp += step_width;
  }
  std::cout << "I am " << GlobalParams.MPI_Rank << ". My dofs are ";
  for (int i = 0;i < cnt_steps ; i++) {
    if(mine[i]) {
      std::cout << i << " ";
    }
  }
  std::cout << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  double* all_reals;
  double* all_imags;
  double* all_absolutes;
  int * cnt_recv;
  if(GlobalParams.MPI_Rank == 0) {
    all_reals = new double[cnt_steps];
    all_imags = new double[cnt_steps];
    all_absolutes = new double[cnt_steps];
    cnt_recv = new int[GlobalParams.NumberProcesses];
    cnt_recv[0] = own_cnt;
    for(unsigned int i =1; i < GlobalParams.NumberProcesses-1; i++) {
      cnt_recv[i] = displs[i+1]-displs[i];
    }
    cnt_recv[GlobalParams.NumberProcesses-1] = cnt_steps - displs[GlobalParams.NumberProcesses-1];
  }
  deallog << "Output proc " << GlobalParams.MPI_Rank << "." <<std::endl;
  for(unsigned int i = 0; i < own_cnt; i++) {
    deallog << a_reals[i] << " " << a_imags[i] << " " << a_abolutes[i] << std::endl;
  }
  std::cout << "Ready proc " << GlobalParams.MPI_Rank << "." <<std::endl;
  MPI_Gatherv(a_reals, own_cnt, MPI_DOUBLE, all_reals, cnt_recv, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(a_imags, own_cnt, MPI_DOUBLE, all_imags, cnt_recv, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(a_abolutes, own_cnt, MPI_DOUBLE, all_absolutes, cnt_recv, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(GlobalParams.MPI_Rank == 0){
    std::ofstream result_file;
    result_file.open((solutionpath + "/complex qualities.dat").c_str(),std::ios_base::openmode::_S_trunc);
    result_file << "z \t re(f) \t im(f) \t |f|" <<std::endl;
    z_temp = -GlobalParams.M_R_ZLength/2.0;
    for(unsigned int i = 0; i < cnt_steps; i++) {
      result_file << z_temp<< "\t" << all_reals[i] << "\t" << all_imags[i] << "\t" << all_absolutes[i] << std::endl;
      z_temp += step_width;
    }
    result_file.close();
  }

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


