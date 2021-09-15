#include "../Core/Types.h"
#include "LocalProblem.h"
#include "../BoundaryCondition/HSIESurface.h"
#include "../BoundaryCondition/PMLSurface.h"
#include "../BoundaryCondition/BoundaryCondition.h"
#include "../Helpers/staticfunctions.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>
#include <string>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/solver_idr.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_project.h>
#include <deal.II/numerics/vector_tools_project.templates.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_point_value.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/matrix_free.templates.h>
#include <mpi.h>
#include "../Helpers/PointSourceField.h"
#include "../Solutions/ExactSolutionRamped.h"
#include "../Solutions/ExactSolutionConjugate.h"


template void dealii::VectorTools::project<3,dealii::Vector<ComplexNumber>,3>(const dealii::Mapping<3, 3> &, const dealii::DoFHandler<3, 3> &, const Constraints &,
        const dealii::Quadrature<3> &,
        const dealii::Function<3, ComplexNumber> &,
        dealii::Vector<ComplexNumber> &,
        const bool,
        const dealii::Quadrature<2> &,
        const bool);

LocalProblem::LocalProblem() :
  HierarchicalProblem(0, SweepingDirection::Z), 
  sc(), 
  solver(sc, MPI_COMM_SELF) {
    print_info("Local Problem", "Done building base problem. Preparing matrix.");
    matrix = new dealii::PETScWrappers::MPI::SparseMatrix();
    for(unsigned int i = 0; i < 6; i++) Geometry.levels[0].is_surface_truncated[i] = true;
    if((GlobalParams.prescribe_0_on_input_side || (!GlobalParams.use_tapered_input_signal)) && GlobalParams.Index_in_z_direction == 0) {
      // Geometry.levels[0].is_surface_truncated[4] = false;
    }
    own_dofs = Geometry.levels[0].dof_distribution[0];
}

LocalProblem::~LocalProblem() {
  delete matrix;
}

auto LocalProblem::get_center() -> Position const {
  return Geometry.get_local_center();
}

dealii::IndexSet LocalProblem::compute_interface_dof_set(BoundaryId interface_id) {
  BoundaryId opposing_interface_id = opposing_Boundary_Id(interface_id);
  dealii::IndexSet ret(Geometry.levels[0].n_local_dofs);
  for(unsigned int i = 0; i < 6; i++) {
    if( i == interface_id) {
      std::vector<InterfaceDofData> current = Geometry.levels[level].inner_domain->get_surface_dof_vector_for_boundary_id(interface_id);
      for(unsigned int j = 0; j < current.size(); j++) {
        ret.add_index(current[j].index);
      }      
    } else {
      if(i != opposing_interface_id && Geometry.levels[0].is_surface_truncated[i]) {
        std::vector<InterfaceDofData> current = Geometry.levels[0].surfaces[i]->get_dof_association_by_boundary_id(i);
        for(unsigned int j = 0; j < current.size(); j++) {
          ret.add_index(current[j].index);
        }
      }
    }
  }
  return ret;
}

void LocalProblem::initialize() {
  print_info("LocalProblem::initialize", "Start");
  GlobalTimerManager.switch_context("initialize", level);
  print_info("LocalProblem::initialize", "Number of local dofs: " + std::to_string(Geometry.levels[0].n_local_dofs) , false, LoggingLevel::DEBUG_ALL);
  for(unsigned int i = 0; i < 6; i++) {
    if(Geometry.levels[0].is_surface_truncated[i]){
      surface_dof_associations[i] = Geometry.levels[0].surfaces[i]->get_dof_association();
    }
  }
  for(unsigned int i = 0; i < 6; i++) {
    surface_index_sets[i] = compute_interface_dof_set(i);
  }
  if(GlobalParams.NumberProcesses == 1) {
    reinit();
  }
  print_info("LocalProblem::initialize", "End");
}

void LocalProblem::validate() {
  print_info("LocalProblem::validate", "Start");
  print_info("LocalProblem::validate", "Matrix size: (" + std::to_string(matrix->m()) + " x "  + std::to_string(matrix->n()) + ") and l1-norm " + std::to_string(matrix->l1_norm()), false, LoggingLevel::PRODUCTION_ONE);
  print_info("LocalProblem::validate", "End");
}

void LocalProblem::assemble() {
  Geometry.levels[level].inner_domain->assemble_system(&constraints, matrix, &rhs);
  GlobalTimerManager.switch_context("assemble", level);
  for (unsigned int surface = 0; surface < 6; surface++) {
    if(Geometry.levels[0].is_surface_truncated[surface]) {
      Geometry.levels[0].surfaces[surface]->fill_matrix(matrix, &rhs, &constraints);
    }
  }
  matrix->compress(dealii::VectorOperation::add);
  rhs.compress(dealii::VectorOperation::add);
}

void LocalProblem::reinit_rhs() {
  // rhs = new NumericVectorDistributed(MPI_COMM_SELF, Geometry.levels[0].n_local_dofs, Geometry.levels[0].n_local_dofs);
  // rhs = new dealii::PETScWrappers::MPI::Vector(own_dofs, GlobalMPI.communicators_by_level[level]);
}

void LocalProblem::reinit() {
  reinit_rhs();
  rhs = dealii::PETScWrappers::MPI::Vector(own_dofs, MPI_COMM_SELF);
  solution.reinit(MPI_COMM_SELF, Geometry.levels[0].n_local_dofs, Geometry.levels[0].n_local_dofs, false);
  solution = 0;
  make_constraints();
  constraints.close();
  make_sparsity_pattern();
  std::vector<unsigned int> lines_per_proc;
  lines_per_proc.push_back(sp.n_cols());
  matrix->reinit(MPI_COMM_SELF, sp, lines_per_proc, lines_per_proc, 0);
}

void LocalProblem::solve() {
  GlobalTimerManager.switch_context("solve", level);
  Timer timer1;
  timer1.start ();
  solution = 0;
  solver.solve(*matrix, solution, rhs);
  constraints.distribute(solution);
  timer1.stop();
  solve_counter ++;
}

void LocalProblem::initialize_index_sets() {
  n_procs_in_sweep = 1;
  rank = 0;
}

auto LocalProblem::compare_to_exact_solution() -> void {
  NumericVectorLocal solution_inner(Geometry.levels[level].inner_domain->n_locally_active_dofs);
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->n_locally_active_dofs; i++) {
    solution_inner[i] = solution(i);
  }

  std::ofstream myfile ("output_z.dat");
  for(unsigned int i = 0; i < 100; i++) {
    double z = -GlobalParams.Geometry_Size_X/2.0 + i*GlobalParams.Geometry_Size_X/99.0;
    Position p = {0,0, z};
    NumericVectorLocal local_solution(3);
    NumericVectorLocal exact_solution(3);
    VectorTools::point_value(Geometry.levels[level].inner_domain->dof_handler, solution_inner, p, local_solution);
    GlobalParams.source_field->vector_value(p, exact_solution);
    myfile << "0\t0\t" << z << "\t" << local_solution[0].real()<< "\t" << local_solution[0].imag() ;
    myfile << "\t" << local_solution[1].real()<< "\t"<< local_solution[1].imag();
    myfile << "\t" << local_solution[2].real()<< "\t"<< local_solution[2].imag();
    myfile << "\t" << exact_solution[0].real()<< "\t"<< exact_solution[0].imag() ;
    myfile << "\t" << exact_solution[1].real()<< "\t"<< exact_solution[1].imag();
    myfile << "\t" << exact_solution[2].real()<< "\t"<< exact_solution[2].imag();
    myfile << std::endl;
  }
  myfile.close();
  myfile.open("output_y.dat");
  for(unsigned int i = 0; i < 100; i++) {
    double y = -GlobalParams.Geometry_Size_X/2.0 + i*GlobalParams.Geometry_Size_X/99.0;
    Position p = {0,y,0};
    NumericVectorLocal local_solution(3);
    NumericVectorLocal exact_solution(3);
    VectorTools::point_value(Geometry.levels[level].inner_domain->dof_handler, solution_inner, p, local_solution);
    GlobalParams.source_field->vector_value(p, exact_solution);
    myfile <<"0\t" << y << "\t0\t"<< local_solution[0].real()<< "\t"<< local_solution[0].imag() ;
    myfile << "\t" << local_solution[1].real()<< "\t"<< local_solution[1].imag();
    myfile << "\t" << local_solution[2].real()<< "\t"<< local_solution[2].imag();
    myfile << "\t" << exact_solution[0].real()<< "\t"<< exact_solution[0].imag() ;
    myfile << "\t" << exact_solution[1].real()<< "\t"<< exact_solution[1].imag();
    myfile << "\t" << exact_solution[2].real()<< "\t"<< exact_solution[2].imag();
    myfile << std::endl;
  }
  myfile.close();
  myfile.open("output_x.dat");
  for(unsigned int i = 0; i < 100; i++) {
    double x = -GlobalParams.Geometry_Size_X/2.0 + i*GlobalParams.Geometry_Size_X/99.0;
    Position p = {x,0,0};
    NumericVectorLocal local_solution(3);
    NumericVectorLocal exact_solution(3);
    VectorTools::point_value(Geometry.levels[level].inner_domain->dof_handler, solution_inner, p, local_solution);
    GlobalParams.source_field->vector_value(p, exact_solution);
    myfile << x << "\t0\t0";
    myfile << "\t" << local_solution[0].real()<< "\t"<< local_solution[0].imag() ;
    myfile << "\t" << local_solution[1].real()<< "\t"<< local_solution[1].imag();
    myfile << "\t" << local_solution[2].real()<< "\t"<< local_solution[2].imag();
    myfile << "\t" << exact_solution[0].real()<< "\t"<< exact_solution[0].imag() ;
    myfile << "\t" << exact_solution[1].real()<< "\t"<< exact_solution[1].imag();
    myfile << "\t" << exact_solution[2].real()<< "\t"<< exact_solution[2].imag();
    myfile << std::endl;
  }
  myfile.close();
}

void LocalProblem::compute_solver_factorization() {
  Timer timer1;
  // print_info("LocalProblem::compute_solver_factorization", "Begin solver factorization: ", true, LoggingLevel::PRODUCTION_ONE);
  timer1.start();
  solve();
  timer1.stop();
  solution = 0;
  // print_info("LocalProblem::compute_solver_factorization", "Walltime: " + std::to_string(timer1.wall_time()) , true, LoggingLevel::PRODUCTION_ONE);
}

double LocalProblem::compute_error(dealii::VectorTools::NormType in_norm, Function<3,ComplexNumber> * in_exact, dealii::Vector<ComplexNumber> & in_solution, dealii::DataOut<3> * in_data_out) {
  Timer timer;
  timer.start ();
  double error = 0;
  dealii::Vector<double> cellwise_error(Geometry.levels[level].inner_domain->triangulation.n_active_cells());
  dealii::VectorTools::integrate_difference(Geometry.levels[level].inner_domain->dof_handler, in_solution, *in_exact, cellwise_error, dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2), in_norm);
  unsigned int idx = 0;
  for(auto it = Geometry.levels[level].inner_domain->dof_handler.begin_active(); it != Geometry.levels[level].inner_domain->dof_handler.end(); it++) {
    bool zero_component = false;
    if(GlobalParams.use_tapered_input_signal) {
      const double z = it->center()[2];
      if(z >= GlobalParams.tapering_min_z && z < GlobalParams.tapering_max_z) {
        zero_component = true;
      }
    }
    if(zero_component) {
      cellwise_error[idx] = 0;
    }
    idx++;
  }
  error = dealii::VectorTools::compute_global_error(Geometry.levels[level].inner_domain->triangulation, cellwise_error, in_norm);
  timer.stop ();
  std::string error_name ="";
  if(in_norm == dealii::VectorTools::NormType::H1_norm) {
    error_name = "H1";
  } else {
    error_name = "L2";
  }
  in_data_out->add_data_vector(cellwise_error, "Cellwise_" + error_name + "_error");
  print_info("LocalProblem::compute_error", error_name + " Error: " + std::to_string(error) + " ( computed in " + std::to_string(timer.cpu_time()) + "s)");
  return error;
}

double LocalProblem::compute_L2_error() {
  NumericVectorLocal solution_inner(Geometry.levels[level].inner_domain->n_locally_active_dofs);
  for(unsigned int i = 0; i < Geometry.levels[level].inner_domain->n_locally_active_dofs; i++) {
    solution_inner[i] = solution(i);
  }
  dealii::Vector<double> cellwise_error(Geometry.levels[level].inner_domain->triangulation.n_active_cells());
  dealii::VectorTools::integrate_difference(
    MappingQGeneric<3>(1),
    Geometry.levels[level].inner_domain->dof_handler,
    solution_inner,
    *GlobalParams.source_field,
    cellwise_error,
    dealii::QGauss<3>(GlobalParams.Nedelec_element_order + 2),
    dealii::VectorTools::NormType::L2_norm );
  return dealii::VectorTools::compute_global_error(Geometry.levels[level].inner_domain->triangulation, cellwise_error, dealii::VectorTools::NormType::L2_norm);
}
