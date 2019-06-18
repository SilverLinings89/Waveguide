// Copyright 2018 Pascal Kraft
#ifndef WaveguideCppFlag
#define WaveguideCppFlag

#include "Waveguide.h"
#include <deal.II/base/std_cxx11/bind.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/lac/block_matrix_array.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <sys/time.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "../Helpers/ExactSolution.h"
#include "../Helpers/staticfunctions.h"
#include "../SpaceTransformations/HomogenousTransformationRectangular.h"
#include "../SpaceTransformations/InhomogenousTransformationRectangular.h"
#include "../SpaceTransformations/SpaceTransformation.h"
#include "PreconditionerSweeping.h"
#include "SolutionWeight.h"

int steps = 0;
Parameters GlobalParams;
ModeManager ModeMan;
dealii::ConvergenceTable Convergence_Table;
dealii::TableHandler Optimization_Steps;
double *steps_widths;

Waveguide::Waveguide(MPI_Comm in_mpi_comm, MeshGenerator *in_mg,
    SpaceTransformation *in_st)
: fe(FE_Nedelec<3>(GlobalParams.So_ElementOrder), 2),
  triangulation(Triangulation<3>::MeshSmoothing(Triangulation<3>::none)),
  even(Utilities::MPI::this_mpi_process(in_mpi_comm) % 2 == 0),
  rank(Utilities::MPI::this_mpi_process(in_mpi_comm)),
  real(0),
  imag(3),
  solver_control(GlobalParams.So_TotalSteps, GlobalParams.So_Precision,
      true, true),
      dof_handler(triangulation),
      run_number(0),
      condition_file_counter(0),
      eigenvalue_file_counter(0),
      Layers(GlobalParams.NumberProcesses),
      Dofs_Below_Subdomain(Layers),
      Block_Sizes(Layers),
      is_stored(false),
      Sectors(GlobalParams.M_W_Sectors),
      minimum_local_z(2.0 * GlobalParams.M_R_ZLength),
      maximum_local_z(-2.0 * GlobalParams.M_R_ZLength),
      pout(std::cout, rank == 0),
      timer(in_mpi_comm, pout, TimerOutput::OutputFrequency::summary,
          TimerOutput::wall_times),
          es(GlobalParams.M_C_Shape == ConnectorType::Rectangle) {
  mg = in_mg;
  st = in_st;
  mpi_comm = in_mpi_comm;
  solution = NULL;
  is_stored = false;
  solver_control.log_frequency(10);
  const int number = Layers - 1;
  qualities = new double[number];
  execute_recomputation = false;
  mkdir((solutionpath + "/" + "primal").c_str(), ACCESSPERMS);
  mkdir((solutionpath + "/" + "dual").c_str(), ACCESSPERMS);
  cell_layer_z = 0;
  interface_dof_count = 0;
  n_dofs = 0;
}

Waveguide::~Waveguide() {}

std::complex<double> Waveguide::evaluate_for_Position(double x, double y,
    double z) {
  dealii::Point<3, double> position(x, y, z);
  Vector<double> result(6);
  Vector<double> mode(6);
  if (primal) {
    VectorTools::point_value(dof_handler, primal_with_relevant, position,
        result);
  } else {
    VectorTools::point_value(dof_handler, primal_solution, position, result);
  }
  position[2] = GlobalParams.Minimum_Z;
  this->es.vector_value(position, mode);

  std::complex<double> c1(result(0), result(3));
  std::complex<double> c2(result(1), result(4));
  std::complex<double> c3(result(2), result(5));
  std::complex<double> m1(mode(0), -mode(3));
  std::complex<double> m2(mode(1), -mode(4));
  std::complex<double> m3(mode(2), -mode(5));

  return m1 * c1 + m2 * c2 + m3 * c3;
}

std::complex<double> Waveguide::evaluate_Energy_for_Position(double x, double y,
    double z) {
  dealii::Point<3, double> position(x, y, z);
  Vector<double> result(6);
  if (primal) {
    VectorTools::point_value(dof_handler, primal_with_relevant, position,
        result);
  } else {
    VectorTools::point_value(dof_handler, primal_solution, position, result);
  }
  double val = result(0) * result(0) + result(1) * result(1) +
      result(2) * result(2) + result(3) * result(3) +
      result(4) * result(4) + result(5) * result(5);
  val = std::sqrt(val);
  std::complex<double> ret;
  ret.real(val);
  ret.imag(0);
  double eps = 1.0;
  if (this->mg->math_coordinate_in_waveguide(position)) {
    eps = GlobalParams.M_W_epsilonin;
  } else {
    eps = GlobalParams.M_W_epsilonout;
  }
  return eps * ret;
}

void Waveguide::estimate_solution() {
  MPI_Barrier(mpi_comm);
  deallog.push("estimate_solution");
  DoFHandler<3>::active_cell_iterator cell, endc;
  unsigned int min_dof = locally_owned_dofs.nth_index_in_set(0);
  unsigned int max_dof =
      locally_owned_dofs.nth_index_in_set(locally_owned_dofs.n_elements() - 1);
  cell = dof_handler.begin_active(), endc = dof_handler.end();
  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {
      for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
        std::vector<types::global_dof_index> local_dof_indices(
            fe.dofs_per_line);
        for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
          ((cell->face(i))->line(j))->get_dof_indices(local_dof_indices);
          //          for (unsigned int i = 0; i < fe.dofs_per_line; i++) {
          //            local_dof_indices[i] =
          //            local_to_global_index(local_dof_indices[i]);
          //          }
          Tensor<1, 3, double> ptemp =
              ((cell->face(i))->line(j))->center(true, false);
          if (std::abs(ptemp[2] - GlobalParams.Minimum_Z) > 0.0001) {
            Point<3, double> p(ptemp[0], ptemp[1], ptemp[2]);
            Tensor<1, 3, double> dtemp = ((cell->face(i))->line(j))->vertex(0) -
                ((cell->face(i))->line(j))->vertex(1);
            dtemp = dtemp / dtemp.norm();
            Point<3, double> direction(dtemp[0], dtemp[1], dtemp[2]);
            Vector<double> val(6);
            es.vector_value(p, val);
            double a = direction(0) * val(0) + direction(1) * val(1) +
                direction(2) * val(2);
            double b = direction(0) * val(3) + direction(1) * val(4) +
                direction(2) * val(5);
            if (local_dof_indices[0] >= min_dof &&
                local_dof_indices[0] < max_dof) {
              EstimatedSolution[local_dof_indices[0]] = a;
            }
            if (local_dof_indices[1] >= min_dof &&
                local_dof_indices[1] < max_dof) {
              EstimatedSolution[local_dof_indices[1]] = b;
            }
          }
        }
      }
    }
  }
  MPI_Barrier(mpi_comm);
  EstimatedSolution.compress(VectorOperation::insert);
  deallog.pop();
}

bool compareConstraintPairs(ConstraintPair v1, ConstraintPair v2) {
  return (v1.left < v2.left);
}

Tensor<2, 3, std::complex<double>> Waveguide::Conjugate_Tensor(
    Tensor<2, 3, std::complex<double>> input) {
  Tensor<2, 3, std::complex<double>> ret;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      ret[i][j].real(input[i][j].real());
      ret[i][j].imag(-input[i][j].imag());
    }
  }
  return ret;
}

Tensor<1, 3, std::complex<double>> Waveguide::Conjugate_Vector(
    Tensor<1, 3, std::complex<double>> input) {
  Tensor<1, 3, std::complex<double>> ret;

  for (int i = 0; i < 3; i++) {
    ret[i].real(input[i].real());
    ret[i].imag(-input[i].imag());
  }
  return ret;
}

unsigned int Waveguide::local_to_global_index(unsigned int local_index) {
  if (GlobalParams.MPI_Rank == 0) {
    return local_index;
  } else {
    if (local_index < interface_dof_count) {
      return periodicity_constraints[local_index].right +
          (GlobalParams.MPI_Rank - 1) * (n_dofs - interface_dof_count);
    } else {
      return local_index +
          GlobalParams.MPI_Rank * (n_dofs - interface_dof_count);
    }
  }
}

unsigned int Waveguide::global_to_local_index(unsigned int local_index) {
  if (GlobalParams.MPI_Rank == 0) {
    return local_index;
  } else {
    int temp =
        local_index - GlobalParams.MPI_Rank * (n_dofs - interface_dof_count);
    if (temp < (int)interface_dof_count) {
      temp += n_dofs - interface_dof_count;
      for (unsigned int i = 0; i < periodicity_constraints.size(); i++) {
        if ((int)periodicity_constraints[i].right == temp)
          return periodicity_constraints[i].left;
      }
      std::cout
      << "Error in computation of local index. Out of bounds? Index was "
      << local_index << std::endl;
      return n_dofs - 1;
    } else {
      return temp;
    }
  }
}

void Waveguide::make_grid() {
  mg->prepare_triangulation(&triangulation);
  dof_handler.distribute_dofs(fe);
  SortDofsDownstream();
  n_dofs = dof_handler.n_dofs();
  DoFTools::make_periodicity_constraints(dof_handler, 4, 5, 2,
      periodic_constraints);
  interface_dof_count = periodic_constraints.n_constraints();
  n_global_dofs = GlobalParams.NumberProcesses * n_dofs -
      (GlobalParams.NumberProcesses - 1) * interface_dof_count;

  DoFHandler<3>::active_cell_iterator cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  minimum_local_z = 2.0 * GlobalParams.M_R_ZLength;
  maximum_local_z = -2.0 * GlobalParams.M_R_ZLength;
  for (; cell != endc; ++cell) {
    if (cell->is_locally_owned()) {
      for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
        double temp = (cell->face(i)->center())[2];
        if (temp < minimum_local_z) minimum_local_z = temp;
        if (temp > maximum_local_z) maximum_local_z = temp;
      }
    }
  }
}

bool compareIndexCenterPairs(std::pair<int, double> c1,
    std::pair<int, double> c2) {
  return c1.second < c2.second;
}

void Waveguide::SortDofsDownstream() {
  std::vector<std::pair<int, double>> current;
  DoFHandler<3>::active_cell_iterator cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  std::vector<unsigned int> lines_touched;
  std::vector<types::global_dof_index> local_line_dofs(fe.dofs_per_line);
  for (; cell != endc; ++cell) {
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
      for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
        if (!(std::find(lines_touched.begin(), lines_touched.end(),
            cell->face(i)->line(j)->index()) !=
                lines_touched.end())) {
          ((cell->face(i))->line(j))->get_dof_indices(local_line_dofs);
          for (unsigned k = 0; k < local_line_dofs.size(); k++) {
            current.push_back(std::pair<int, double>(
                local_line_dofs[k], (cell->face(i))->line(j)->center()[2]));
          }
          lines_touched.push_back(cell->face(i)->line(j)->index());
        }
      }
    }
  }
  std::sort(current.begin(), current.end(), compareIndexCenterPairs);
  std::vector<unsigned int> new_numbering;
  new_numbering.resize(current.size());
  for (unsigned int i = 0; i < current.size(); i++) {
    new_numbering[current[i].first] = i;
  }
  dof_handler.renumber_dofs(new_numbering);
}

void Waveguide::Shift_Constraint_Matrix(ConstraintMatrix *in_cm) {
  ConstraintMatrix new_global;
  new_global.reinit(locally_relevant_dofs);
  for (unsigned int i = 0; i < n_dofs; i++) {
    if (in_cm->is_constrained(i)) {
      const std::vector<std::pair<unsigned int, double>> *curr_line =
          in_cm->get_constraint_entries(i);
      for (unsigned int j = 0; j < curr_line->size(); j++) {
        new_global.add_entry(
            local_to_global_index(i),
            local_to_global_index(curr_line->operator[](j).first),
            curr_line->operator[](j).second);
      }
    }
  }
  in_cm->clear();
  in_cm->copy_from(new_global);
}

void Waveguide::Compute_Dof_Numbers() {
  deallog << "Total Dof Count per block: " << n_dofs << " interface dof count "
      << interface_dof_count << std::endl;
  std::vector<types::global_dof_index> dof_indices(fe.dofs_per_face);
  std::vector<types::global_dof_index> DofsPerSubdomain(Layers);
  std::vector<int> InternalBoundaryDofs(Layers);
  DofsPerSubdomain[0] = n_dofs;
  for (unsigned int i = 1; i < Layers; i++) {
    DofsPerSubdomain[i] = n_dofs - interface_dof_count;
  }
  for (unsigned int i = 0; i < Layers; i++) {
    Block_Sizes[i] = DofsPerSubdomain[i];
  }
  deallog << "Layers: " << Layers << std::endl;
  for (unsigned int i = 0; i < Layers; i++) {
    deallog << Block_Sizes[i] << std::endl;
  }

  Dofs_Below_Subdomain[0] = 0;

  for (unsigned int i = 1; i < Layers; i++) {
    Dofs_Below_Subdomain[i] = Dofs_Below_Subdomain[i - 1] + Block_Sizes[i - 1];
  }

  for (unsigned int i = 0; i < Layers; i++) {
    IndexSet temp(n_global_dofs);
    temp.clear();
    deallog << "Adding Block " << i + 1 << " from " << Dofs_Below_Subdomain[i]
                                                                            << " to " << Dofs_Below_Subdomain[i] + Block_Sizes[i] - 1
                                                                            << std::endl;
    temp.add_range(Dofs_Below_Subdomain[i],
        Dofs_Below_Subdomain[i] + Block_Sizes[i]);
    set.push_back(temp);
  }
}

IndexSet Waveguide::combine_indexes(IndexSet lower, IndexSet upper) const {
  IndexSet ret(lower.size() + upper.size());
  ret.add_indices(lower);
  ret.add_indices(upper, lower.size());
  return ret;
}

void Waveguide::switch_to_primal(SpaceTransformation *primal_st) {
  st = primal_st;
  solution = &primal_solution;
  primal = true;
  path_prefix = "primal";
}

void Waveguide::switch_to_dual(SpaceTransformation *dual_st) {
  st = dual_st;
  solution = &dual_solution;
  primal = false;
  path_prefix = "dual";
}

void Waveguide::setup_system() {
  deallog.push("setup_system");
  for (unsigned int i = 0; i < n_dofs; i++) {
    if (periodic_constraints.is_constrained(i)) {
      unsigned int l = 0;
      unsigned int r = 0;
      std::pair<unsigned int, double> entry =
          periodic_constraints.get_constraint_entries(i)->operator[](0);
      if (entry.first < i) {
        l = entry.first;
        r = i;
      } else {
        r = entry.first;
        l = i;
      }
      ConstraintPair temp;
      temp.left = l;
      temp.right = r;
      temp.sign = entry.second > 0;
      periodicity_constraints.push_back(temp);
    }
  }
  std::sort(periodicity_constraints.begin(), periodicity_constraints.end(),
      compareConstraintPairs);

  unsigned int global_dof_count =
      GlobalParams.NumberProcesses * n_dofs -
      (GlobalParams.NumberProcesses - 1) * interface_dof_count;
  locally_owned_dofs.set_size(global_dof_count);
  if (GlobalParams.MPI_Rank == 0) {
    locally_owned_dofs.add_range(0, n_dofs);
  } else {
    locally_owned_dofs.add_range(
        GlobalParams.MPI_Rank * n_dofs -
        (GlobalParams.MPI_Rank - 1) * interface_dof_count,
        (GlobalParams.MPI_Rank + 1) * n_dofs -
        (GlobalParams.MPI_Rank) * interface_dof_count);
  }
  locally_relevant_dofs.set_size(global_dof_count);
  if (GlobalParams.MPI_Rank == 0) {
    locally_relevant_dofs.add_range(0, n_dofs + 2 * interface_dof_count);
  } else {
    if (GlobalParams.MPI_Rank + 1 < GlobalParams.NumberProcesses) {
      locally_relevant_dofs.add_range(
          GlobalParams.MPI_Rank * n_dofs -
          (GlobalParams.MPI_Rank - 1) * interface_dof_count -
          (2 * interface_dof_count),
          (GlobalParams.MPI_Rank + 1) * n_dofs -
          (GlobalParams.MPI_Rank) * interface_dof_count +
          2 * interface_dof_count);
    } else {
      locally_relevant_dofs.add_range(
          GlobalParams.MPI_Rank * n_dofs -
          (GlobalParams.MPI_Rank - 1) * interface_dof_count -
          (2 * interface_dof_count),
          global_dof_count);
    }
  }

  // Here we start computing the distribution of entries(indices thereof) to the
  // specific blocks of the 3 matrices(system matrix and the 2 preconditioner
  // matrices.)
  int prec_even_block_count = Utilities::MPI::n_mpi_processes(mpi_comm) / 2;
  if (Utilities::MPI::n_mpi_processes(mpi_comm) % 2 == 1) {
    prec_even_block_count++;
  }

  i_sys_owned.resize(Layers);

  i_sys_readable.resize(Layers);

  for (unsigned int i = 0; i < Layers; i++) {
    int size = Block_Sizes[i];
    bool local = (i == rank);
    bool readable = (i == rank) || (i == rank - 1);
    IndexSet temp(size);
    if (local) {
      temp.add_range(0, size);
    }
    IndexSet temp2(size);
    if (readable) {
      if (i == rank) {
        temp2.add_range(0, size);
      } else {
        temp2.add_range(size - interface_dof_count, size);
      }
    }

    i_sys_owned[i] = temp;
    i_sys_readable[i] = temp2;
  }

  i_prec_even_owned_row.resize(Layers);
  i_prec_even_owned_col.resize(Layers);
  i_prec_even_writable.resize(Layers);
  i_prec_odd_owned_row.resize(Layers);
  i_prec_odd_owned_col.resize(Layers);
  i_prec_odd_writable.resize(Layers);

  for (unsigned int i = 0; i < Layers; i++) {
    int size = Block_Sizes[i];
    bool even_row_owned = false;
    bool even_row_writable = false;
    bool even_col_owned = false;
    if (even) {
      if (i == rank || i == rank + 1) {
        even_row_owned = true;
        even_row_writable = true;
        even_col_owned = true;
      } else {
        if (i == rank - 1) {
          even_row_writable = true;
        }
      }
    } else {
      if (i == rank || i == rank - 1) {
        even_row_writable = true;
      }
    }
    IndexSet ero(size);
    IndexSet erw(size);
    IndexSet eco(size);
    if (even_row_owned) {
      ero.add_range(0, size);
    }
    if (even_row_writable) {
      erw.add_range(0, size);
    }
    if (even_col_owned) {
      eco.add_range(0, size);
    }

    i_prec_even_owned_row[i] = ero;
    i_prec_even_owned_col[i] = eco;
    i_prec_even_writable[i] = erw;
  }

  for (unsigned int i = 0; i < Layers; i++) {
    int size = Block_Sizes[i];
    bool odd_row_owned = false;
    bool odd_row_writable = false;
    bool odd_col_owned = false;
    if (!even) {
      if (i == rank || i == rank + 1) {
        odd_row_owned = true;
        odd_row_writable = true;
        odd_col_owned = true;
      }
      if (i == rank - 1) {
        odd_row_writable = true;
      }
    } else {
      if (i == rank || i == rank - 1) {
        odd_row_writable = true;
      }
    }
    IndexSet oro(size);
    IndexSet orw(size);
    IndexSet oco(size);
    if (odd_row_owned) {
      oro.add_range(0, size);
    }
    if (odd_row_writable) {
      orw.add_range(0, size);
    }
    if (odd_col_owned) {
      oco.add_range(0, size);
    }

    if (rank == 0 && i == 0) {
      oro.add_range(0, size);
      orw.add_range(0, size);
      oco.add_range(0, size);
    }

    if (rank == Layers - 1 && i == Layers - 1) {
      oro.add_range(0, size);
      orw.add_range(0, size);
      oco.add_range(0, size);
    }

    i_prec_odd_owned_row[i] = oro;
    i_prec_odd_owned_col[i] = oco;
    i_prec_odd_writable[i] = orw;
  }

  int even_blocks = GlobalParams.NumberProcesses / 2;
  int odd_blocks = GlobalParams.NumberProcesses / 2;

  if (GlobalParams.NumberProcesses % 2 == 1) {
    even_blocks++;
  } else {
    odd_blocks++;
  }

  std::vector<IndexSet> temp0 = i_prec_odd_owned_row;
  std::vector<IndexSet> temp1 = i_prec_odd_owned_col;
  std::vector<IndexSet> temp2 = i_prec_odd_writable;
  i_prec_odd_owned_row.clear();
  i_prec_odd_owned_col.clear();
  i_prec_odd_writable.clear();
  i_prec_odd_owned_row.push_back(temp0[0]);
  i_prec_odd_owned_col.push_back(temp1[0]);
  i_prec_odd_writable.push_back(temp2[0]);

  for (int i = 2; i < static_cast<int>(Layers); i += 2) {
    i_prec_odd_owned_row.push_back(combine_indexes(temp0[i - 1], temp0[i]));
    i_prec_odd_owned_col.push_back(combine_indexes(temp1[i - 1], temp1[i]));
    i_prec_odd_writable.push_back(combine_indexes(temp2[i - 1], temp2[i]));
  }

  if (GlobalParams.NumberProcesses % 2 == 0) {
    i_prec_odd_owned_row.push_back(temp0[GlobalParams.NumberProcesses - 1]);
    i_prec_odd_owned_col.push_back(temp1[GlobalParams.NumberProcesses - 1]);
    i_prec_odd_writable.push_back(temp2[GlobalParams.NumberProcesses - 1]);
  }

  temp0 = i_prec_even_owned_row;
  temp1 = i_prec_even_owned_col;
  temp2 = i_prec_even_writable;

  i_prec_even_owned_row.clear();
  i_prec_even_owned_col.clear();
  i_prec_even_writable.clear();

  for (int i = 1; i < static_cast<int>(Layers); i += 2) {
    i_prec_even_owned_row.push_back(combine_indexes(temp0[i - 1], temp0[i]));
    i_prec_even_owned_col.push_back(combine_indexes(temp1[i - 1], temp1[i]));
    i_prec_even_writable.push_back(combine_indexes(temp2[i - 1], temp2[i]));
  }

  if (GlobalParams.NumberProcesses % 2 == 1) {
    i_prec_even_owned_row.push_back(temp0[GlobalParams.NumberProcesses - 1]);
    i_prec_even_owned_col.push_back(temp1[GlobalParams.NumberProcesses - 1]);
    i_prec_even_writable.push_back(temp2[GlobalParams.NumberProcesses - 1]);
  }
 
  locally_owned_dofs_all_processors.resize(Layers);

  for (unsigned int i = 0; i < Layers; i++) {
    locally_owned_dofs_all_processors[i].clear();
    locally_owned_dofs_all_processors[i].set_size(n_global_dofs);
  }

  for (unsigned int i = 0; i < GlobalParams.NumberProcesses; i++) {
    if (i == 0) {
      locally_owned_dofs_all_processors[i].add_range(0, n_dofs);
    } else {
      locally_owned_dofs_all_processors[i].add_range(
          GlobalParams.MPI_Rank * n_dofs -
          (GlobalParams.MPI_Rank - 1) * interface_dof_count,
          (GlobalParams.MPI_Rank + 1) * n_dofs -
          (GlobalParams.MPI_Rank) * interface_dof_count);
    }
  }

  UpperDofs = locally_owned_dofs;

  LowerDofs = locally_owned_dofs;

  if (rank != 0) {
    LowerDofs.add_indices(locally_owned_dofs_all_processors[rank - 1], 0);
  }

  if (rank != Layers - 1) {
    UpperDofs.add_indices(locally_owned_dofs_all_processors[rank + 1], 0);
  }

  std::vector<unsigned int> final_numbering;
  for (unsigned int i = 0; i < n_dofs; i++) {
    final_numbering.push_back(local_to_global_index(i));
  }

  dof_handler.renumber_dofs(final_numbering);

  Prepare_Boundary_Constraints();

  deallog << "Boundaryconditions prepared." << std::endl;
  reinit_all();

  deallog.pop();
}

void Waveguide::Prepare_Boundary_Constraints() {
  cm.clear();
  cm.reinit(locally_relevant_dofs);

  cm_prec_even.clear();
  cm_prec_odd.clear();
  cm_prec_even.reinit(locally_relevant_dofs);
  cm_prec_odd.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, cm);
  DoFTools::make_hanging_node_constraints(dof_handler, cm_prec_even);
  DoFTools::make_hanging_node_constraints(dof_handler, cm_prec_odd);

  ProjectBoundaryConditions();
  MakeBoundaryConditions();
  MakePreconditionerBoundaryConditions();

  cm.close();
  cm_prec_even.close();
  cm_prec_odd.close();
}

void Waveguide::reinit_all() {
  deallog.push("reinit_all");

  deallog << "reinitializing right-hand side" << std::endl;
  reinit_rhs();

  deallog << "reinitializing solutiuon" << std::endl;
  reinit_solution();

  deallog << "reinitializing preconditioner" << std::endl;
  reinit_preconditioner();

  deallog << "reinitializing system matrix" << std::endl;
  reinit_systemmatrix();

  deallog << "Done" << std::endl;
  deallog.pop();
}

void Waveguide::reinit_for_rerun() {
  reinit_rhs();
  reinit_preconditioner_fast();
  reinit_systemmatrix();
}

void Waveguide::reinit_systemmatrix() {
  deallog.push("reinit_systemmatrix");

  ConstraintMatrix cm_temp;
  cm_temp.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, cm_temp);

  cm_temp.close();

  deallog << "Generating BSP" << std::endl;

  TrilinosWrappers::BlockSparsityPattern sp(i_sys_owned, MPI_COMM_WORLD);

  deallog << "Collecting sizes ..." << std::endl;

  sp.collect_sizes();

  deallog << "Making BSP ..." << std::endl;
  DoFTools::make_sparsity_pattern(dof_handler, sp, cm_temp, true);
  //  for (TrilinosWrappers::SparsityPatternIterators::Iterator it =
  //           sp_temp.begin();
  //       it != sp_temp.end(); it++) {
  //    unsigned int row = it->row();
  //    unsigned int col = it->column();
  //    deallog << "Row: " << row << std::endl;
  //    deallog << "Col: " << col << std::endl;
  //    sp.add(it->row(), it->column());
  //  }
  sp.compress();
  deallog << "Non-zero entries in sp: " << sp.n_nonzero_elements() << std::endl;
  deallog << "Initializing system_matrix ..." << std::endl;
  sp.compress();
  system_matrix.reinit(sp);
  deallog.pop();
}

void Waveguide::reinit_rhs() {
  system_rhs.reinit(i_sys_owned, MPI_COMM_WORLD);

  preconditioner_rhs.reinit(n_global_dofs);
}

void Waveguide::reinit_solution() {
  std::vector<IndexSet> ghost, partitioning;

  for (unsigned int i = 0; i < i_sys_owned.size(); i++) {
    IndexSet all = IndexSet(i_sys_owned[i].size());
    all.add_range(0, i_sys_owned[i].size());
    IndexSet none = IndexSet(i_sys_owned[i].size());
    if (i == GlobalParams.MPI_Rank) {
      partitioning.push_back(all);
    } else {
      partitioning.push_back(none);
    }
    if (i == GlobalParams.MPI_Rank - 1) {
      ghost.push_back(all);
    } else {
      ghost.push_back(none);
    }
  }
  solution->reinit(partitioning, ghost, mpi_comm, true);
  EstimatedSolution.reinit(i_sys_owned, mpi_comm);
  ErrorOfSolution.reinit(i_sys_owned, mpi_comm);
}

void Waveguide::reinit_storage() { storage.reinit(i_sys_owned, mpi_comm); }

void Waveguide::reinit_preconditioner() {
  deallog.push("reinit_preconditioner");

  deallog.push("Generating BSP");

  ConstraintMatrix cm_temp;
  cm_temp.reinit(locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, cm_temp);

  cm_temp.close();

  deallog << "Started" << std::endl;

  TrilinosWrappers::BlockSparsityPattern epsp(i_prec_even_owned_row,
      i_prec_even_owned_col,
      i_prec_even_writable, mpi_comm);

  deallog << "Even worked. Continuing Odd." << std::endl;

  TrilinosWrappers::BlockSparsityPattern opsp(i_prec_odd_owned_row,
      i_prec_odd_owned_col,
      i_prec_odd_writable, mpi_comm);

  deallog << "Odd worked. Done" << std::endl;

  deallog.pop();

  deallog << "Collecting sizes ..." << std::endl;
  epsp.collect_sizes();
  opsp.collect_sizes();

  deallog << "Even Preconditioner Matrices ..." << std::endl;
  DoFTools::make_sparsity_pattern(dof_handler, epsp, cm_prec_even, true);
  epsp.compress();
  deallog << "Odd Preconditioner Matrices ..." << std::endl;
  DoFTools::make_sparsity_pattern(dof_handler, opsp, cm_prec_odd, true);
  opsp.compress();

  deallog << "Done" << std::endl;

  deallog.pop();

  deallog.push("Initializing matrices");
  deallog << "Even ..." << std::endl;
  prec_matrix_even.reinit(epsp);
  deallog << "Odd ..." << std::endl;
  prec_matrix_odd.reinit(opsp);
  deallog << "Done." << std::endl;
  deallog.pop();
  deallog.pop();
}

void Waveguide::reinit_preconditioner_fast() {}

void Waveguide::assemble_system() {
  reinit_rhs();

  QGauss<3> quadrature_formula(2);
  const FEValuesExtractors::Vector real(0);
  const FEValuesExtractors::Vector imag(3);
  FEValues<3> fe_values(fe, quadrature_formula,
      update_values | update_gradients | update_JxW_values |
      update_quadrature_points);
  std::vector<Point<3>> quadrature_points;
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  deallog << "Starting Assemblation process" << std::endl;

  FullMatrix<double> cell_matrix_real(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_prec_odd(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_prec_even(dofs_per_cell, dofs_per_cell);

  double e_temp = 1.0;
  if (!GlobalParams.C_AllOne) {
    e_temp *= GlobalParams.C_Epsilon;
  }
  double mu_temp = 1.0;
  if (!GlobalParams.C_AllOne) {
    mu_temp *= GlobalParams.C_Mu;
  }

  const double eps_in = GlobalParams.M_W_epsilonin * e_temp;
  const double eps_out = GlobalParams.M_W_epsilonout * e_temp;
  const double mu_zero = mu_temp;

  Vector<double> cell_rhs(dofs_per_cell);
  cell_rhs = 0;
  Tensor<2, 3, std::complex<double>> transformation, epsilon, epsilon_pre1,
  epsilon_pre2, mu, mu_prec1, mu_prec2;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  // std::vector<types::global_dof_index>
  // local_dof_indices_global(dofs_per_cell);
  DoFHandler<3>::active_cell_iterator cell, endc;
  cell = dof_handler.begin_active(), endc = dof_handler.end();
  std::complex<double> k_a_sqr(GlobalParams.C_omega,
      GlobalParams.So_PreconditionerDampening);
  k_a_sqr = k_a_sqr * k_a_sqr;
  for (; cell != endc; ++cell) {
    cell->get_dof_indices(local_dof_indices);
    cell_rhs.reinit(dofs_per_cell, false);
    fe_values.reinit(cell);
    quadrature_points = fe_values.get_quadrature_points();
    bool has_left = false;
    bool has_right = false;
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
      if (cell->face(i)->center(true, false)[2] - cell_layer_z < 0.0001)
        has_left = true;
      if (cell->face(i)->center(true, false)[2] + 0.0001 > cell_layer_z)
        has_right = true;
    }
    bool compute_rhs = has_left && has_right;
    std::vector<types::global_dof_index> input_dofs(fe.dofs_per_line);
    IndexSet input_dofs_local_set(fe.dofs_per_cell);
    std::vector<Point<3, double>> input_dof_centers(fe.dofs_per_cell);
    std::vector<Tensor<1, 3, double>> input_dof_dirs(fe.dofs_per_cell);

    if (compute_rhs) {
      for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
        if (cell->face(i)->center(true, false)[2] - cell_layer_z < 0.0001) {
          for (unsigned int e = 0; e < GeometryInfo<3>::lines_per_face; e++) {
            (cell->face(i)->line(e))->get_dof_indices(input_dofs);
            for (unsigned int j = 0; j < fe.dofs_per_cell; j++) {
              for (unsigned int k = 0; k < fe.dofs_per_line; k++) {
                if (local_dof_indices[j] == input_dofs[k]) {
                  input_dofs_local_set.add_index(j);
                  input_dof_centers[j] = cell->face(i)->line(e)->center();
                  input_dof_dirs[j] = cell->face(i)->line(e)->vertex(1) -
                      cell->face(i)->line(e)->vertex(0);
                }
              }
            }
          }
        }
      }
    }
    cell_matrix_real = 0;
    cell_matrix_prec_odd = 0;
    cell_matrix_prec_even = 0;
    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
      if (!locals_set) {
        if (quadrature_points[q_index][2] < minimum_local_z) {
          minimum_local_z = quadrature_points[q_index][2];
        }
        if (quadrature_points[q_index][2] > maximum_local_z) {
          maximum_local_z = quadrature_points[q_index][2];
        }
      }
      transformation = st->get_Tensor(quadrature_points[q_index]);

      if (mg->math_coordinate_in_waveguide(quadrature_points[q_index])) {
        epsilon = transformation * eps_in;
      } else {
        epsilon = transformation * eps_out;
      }

      mu = invert(transformation) / mu_zero;

      if (even) {
        epsilon_pre1 = st->get_Tensor(quadrature_points[q_index]);
        mu_prec1 = st->get_Tensor(quadrature_points[q_index]);
        epsilon_pre2 =
            st->get_Preconditioner_Tensor(quadrature_points[q_index], rank);
        mu_prec2 =
            st->get_Preconditioner_Tensor(quadrature_points[q_index], rank);
      } else {
        epsilon_pre2 = st->get_Tensor(quadrature_points[q_index]);
        mu_prec2 = st->get_Tensor(quadrature_points[q_index]);
        epsilon_pre1 =
            st->get_Preconditioner_Tensor(quadrature_points[q_index], rank);
        mu_prec1 =
            st->get_Preconditioner_Tensor(quadrature_points[q_index], rank);
      }

      mu_prec1 = invert(mu_prec1) / mu_zero;
      mu_prec2 = invert(mu_prec2) / mu_zero;

      if (mg->math_coordinate_in_waveguide(quadrature_points[q_index])) {
        epsilon_pre1 *= eps_in;
        epsilon_pre2 *= eps_in;
      } else {
        epsilon_pre1 *= eps_out;
        epsilon_pre2 *= eps_out;
      }

      const double JxW = fe_values.JxW(q_index);
      for (unsigned int i = 0; i < dofs_per_cell; i++) {
        Tensor<1, 3, std::complex<double>> I_Curl;
        Tensor<1, 3, std::complex<double>> I_Val;
        for (int k = 0; k < 3; k++) {
          I_Curl[k].imag(fe_values[imag].curl(i, q_index)[k]);
          I_Curl[k].real(fe_values[real].curl(i, q_index)[k]);
          I_Val[k].imag(fe_values[imag].value(i, q_index)[k]);
          I_Val[k].real(fe_values[real].value(i, q_index)[k]);
        }

        for (unsigned int j = 0; j < dofs_per_cell; j++) {
          Tensor<1, 3, std::complex<double>> J_Curl;
          Tensor<1, 3, std::complex<double>> J_Val;
          for (int k = 0; k < 3; k++) {
            J_Curl[k].imag(fe_values[imag].curl(j, q_index)[k]);
            J_Curl[k].real(fe_values[real].curl(j, q_index)[k]);
            J_Val[k].imag(fe_values[imag].value(j, q_index)[k]);
            J_Val[k].real(fe_values[real].value(j, q_index)[k]);
          }

          std::complex<double> x =
              (mu * I_Curl) * Conjugate_Vector(J_Curl) * JxW -
              ((epsilon * I_Val) * Conjugate_Vector(J_Val)) * JxW *
              GlobalParams.C_omega * GlobalParams.C_omega;
          cell_matrix_real[i][j] += x.real();

          std::complex<double> pre1 =
              (mu_prec1 * I_Curl) * Conjugate_Vector(J_Curl) * JxW -
              ((epsilon_pre1 * I_Val) * Conjugate_Vector(J_Val)) * JxW *
              k_a_sqr;
          cell_matrix_prec_even[i][j] += pre1.real();

          std::complex<double> pre2 =
              (mu_prec2 * I_Curl) * Conjugate_Vector(J_Curl) * JxW -
              ((epsilon_pre2 * I_Val) * Conjugate_Vector(J_Val)) * JxW *
              k_a_sqr;
          cell_matrix_prec_odd[i][j] += pre2.real();
        }
        if (compute_rhs && quadrature_points[q_index][2] > cell_layer_z) {
          std::complex<double> rhs2 =
              (mu * I_Curl) *
              Conjugate_Vector(es.curl(quadrature_points[q_index])) * JxW -
              ((epsilon * I_Val)) *
              Conjugate_Vector(es.val(quadrature_points[q_index])) * JxW *
              GlobalParams.C_omega * GlobalParams.C_omega;
          cell_rhs[i] -= rhs2.real();
        }
      }

      cm.distribute_local_to_global(cell_matrix_real, cell_rhs,
          local_dof_indices, system_matrix,
          system_rhs, false);

      cm_prec_odd.distribute_local_to_global(cell_matrix_prec_odd, cell_rhs,
          local_dof_indices, prec_matrix_odd,
          preconditioner_rhs, false);
      cm_prec_even.distribute_local_to_global(
          cell_matrix_prec_even, cell_rhs, local_dof_indices, prec_matrix_even,
          preconditioner_rhs, false);
    }
  }
  locals_set = true;

  MPI_Barrier(mpi_comm);

  deallog << "Assembling done. L2-Norm of RHS: " << system_rhs.l2_norm() << std::endl;

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  prec_matrix_even.compress(VectorOperation::add);
  prec_matrix_odd.compress(VectorOperation::add);

  if (primal) {
    cm.distribute(primal_solution);
  } else {
    cm.distribute(dual_solution);
  }
  cm.distribute(EstimatedSolution);
  cm.distribute(ErrorOfSolution);
  MPI_Barrier(mpi_comm);
  deallog << "Distributing solution done." << std::endl;
  estimate_solution();
}

void Waveguide::MakeBoundaryConditions() {
  DoFHandler<3>::active_cell_iterator cell, endc;
  cell = dof_handler.begin_active(), endc = dof_handler.end();
  std::vector<types::global_dof_index> local_line_dofs(fe.dofs_per_line);
  InputInterfaceDofs = IndexSet(n_global_dofs);
  cell_layer_z = 0.0;

  for (; cell != endc; ++cell) {
    double cell_min_z = GlobalParams.M_R_ZLength;
    double cell_max_z = -GlobalParams.M_R_ZLength;
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
      Point<3, double> pos = cell->face(i)->center(true, true);
      if (pos[2] < cell_min_z) cell_min_z = pos[2];
      if (pos[2] > cell_max_z) cell_max_z = pos[2];
    }
    if (cell_min_z < -GlobalParams.M_R_ZLength / 2.0 &&
        cell_max_z >= -GlobalParams.M_R_ZLength / 2.0) {
      if (cell_min_z < cell_layer_z) cell_layer_z = cell_min_z;
    }
  }
  cell_layer_z = Utilities::MPI::min(cell_layer_z, MPI_COMM_WORLD);
  deallog << "The input cell interface layer is located at " << cell_layer_z
      << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  for (; cell != endc; ++cell) {
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
      Point<3, double> pos = cell->face(i)->center(true, true);
      if (abs(pos[2] - cell_layer_z) < 0.0001) {
        for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
          ((cell->face(i))->line(j))->get_dof_indices(local_line_dofs);
          for (unsigned int k = 0; k < fe.dofs_per_line; k++) {
            InputInterfaceDofs.add_index(local_line_dofs[k]);
          }
        }
      }
    }
  }

  dealii::ZeroFunction<3, double> zf(6);

  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 0,
      cm);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 1,
      cm);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 2,
      cm);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 3,
      cm);

  if (GlobalParams.MPI_Rank == 0) {
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 4,
        cm);
  }

  if ( GlobalParams.MPI_Rank + 1 == GlobalParams.NumberProcesses) {
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 5,
        cm);
  }

  cell = dof_handler.begin_active();
  const unsigned int face_own_count = std::max(
      static_cast<unsigned int>(0),
      fe.dofs_per_face - GeometryInfo<3>::lines_per_face * fe.dofs_per_line);
  const unsigned int cell_own_count = std::max(
      static_cast<unsigned int>(0),
      fe.dofs_per_cell - GeometryInfo<3>::faces_per_cell * fe.dofs_per_face +
      GeometryInfo<3>::lines_per_cell * fe.dofs_per_line);
  std::vector<types::global_dof_index> local_face_dofs(fe.dofs_per_face);
  deallog << "Dofs per line: " << fe.dofs_per_line << std::endl;
  deallog << "Dofs per face: " << fe.dofs_per_face << std::endl;
  deallog << "Dofs per cell: " << fe.dofs_per_cell << std::endl;
  if (run_number == 0) {
    fixed_dofs.set_size(n_global_dofs);
  }

  for (; cell != endc; ++cell) {
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
      Point<3, double> center = (cell->face(i))->center(true, false);
      if (center[0] < 0) center[0] *= (-1.0);
      if (center[1] < 0) center[1] *= (-1.0);

      if (std::abs(center[0] - GlobalParams.M_R_XLength / 2.0) < 0.0001) {
        for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
          ((cell->face(i))->line(j))->get_dof_indices(local_line_dofs);
          for (unsigned int k = 0; k < fe.dofs_per_line; k++) {
            if (locally_owned_dofs.is_element(local_line_dofs[k])) {
              cm.add_line(local_line_dofs[k]);
              cm.set_inhomogeneity(local_line_dofs[k], 0.0);
              fixed_dofs.add_index(local_line_dofs[k]);
            }
          }
        }
        if (face_own_count > 0) {
          cell->face(i)->get_dof_indices(local_face_dofs);
          for (unsigned int j =
              GeometryInfo<3>::lines_per_face * fe.dofs_per_line;
              j < fe.dofs_per_face; j++) {
            if (locally_owned_dofs.is_element(local_face_dofs[j])) {
              cm.add_line(local_face_dofs[j]);
              cm.set_inhomogeneity(local_face_dofs[j], 0.0);
              fixed_dofs.add_index(local_face_dofs[j]);
            }
          }
        }
      }

      if (std::abs(center[1] - GlobalParams.M_R_YLength / 2.0) < 0.0001) {
        for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
          ((cell->face(i))->line(j))->get_dof_indices(local_line_dofs);
          for (unsigned int k = 0; k < fe.dofs_per_line; k++) {
            if (locally_owned_dofs.is_element(local_line_dofs[k])) {
              cm.add_line(local_line_dofs[k]);
              cm.set_inhomogeneity(local_line_dofs[k], 0.0);
              fixed_dofs.add_index(local_line_dofs[k]);
            }
          }
        }
        if (face_own_count > 0) {
          cell->face(i)->get_dof_indices(local_face_dofs);
          for (unsigned int j =
              GeometryInfo<3>::lines_per_face * fe.dofs_per_line;
              j < fe.dofs_per_face; j++) {
            if (locally_owned_dofs.is_element(local_face_dofs[j])) {
              cm.add_line(local_face_dofs[j]);
              cm.set_inhomogeneity(local_face_dofs[j], 0.0);
              fixed_dofs.add_index(local_face_dofs[j]);
            }
          }
        }
      }

      if (std::abs(center[2] - GlobalParams.Minimum_Z) < 0.0001) {
        for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
          if ((cell->face(i))->line(j)->at_boundary()) {
            ((cell->face(i))->line(j))->get_dof_indices(local_line_dofs);
            for (unsigned int k = 0; k < fe.dofs_per_line; k++) {
              if (locally_owned_dofs.is_element(local_line_dofs[k])) {
                fixed_dofs.add_index(local_line_dofs[k]);
              }
            }
            if (fe.dofs_per_line >= 2) {
              cm.add_line(local_line_dofs[1]);
              cm.set_inhomogeneity(local_line_dofs[1], 0.0);
            }
          }
        }
        if (face_own_count > 0) {
          cell->face(i)->get_dof_indices(local_face_dofs);
          for (unsigned int j =
              GeometryInfo<3>::lines_per_face * fe.dofs_per_line;
              j < fe.dofs_per_face; j++) {
            if (locally_owned_dofs.is_element(local_face_dofs[j])) {
              fixed_dofs.add_index(local_face_dofs[j]);
            }
          }
        }
      }

      if (std::abs(center[2] - GlobalParams.Maximum_Z) < 0.0001) {
        for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
          ((cell->face(i))->line(j))->get_dof_indices(local_line_dofs);
          for (unsigned int k = 0; k < fe.dofs_per_line; k++) {
            if (locally_owned_dofs.is_element(local_line_dofs[k])) {
              cm.add_line(local_line_dofs[k]);
              cm.set_inhomogeneity(local_line_dofs[k], 0.0);
              fixed_dofs.add_index(local_line_dofs[k]);
            }
          }
        }
        if (face_own_count > 0) {
          cell->face(i)->get_dof_indices(local_face_dofs);
          for (unsigned int j =
              GeometryInfo<3>::lines_per_face * fe.dofs_per_line;
              j < fe.dofs_per_face; j++) {
            if (locally_owned_dofs.is_element(local_face_dofs[j])) {
              cm.add_line(local_face_dofs[j]);
              cm.set_inhomogeneity(local_face_dofs[j], 0.0);
              fixed_dofs.add_index(local_face_dofs[j]);
            }
          }
        }
      }
    }
  }
}

void Waveguide::ProjectBoundaryConditions() {
  dealii::ZeroFunction<3, double> zf(6);

  if (even) {
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 4,
        cm_prec_even);
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 5,
        cm_prec_odd);
  } else {
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf,
        5, cm_prec_even);
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf,
        4, cm_prec_odd);
  }
}

void Waveguide::MakePreconditionerBoundaryConditions() {
  dealii::DoFHandler<3>::active_cell_iterator cell_loc, endc;
  cell_loc = dof_handler.begin_active();
  endc = dof_handler.end();
  dealii::ZeroFunction<3, double> zf(6);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 0,
      cm_prec_even);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 1,
      cm_prec_even);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 2,
      cm_prec_even);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 3,
      cm_prec_even);

  if (GlobalParams.MPI_Rank == 0) {
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 4,
        cm_prec_even);
  }

  if (GlobalParams.MPI_Rank + 1 == GlobalParams.NumberProcesses) {
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 5,
        cm_prec_even);
  }

  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 0,
      cm_prec_odd);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 1,
      cm_prec_odd);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 2,
      cm_prec_odd);
  VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 3,
      cm_prec_odd);

  if (GlobalParams.MPI_Rank == 0) {
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 4,
        cm_prec_odd);
  }

  if (GlobalParams.MPI_Rank + 1 == GlobalParams.NumberProcesses) {
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, zf, 5,
        cm_prec_odd);
  }

  double layer_length = GlobalParams.LayerThickness;
  IndexSet own(n_global_dofs);
  own.add_indices(locally_owned_dofs);
  if (run_number == 0) {
    sweepable.set_size(n_global_dofs);
    sweepable.add_indices(locally_owned_dofs);
  }
  if (rank != 0) {
    own.add_indices(LowerDofs);
  }

  const int face_own_count =
      fe.dofs_per_face - GeometryInfo<3>::lines_per_face * fe.dofs_per_line;
  const bool has_non_edge_dofs = (face_own_count > 0);

  cm_prec_even.merge(
      cm, dealii::ConstraintMatrix::MergeConflictBehavior::right_object_wins,
      true);
  cm_prec_odd.merge(
      cm, dealii::ConstraintMatrix::MergeConflictBehavior::right_object_wins,
      true);

  for (; cell_loc != endc; ++cell_loc) {
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; i++) {
      Point<3, double> center = (cell_loc->face(i))->center(true, false);
      if (center[0] < 0) center[0] *= (-1.0);
      if (center[1] < 0) center[1] *= (-1.0);

      // Set x-boundary values
      if (std::abs(center[0] - GlobalParams.M_R_XLength / 2.0) < 0.0001) {
        Add_Zero_Restraint(&cm_prec_odd, cell_loc, i, fe.dofs_per_line,
            fe.dofs_per_face, has_non_edge_dofs,
            locally_owned_dofs);
        Add_Zero_Restraint(&cm_prec_even, cell_loc, i, fe.dofs_per_line,
            fe.dofs_per_face, has_non_edge_dofs,
            locally_owned_dofs);
      }

      // Set y-boundary values
      if (std::abs(center[1] - GlobalParams.M_R_YLength / 2.0) < 0.0001) {
        Add_Zero_Restraint(&cm_prec_odd, cell_loc, i, fe.dofs_per_line,
            fe.dofs_per_face, has_non_edge_dofs,
            locally_owned_dofs);
        Add_Zero_Restraint(&cm_prec_even, cell_loc, i, fe.dofs_per_line,
            fe.dofs_per_face, has_non_edge_dofs,
            locally_owned_dofs);
      }
      if (even) {
        if (rank != 0) {
          if (std::abs(center[2] - GlobalParams.Minimum_Z -
              (rank * layer_length)) < 0.0001) {
            Add_Zero_Restraint(&cm_prec_even, cell_loc, i, fe.dofs_per_line,
                fe.dofs_per_face, has_non_edge_dofs,
                locally_owned_dofs);
          }
        }

        if (std::abs(center[2] - GlobalParams.Minimum_Z -
            ((rank + 2) * layer_length)) < 0.0001) {
          Add_Zero_Restraint(&cm_prec_even, cell_loc, i, fe.dofs_per_line,
              fe.dofs_per_face, has_non_edge_dofs,
              locally_owned_dofs);
        }

        if (std::abs(center[2] - GlobalParams.Minimum_Z -
            ((rank + 1) * layer_length)) < 0.0001) {
          Add_Zero_Restraint(&cm_prec_odd, cell_loc, i, fe.dofs_per_line,
              fe.dofs_per_face, has_non_edge_dofs,
              locally_owned_dofs);
        }

        if (std::abs(center[2] - GlobalParams.Minimum_Z -
            ((rank - 1) * layer_length)) < 0.0001) {
          Add_Zero_Restraint(&cm_prec_odd, cell_loc, i, fe.dofs_per_line,
              fe.dofs_per_face, has_non_edge_dofs,
              locally_owned_dofs);
        }
      } else {
        if (std::abs(center[2] - GlobalParams.Minimum_Z -
            (rank * layer_length)) < 0.0001) {
          Add_Zero_Restraint(&cm_prec_odd, cell_loc, i, fe.dofs_per_line,
              fe.dofs_per_face, has_non_edge_dofs,
              locally_owned_dofs);
        }

        if (std::abs(center[2] - GlobalParams.Minimum_Z -
            ((rank + 2) * layer_length)) < 0.0001) {
          Add_Zero_Restraint(&cm_prec_odd, cell_loc, i, fe.dofs_per_line,
              fe.dofs_per_face, has_non_edge_dofs,
              locally_owned_dofs);
        }

        if (std::abs(center[2] - GlobalParams.Minimum_Z -
            ((rank + 1) * layer_length)) < 0.0001) {
          Add_Zero_Restraint(&cm_prec_even, cell_loc, i, fe.dofs_per_line,
              fe.dofs_per_face, has_non_edge_dofs,
              locally_owned_dofs);
        }
      }
    }
  }
}

void Waveguide::Add_Zero_Restraint(
    ConstraintMatrix *in_cm,
    DoFHandler<3>::active_cell_iterator &in_cell, unsigned int in_face,
    unsigned int DofsPerLine, unsigned int DofsPerFace, bool in_non_face_dofs,
    IndexSet locally_owned_dofs) {
  std::vector<types::global_dof_index> local_line_dofs(DofsPerLine);
  std::vector<types::global_dof_index> local_face_dofs(DofsPerFace);
  for (unsigned int j = 0; j < GeometryInfo<3>::lines_per_face; j++) {
    ((in_cell->face(in_face))->line(j))->get_dof_indices(local_line_dofs);
    for (unsigned int k = 0; k < DofsPerLine; k++) {
      if (locally_owned_dofs.is_element(local_line_dofs[k])) {
        in_cm->add_line(local_line_dofs[k]);
        in_cm->set_inhomogeneity(local_line_dofs[k], 0.0);
      }
    }
  }
  if (in_non_face_dofs) {
    in_cell->face(in_face)->get_dof_indices(local_face_dofs);
    for (unsigned int i = 0; i < fe.dofs_per_face; i++) {
      local_face_dofs[i] = local_to_global_index(local_face_dofs[i]);
    }
    for (unsigned int j = GeometryInfo<3>::lines_per_face * DofsPerLine;
        j < DofsPerFace; j++) {
      if (locally_owned_dofs.is_element(local_face_dofs[j])) {
        in_cm->add_line(local_face_dofs[j]);
        in_cm->set_inhomogeneity(local_face_dofs[j], 0.0);
      }
    }
  }
}

void Waveguide::solve() {

  SolverControl lsc =
      SolverControl(30, 1.e-5, true, true);

  lsc.log_frequency(1);

  if (run_number != 0) {
    result_file.close();
  }

  result_file.open((solutionpath + "/" + path_prefix + "/solution_of_run_" +
      std::to_string(run_number) + ".dat")
      .c_str());

  if (GlobalParams.So_Solver == SolverOptions::GMRES) {
    if (primal) {
      if (run_number > 0) {
        for (unsigned int i = 0; i < locally_owned_dofs.n_elements(); i++) {
          int index = locally_owned_dofs.nth_index_in_set(i);
          primal_solution[index] = primal_with_relevant[index];
        }
      }
    } else {
      if (run_number > 1) {
        for (unsigned int i = 0; i < locally_owned_dofs.n_elements(); i++) {
          int index = locally_owned_dofs.nth_index_in_set(i);
          dual_solution[index] = dual_with_relevant[index];
        }
      }
    }


    dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::BlockVector> solver(
        lsc, dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::BlockVector>::
        AdditionalData(GlobalParams.So_RestartSteps));

    unsigned int above = 0;
    if ( rank + 1 != GlobalParams.NumberProcesses ) {
      above = locally_owned_dofs_all_processors[rank + 1].n_elements();
    }

    unsigned int below = 0;
    if (rank != 0) {
      below = locally_owned_dofs_all_processors[rank - 1].n_elements();
    }

    deallog << "Initializing Preconditioner. Locally owned dofs: "
        << locally_owned_dofs.n_elements() << " above: " << above
        << " below: " << below << std::endl;
    PreconditionerSweeping sweep(mpi_comm, locally_owned_dofs.n_elements(),
        above, below, interface_dof_count,
        dof_handler.max_couplings_between_dofs(),
        locally_owned_dofs, &fixed_dofs, rank);

    if (rank == 0) {
      sweep.Prepare(*solution);
    }

    MPI_Barrier(mpi_comm);

    if (even) {
      sweep.matrix = &prec_matrix_even.block(rank / 2, rank / 2);
    } else {
      sweep.matrix = &prec_matrix_odd.block((rank + 1) / 2, (rank + 1) / 2);
    }

    if ( rank + 1 == GlobalParams.NumberProcesses ) {
      sweep.matrix = &system_matrix.block(rank, rank);
    }

    int upper =
        std::min((int)(GlobalParams.NumberProcesses - 1), (int)(rank + 1));
    int lower = std::max((int)(rank - 1), 0);

    dealii::DynamicSparsityPattern upper_block_sp(
        system_matrix.block(rank, upper).m(),
        system_matrix.block(rank, upper).n());
    dealii::DynamicSparsityPattern lower_block_sp(
        system_matrix.block(rank, lower).m(),
        system_matrix.block(rank, lower).n());

    if (GlobalParams.NumberProcesses != rank + 1) {
      for (auto it = system_matrix.block(rank, upper).begin();
          it != system_matrix.block(rank, upper).end(); it++) {
        upper_block_sp.add(it->row(), it->column());
      }
    }
    SparsityPattern usp;
    usp.copy_from(upper_block_sp);
    dealii::SparseMatrix<double> upper_block(usp);
    if (GlobalParams.NumberProcesses != rank + 1) {
      for (auto it = system_matrix.block(rank, upper).begin();
          it != system_matrix.block(rank, upper).end(); it++) {
        upper_block.set(it->row(), it->column(), it->value());
      }
    }

    if (rank != 0) {
      for (auto it = system_matrix.block(rank, lower).begin();
          it != system_matrix.block(rank, lower).end(); it++) {
        lower_block_sp.add(it->row(), it->column());
      }
    }
    SparsityPattern lsp;
    lsp.copy_from(lower_block_sp);
    dealii::SparseMatrix<double> lower_block(lsp);
    if (rank != 0) {
      for (auto it = system_matrix.block(rank, lower).begin();
          it != system_matrix.block(rank, lower).end(); it++) {
        lower_block.set(it->row(), it->column(), it->value());
      }
    }
    timer.enter_subsection("Preconditioner Initialization");

    sweep.init(solver_control, &upper_block, &lower_block);

    solver.connect(std_cxx11::bind(&Waveguide::residual_tracker, this,
        std_cxx11::_1, std_cxx11::_2,
        std_cxx11::_3));
    timer.leave_subsection();

    deallog << "Preconditioner Ready. Solving..." << std::endl;

    struct timeval tp;
    gettimeofday(&tp, NULL);
    solver_start_milis = tp.tv_sec * 1000 + tp.tv_usec / 1000;
    timer.enter_subsection("GMRES run");
    try {
      if (primal) {
        solver.solve(system_matrix, primal_solution, system_rhs, sweep);
      } else {
        solver.solve(system_matrix, dual_solution, system_rhs, sweep);
      }
    } catch (const dealii::SolverControl::NoConvergence &e) {
      deallog << "NO CONVERGENCE!" << std::endl;
    }
    timer.leave_subsection();

    while (steps < 40) {
      struct timeval tp;
      gettimeofday(&tp, NULL);
      int64_t ms = tp.tv_sec * 1000 + tp.tv_usec / 1000 - solver_start_milis;

      Convergence_Table.add_value(
          path_prefix + std::to_string(run_number) + "Iteration", steps + 1);
      Convergence_Table.add_value(
          path_prefix + std::to_string(run_number) + "Residual", 0.0);
      Convergence_Table.add_value(
          path_prefix + std::to_string(run_number) + "Time",
          std::to_string(ms));
      steps++;
    }

    if ((GlobalParams.O_C_D_ConvergenceFirst && run_number == 0) ||
        GlobalParams.O_C_D_ConvergenceAll) {
      Convergence_Table.add_column_to_supercolumn(
          path_prefix + std::to_string(run_number) + "Iteration",
          "Run " + std::to_string(run_number));
      Convergence_Table.add_column_to_supercolumn(
          path_prefix + std::to_string(run_number) + "Residual",
          "Run " + std::to_string(run_number));
      Convergence_Table.add_column_to_supercolumn(
          path_prefix + std::to_string(run_number) + "Time",
          "Run " + std::to_string(run_number));
      Convergence_Table.evaluate_convergence_rates(
          path_prefix + std::to_string(run_number) + "Residual",
          path_prefix + std::to_string(run_number) + "Iteration",
          ConvergenceTable::RateMode::reduction_rate);
    }

    deallog << "Norm of the solution: " << solution->l2_norm() << std::endl;
  }

  if (primal) {
    cm.distribute(primal_solution);
  } else {
    cm.distribute(dual_solution);
  }

  if (GlobalParams.So_Solver == SolverOptions::UMFPACK) {
    SolverControl sc2(2, false, false);
    TrilinosWrappers::SolverDirect temp_s(
        sc2, TrilinosWrappers::SolverDirect::AdditionalData(
            false, PrecOptionNames[GlobalParams.So_Preconditioner]));
  }
  
  solution->update_ghost_values();
  
  double *interface_vals = new double[interface_dof_count];
  if (GlobalParams.MPI_Rank < GlobalParams.NumberProcesses - 1) {
    for (unsigned int i = 0; i < interface_dof_count; i++) {
      interface_vals[i] = locally_owned_dofs.nth_index_in_set(
          locally_relevant_dofs.n_elements() - interface_dof_count + i);
    }
  }
  double *interface_vals_received = new double[interface_dof_count];
  if (GlobalParams.MPI_Rank == 0) {
    MPI_Send(interface_vals, interface_dof_count, MPI_DOUBLE, 1, 1,
        MPI_COMM_WORLD);
  }
  if (GlobalParams.MPI_Rank > 0 &&
      GlobalParams.MPI_Rank < GlobalParams.NumberProcesses - 1) {
    MPI_Recv(interface_vals_received, interface_dof_count, MPI_DOUBLE,
        GlobalParams.MPI_Rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(interface_vals, interface_dof_count, MPI_DOUBLE,
        GlobalParams.MPI_Rank + 1, 1, MPI_COMM_WORLD);
  }
  if (GlobalParams.MPI_Rank == GlobalParams.NumberProcesses - 1) {
    MPI_Recv(interface_vals_received, interface_dof_count, MPI_DOUBLE,
        GlobalParams.MPI_Rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (primal) {
    primal_with_relevant.reinit(n_global_dofs);
  } else {
    dual_with_relevant.reinit(n_global_dofs);
  }
  if (GlobalParams.MPI_Rank > 0) {
    for (unsigned int i = 0; i < n_dofs; i++) {
      if (i < interface_dof_count) {
        int idx2 = i + (n_dofs - interface_dof_count);
        idx2 = local_to_global_index(idx2);
        idx2 -= (n_dofs - interface_dof_count);
        if (primal){
          primal_with_relevant[local_to_global_index(i)] = 0;
        } else {
          dual_with_relevant[local_to_global_index(i)] = 0;
        }

      } else {
        int temp = local_to_global_index(i);
        if (primal) {
          primal_with_relevant[temp] = solution->operator[](temp);
        } else {
          dual_with_relevant[temp] = solution->operator[](temp);
        }
      }
    }

  } else {
    if (primal) {
      for (unsigned int i = 0; i < n_dofs; i++) {
        primal_with_relevant[i] = solution->operator[](i);
      }
    } else {
      for (unsigned int i = 0; i < n_dofs; i++) {
        dual_with_relevant[i] = solution->operator[](i);
      }
    }
  }
  
  GrowingVectorMemory<
  TrilinosWrappers::MPI::BlockVector>::release_unused_memory();
}

void Waveguide::store() {
  reinit_storage();
  // storage.reinit(dof_handler.n_dofs());
  if (primal) {
    storage = primal_solution;
  } else {
    storage = dual_solution;
  }
  is_stored = true;
}

void Waveguide::output_results(bool) {
  if (true) {
    DataOut<3> data_out;

    data_out.attach_dof_handler(dof_handler);
    if (primal) {
      data_out.add_data_vector(
          primal_with_relevant, "Solution",
          dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3,
          3>::DataVectorType::type_dof_data);
    } else {
      data_out.add_data_vector(
          dual_with_relevant, "Solution",
          dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3,
          3>::DataVectorType::type_dof_data);
    }

    data_out.build_patches();

    std::ofstream outputvtu(solutionpath + "/" + path_prefix + "/solution-run" +
        std::to_string(run_number) + "-P" +
        std::to_string(rank) + ".vtu");
    data_out.write_vtu(outputvtu);

    if (rank == 0) {
      std::ofstream outputpvtu(solutionpath + "/" + path_prefix +
          "/solution-run" + std::to_string(run_number) +
          ".pvtu");
      std::vector<std::string> filenames;
      for (unsigned int i = 0; i < GlobalParams.NumberProcesses; i++) {
        filenames.push_back("solution-run" + std::to_string(run_number) + "-P" +
            std::to_string(i) + ".vtu");
      }
      data_out.write_pvtu_record(outputpvtu, filenames);
    }

    if (false) {
      std::ofstream pattern(solutionpath + "/" + path_prefix + "/pattern.gnu");

      std::ofstream patternscript(solutionpath + "/" + path_prefix +
          "/displaypattern.gnu");
      patternscript << "set style line 1000 lw 1 lc \"black\"" << std::endl;
      for (int i = 0; i < GlobalParams.M_W_Sectors; i++) {
        patternscript << "set arrow " << 1000 + 2 * i << " from 0,-"
            << Dofs_Below_Subdomain[i] << " to " << n_global_dofs
            << ",-" << Dofs_Below_Subdomain[i]
                                            << " nohead ls 1000 front" << std::endl;
        patternscript << "set arrow " << 1001 + 2 * i << " from "
            << Dofs_Below_Subdomain[i] << ",0 to "
            << Dofs_Below_Subdomain[i] << ", -" << n_global_dofs
            << " nohead ls 1000 front" << std::endl;
      }
      patternscript << "set arrow " << 1000 + 2 * GlobalParams.M_W_Sectors
          << " from 0,-" << n_global_dofs << " to " << n_global_dofs
          << ",-" << n_global_dofs << " nohead ls 1000 front"
          << std::endl;
      patternscript << "set arrow " << 1001 + 2 * GlobalParams.M_W_Sectors
          << " from " << n_global_dofs << ",0 to " << n_global_dofs
          << ", -" << n_global_dofs << " nohead ls 1000 front"
          << std::endl;

      patternscript << "plot \"pattern.gnu\" with dots" << std::endl;
      patternscript.flush();
    }
  }
  MPI_Barrier(mpi_comm);
  if (GlobalParams.O_O_V_S_SolutionFirst) {

    set_the_st(this->st);

    GridTools::transform(&Triangulation_Transform_to_physical, triangulation);

    MPI_Barrier(mpi_comm);

    DataOut<3> data_out;

    data_out.attach_dof_handler(dof_handler);
    if (primal) {
      data_out.add_data_vector(
          primal_with_relevant, "Solution",
          dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3,
          3>::DataVectorType::type_dof_data);
    } else {
      data_out.add_data_vector(
          dual_with_relevant, "Solution",
          dealii::DataOut_DoFData<dealii::DoFHandler<3>, 3,
          3>::DataVectorType::type_dof_data);
    }
    data_out.build_patches();

    std::ofstream outputvtu(
        solutionpath + "/" + path_prefix + "/solution-transformed-run" +
        std::to_string(run_number) + "-P" + std::to_string(rank) + ".vtu");
    data_out.write_vtu(outputvtu);
    MPI_Barrier(mpi_comm);
  }
}

Point<3, double> Waveguide::transform_coordinate(const Point<3, double> in_p) {
  return st->math_to_phys(in_p);
}

void Waveguide::run() {
  deallog.push("Waveguide_" + path_prefix + "_run");

  if (run_number == 0) {
    deallog << "Setting up the mesh..." << std::endl;
    timer.enter_subsection("Setup Mesh");
    make_grid();
    timer.leave_subsection();

    Compute_Dof_Numbers();

    deallog << "Setting up FEM..." << std::endl;
    timer.enter_subsection("Setup FEM");
    setup_system();
    timer.leave_subsection();

    timer.enter_subsection("Reset");
    timer.leave_subsection();

  } else {
    Prepare_Boundary_Constraints();

    timer.enter_subsection("Reset");
    reinit_all();
    timer.leave_subsection();
  }

  deallog.push("Assembly");
  deallog << "Assembling the system..." << std::endl;
  timer.enter_subsection("Assemble");
  assemble_system();
  timer.leave_subsection();
  deallog.pop();

  deallog.push("Solving");
  deallog << "Solving the system..." << std::endl;
  timer.enter_subsection("Solve");
  solve();
  timer.leave_subsection();
  deallog.pop();

  timer.enter_subsection("Evaluate");
  timer.leave_subsection();

  timer.print_summary();

  deallog << "Writing outputs..." << std::endl;
  timer.reset();

  output_results(false);

  deallog.pop();
  run_number++;
}

void Waveguide::print_eigenvalues(
    const std::vector<std::complex<double>> &input) {
  for (unsigned int i = 0; i < input.size(); i++) {
    eigenvalue_file << input.at(i).real() << "  " << input.at(i).imag()
                        << std::endl;
  }
  eigenvalue_file << std::endl;
}

void Waveguide::print_condition(double condition) {
  condition_file << condition << std::endl;
}

std::vector<std::complex<double>>
Waveguide::assemble_adjoint_local_contribution(double stepwidth) {
  deallog.push("Waveguide:adj_local");

  deallog << "Computing adjoint based shape derivative contributions..."
      << std::endl;

  int other_proc = GlobalParams.NumberProcesses - rank - 1;
  const unsigned int ndofs = st->NDofs();
  std::vector<std::complex<double>> ret;
  ret.resize(ndofs);
  for (unsigned int i = 0; i < ndofs; i++) {
    ret[i] = 0;
  }
  std::vector<bool> local_supported_dof;
  local_supported_dof.resize(ndofs);
  int min = ndofs;
  int max = -1;
  for (int i = 0; i < static_cast<int>(ndofs); i++) {
    if (st->IsDofFree(i)) {
      std::pair<double, double> support = st->dof_support(i);
      if ((support.first <= minimum_local_z &&
          support.second >= maximum_local_z) ||
          (support.first >= minimum_local_z &&
              support.second <= maximum_local_z) ||
              (support.first <= maximum_local_z &&
                  support.second >= minimum_local_z)) {
        if (i > max) {
          max = i;
        }
        if (i < min) {
          min = i;
        }
      }
    }
  }

  QGauss<3> quadrature_formula(1);
  const FEValuesExtractors::Vector real(0);
  const FEValuesExtractors::Vector imag(3);
  FEValues<3> fe_values(fe, quadrature_formula,
      update_values | update_gradients | update_JxW_values |
      update_quadrature_points);
  std::vector<Point<3>> quadrature_points;
  const unsigned int n_q_points = quadrature_formula.size();

  Tensor<2, 3, std::complex<double>> transformation;

  int total = triangulation.n_active_cells() * quadrature_formula.size();
  int counter = 0;
  double *returned_vector = new double[6];
  for (unsigned int temp_counter = 0; temp_counter < 2; temp_counter++) {
    if (((GlobalParams.NumberProcesses % 2 == 1) &&
        (rank + 1 == GlobalParams.NumberProcesses / 2 ) &&
        temp_counter == 0)) {
      deallog.push("middle phase");
      deallog << "This process is now computing its own contributions to the "
          "shape gradient."
          << std::endl;

      DoFHandler<3>::active_cell_iterator cell, endc;
      cell = dof_handler.begin_active(), endc = dof_handler.end();
      for (; cell != endc; ++cell) {
        if (cell->is_locally_owned()) {
          fe_values.reinit(cell);
          quadrature_points = fe_values.get_quadrature_points();
          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            Tensor<1, 3, std::complex<double>> own_solution =
                solution_evaluation(quadrature_points[q_index]);
            Tensor<1, 3, std::complex<double>> other_solution =
                adjoint_solution_evaluation(quadrature_points[q_index]);

            const double JxW = fe_values.JxW(q_index);
            for (int j = min; j <= max; j++) {
              transformation = st->get_Tensor_for_step(
                  quadrature_points[q_index], j, stepwidth);
              if (st->point_in_dof_support(quadrature_points[q_index], j)) {
                ret[j] += own_solution * transformation * other_solution * JxW;
              }
            }
            counter++;
            if ((counter - 1) / (total / 10) != (counter) / (total / 10)) {
              deallog << static_cast<int>(100 * (counter) / (total)) << "%"
                  << std::endl;
            }
          }
        }
      }

      deallog << "Done." << std::endl;
      deallog.pop();

    } else {
      if (rank >= temp_counter * (GlobalParams.NumberProcesses) / 2 &&
          rank < (1 + temp_counter) * (GlobalParams.NumberProcesses) / 2) {
        deallog.push("local cell phase");
        deallog << "This process is now computing its own contributions to the "
            "shape gradient together with "
            << other_proc << "." << std::endl;

        DoFHandler<3>::active_cell_iterator cell, endc;
        cell = dof_handler.begin_active(), endc = dof_handler.end();
        for (; cell != endc; ++cell) {
          if (cell->is_locally_owned()) {
            fe_values.reinit(cell);
            quadrature_points = fe_values.get_quadrature_points();
            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
              Tensor<1, 3, std::complex<double>> own_solution =
                  solution_evaluation(quadrature_points[q_index]);
              MPI_Send(&quadrature_points[q_index][0], 3, MPI_DOUBLE,
                  other_proc, 0, mpi_comm);
              MPI_Recv(&returned_vector[0], 6, MPI_DOUBLE, other_proc, 0,
                  mpi_comm, MPI_STATUS_IGNORE);
              Tensor<1, 3, std::complex<double>> other_solution;

              other_solution[0].real(returned_vector[0]);
              other_solution[0].imag(-returned_vector[1]);
              other_solution[1].real(returned_vector[2]);
              other_solution[1].imag(-returned_vector[3]);
              other_solution[2].real(returned_vector[4]);
              other_solution[2].imag(-returned_vector[5]);

              const double JxW = fe_values.JxW(q_index);
              for (int j = min; j <= max; j++) {
                transformation = st->get_Tensor_for_step(
                    quadrature_points[q_index], j, stepwidth);
                if (st->point_in_dof_support(quadrature_points[q_index], j)) {
                  ret[j] +=
                      own_solution * transformation * other_solution * JxW;
                }
              }
              counter++;
              if ((counter - 1) / (total / 10) != (counter) / (total / 10)) {
                deallog << static_cast<int>(100 * (counter) / (total)) << "%"
                    << std::endl;
              }
            }
          }
        }

        double *end_signal = new double[3];
        end_signal[0] = GlobalParams.Minimum_Z - 10.0;
        end_signal[1] = GlobalParams.Minimum_Z - 10.0;
        end_signal[2] = GlobalParams.Minimum_Z - 10.0;

        MPI_Send(&end_signal[0], 3, MPI_DOUBLE, other_proc, 0, mpi_comm);
        deallog << "Done." << std::endl;
        deallog.pop();

      } else {
        deallog.push("non-local cell phase");

        deallog
        << "This process is now adjoint based contributions for process "
        << other_proc << "." << std::endl;
        bool normal = true;
        while (normal) {
          double *position_array = new double[3];
          MPI_Recv(&position_array[0], 3, MPI_DOUBLE, other_proc, 0, mpi_comm,
              MPI_STATUS_IGNORE);
          normal = false;
          for (int i = 0; i < 3; i++) {
            if (position_array[i] != GlobalParams.Minimum_Z - 10.0) {
              normal = true;
            }
          }
          // deallog << "Received request for (" << position_array[0] << ", "<<
          // position_array[1] << ", "<<position_array[2]<<")"<<std::endl;
          if (normal) {
            double *result = new double[6];
            Point<3, double> position;
            for (int i = 0; i < 3; i++) {
              position[i] = position_array[i];
            }
            adjoint_solution_evaluation(position, result);
            MPI_Send(&result[0], 6, MPI_DOUBLE, other_proc, 0, mpi_comm);
          }
          // deallog << "Sent a solution."<<std::endl;
        }
        deallog << "Done." << std::endl;
        deallog.pop();
      }
    }

    MPI_Barrier(mpi_comm);
  }

  deallog << "Done. Communicating: " << std::endl;

  double *input = new double[2 * ndofs];
  double *output = new double[2 * ndofs];

  for (unsigned int i = 0; i < ndofs; i++) {
    input[2 * i] = ret[i].real();
    input[2 * i + 1] = ret[i].imag();
  }

  MPI_Allreduce(input, output, 2 * ndofs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  for (unsigned int i = 0; i < ndofs; i++) {
    ret[i] = -1.0 * std::complex<double>(output[2 * i + 1], output[2 * i]) /
        stepwidth;
  }

  delete input;
  delete output;

  deallog.pop();
  return ret;
}

Tensor<1, 3, std::complex<double>> Waveguide::solution_evaluation(
    Point<3, double> position) const {
  Tensor<1, 3, std::complex<double>> ret;
  Vector<double> result(6);

  VectorTools::point_value(dof_handler, primal_with_relevant, position, result);

  ret[0] = std::complex<double>(result(0), result(3));
  ret[1] = std::complex<double>(result(1), result(4));
  ret[2] = std::complex<double>(result(2), result(5));
  return ret;
}

void Waveguide::solution_evaluation(Point<3, double> position,
    double *sol) const {
  Tensor<1, 3, std::complex<double>> ret;
  Vector<double> result(6);
  VectorTools::point_value(dof_handler, primal_with_relevant, position, result);
  for (int i = 0; i < 6; i++) {
    sol[i] = result(i);
  }
}

Tensor<1, 3, std::complex<double>> Waveguide::adjoint_solution_evaluation(
    Point<3, double> position) const {
  Tensor<1, 3, std::complex<double>> ret;
  Vector<double> result(6);
  position[2] = -position[2];
  VectorTools::point_value(dof_handler, dual_with_relevant, position, result);
  ret[0] = std::complex<double>(result(0), result(3));
  ret[1] = std::complex<double>(-result(1), -result(4));
  ret[2] = std::complex<double>(-result(2), -result(5));
  return ret;
}

void Waveguide::adjoint_solution_evaluation(Point<3, double> position,
    double *sol) const {
  Tensor<1, 3, std::complex<double>> ret;
  Vector<double> result(6);
  position[2] = -position[2];
  VectorTools::point_value(dof_handler, dual_with_relevant, position, result);
  for (int i = 0; i < 6; i++) {
    sol[i] = -result(i);
  }
  sol[0] *= -1;
  sol[1] *= -1;
}

void Waveguide::reset_changes() { reinit_all(); }

SolverControl::State Waveguide::residual_tracker(
    unsigned int Iteration, double residual,
    dealii::TrilinosWrappers::MPI::BlockVector) {
  if ((GlobalParams.O_C_D_ConvergenceFirst && run_number == 0) ||
      GlobalParams.O_C_D_ConvergenceAll) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    int64_t ms = tp.tv_sec * 1000 + tp.tv_usec / 1000 - solver_start_milis;

    Convergence_Table.add_value(
        path_prefix + std::to_string(run_number) + "Iteration", Iteration);
    Convergence_Table.add_value(
        path_prefix + std::to_string(run_number) + "Residual", residual);
    Convergence_Table.add_value(
        path_prefix + std::to_string(run_number) + "Time", std::to_string(ms));
  }
  steps = Iteration;
  return SolverControl::success;
  ;
}

#endif
