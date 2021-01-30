#include "./PMLSurface.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include "../Core/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include "./BoundaryCondition.h"

PMLSurface::PMLSurface(unsigned int in_bid, double in_additional_coordinate, dealii::Triangulation<2> & in_surf_tria)
    :BoundaryCondition(in_bid, in_additional_coordinate, in_surf_tria),
    fe_nedelec(GlobalParams.Nedelec_element_order)
{
    constraints_made = false;   
}

PMLSurface::~PMLSurface() {}

void PMLSurface::prepare_mesh() {
    dealii::GridGenerator::extrude_triangulation(surface_triangulation, GlobalParams.PML_N_Layers, GlobalParams.PML_thickness, triangulation);
    if(b_id == 4) {
        dealii::GridTools::transform(Transform_5_to_4, triangulation);
    }
    if(b_id == 3) {
        dealii::GridTools::transform(Transform_5_to_3, triangulation);
    }
    if(b_id == 2) {
        dealii::GridTools::transform(Transform_5_to_2, triangulation);
    }
    if(b_id == 1) {
        dealii::GridTools::transform(Transform_5_to_1, triangulation);
    }
    if(b_id == 0) {
        dealii::GridTools::transform(Transform_5_to_0, triangulation);
    }
    dealii::Tensor<1, 3> shift_vector;
    for(unsigned int i = 0; i < 3; i++) {
        shift_vector[i] = 0;
    }
    if(b_id == 0 || b_id == 1) {
        inner_boundary_id = 0;
        outer_boundary_id = 1;
        shift_vector[0] = additional_coordinate;
    }
    if(b_id == 2 || b_id == 3) {
        inner_boundary_id = 2;
        outer_boundary_id = 3;
        shift_vector[1] = additional_coordinate;
    }
    if(b_id == 4 || b_id == 5) {
        inner_boundary_id = 4;
        outer_boundary_id = 5;
        shift_vector[2] = additional_coordinate;
    }
    dealii::GridTools::shift(shift_vector, triangulation);
}

unsigned int PMLSurface::cells_for_boundary_id(unsigned int boundary_id) {
    unsigned int ret = 0;
    for(auto it = triangulation.begin(); it!= triangulation.end(); it++) {
        if(it->at_boundary(boundary_id)) {
            ret++;
        }
    }
    return ret;
}

void PMLSurface::init_fe() {
    dof_h_nedelec.initialize(triangulation, fe_nedelec);
    dof_h_nedelec.distribute_dofs(fe_nedelec);
    dof_counter = dof_h_nedelec.n_dofs();
    sort_dofs();
}

void PMLSurface::identify_corner_cells() {

}

void PMLSurface::initialize() {
  prepare_mesh();
  init_fe();
  make_inner_constraints();
}

bool PMLSurface::is_point_at_boundary(Position2D in_p, BoundaryId in_bid) {
  return false;
}

void PMLSurface::sort_dofs() {
  triangulation.clear_user_flags();
  std::vector<std::pair<DofNumber, Position>> current;
  std::vector<types::global_dof_index> local_line_dofs(fe_nedelec.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe_nedelec.dofs_per_face);
  std::set<DofNumber> face_set;
  std::vector<DofNumber> local_cell_dofs(fe_nedelec.dofs_per_cell);
  std::set<DofNumber> cell_set;
  auto cell = dof_h_nedelec.begin_active();
  auto endc = dof_h_nedelec.end();
  for (; cell != endc; ++cell) {
    cell->get_dof_indices(local_cell_dofs);
    cell_set.clear();
    cell_set.insert(local_cell_dofs.begin(), local_cell_dofs.end());
    for(unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; face++) {
      cell->face(face)->get_dof_indices(local_face_dofs);
      face_set.clear();
      face_set.insert(local_face_dofs.begin(), local_face_dofs.end());
      for(auto firstit : face_set) {
        cell_set.erase(firstit);
      }
      for(unsigned int line = 0; line < GeometryInfo<3>::lines_per_face; line++) {
        cell->face(face)->line(line)->get_dof_indices(local_line_dofs);
        line_set.clear();
        line_set.insert(local_line_dofs.begin(), local_line_dofs.end());
        for(auto firstit : line_set) {
          face_set.erase(firstit);
        }
        if(!cell->face(face)->line(line)->user_flag_set()){
          for(auto dof: line_set) {
            current.emplace_back(dof, cell->face(face)->line(line)->center());
          }
          cell->face(face)->line(line)->set_user_flag();
        }
      }
      if(!cell->face(face)->user_flag_set()){
        for(auto dof: face_set) {
          current.emplace_back(dof, cell->face(face)->center());
        }
        cell->face(face)->set_user_flag();
      }
    }
    for(auto dof: cell_set) {
      current.emplace_back(dof, cell->center());
    }
  }
  std::sort(current.begin(), current.end(), compareDofBaseData);
  std::vector<unsigned int> new_numbering;
  new_numbering.resize(current.size());
  for (unsigned int i = 0; i < current.size(); i++) {
    new_numbering[current[i].first] = i;
  }
  dof_h_nedelec.renumber_dofs(new_numbering);
}

std::vector<DofIndexAndOrientationAndPosition> PMLSurface::get_dof_association_by_boundary_id(unsigned int in_bid) {
    std::vector<DofIndexAndOrientationAndPosition> ret;
  std::vector<types::global_dof_index> local_line_dofs(fe_nedelec.dofs_per_line);
  std::set<DofNumber> line_set;
  std::vector<DofNumber> local_face_dofs(fe_nedelec.dofs_per_face);
  std::set<DofNumber> face_set;
  triangulation.clear_user_flags();
  for (auto cell : dof_h_nedelec.active_cell_iterators()) {
    if (cell->at_boundary(in_bid)) {
      bool found_one = false;
      for (unsigned int face = 0; face < 6; face++) {
        if (cell->face(face)->boundary_id() == in_bid && found_one) {
          print_info("NumericProblem::get_surface_dof_vector_for_boundary_id", "There was an error!", false, LoggingLevel::PRODUCTION_ALL);
        }
        if (cell->face(face)->boundary_id() == in_bid) {
          found_one = true;
          std::vector<DofNumber> face_dofs_indices(fe_nedelec.dofs_per_face);
          cell->face(face)->get_dof_indices(face_dofs_indices);
          face_set.clear();
          face_set.insert(face_dofs_indices.begin(), face_dofs_indices.end());
          std::vector<DofIndexAndOrientationAndPosition> cell_dofs_and_orientations_and_points;
          for (unsigned int i = 0; i < dealii::GeometryInfo<3>::lines_per_face; i++) {
            std::vector<DofNumber> line_dofs(fe_nedelec.dofs_per_line);
            cell->face(face)->line(i)->get_dof_indices(line_dofs);
            line_set.clear();
            line_set.insert(line_dofs.begin(), line_dofs.end());
            for(auto erase_it: line_set) {
              face_set.erase(erase_it);
            }
            if(!cell->face(face)->line(i)->user_flag_set()) {
              for (unsigned int j = 0; j < fe_nedelec.dofs_per_line; j++) {
                DofIndexAndOrientationAndPosition new_item;
                new_item.index = line_dofs[j];
                new_item.position = cell->face(face)->line(i)->center();
                if(cell->face(face)->line(i)->vertex_index(0) < cell->face(face)->line(i)->vertex_index(1)) {
                  new_item.orientation = get_orientation(cell->face(face)->line(i)->vertex(1),cell->face(face)->line(i)->vertex(0));
                } else {
                  new_item.orientation = get_orientation(cell->face(face)->line(i)->vertex(0),cell->face(face)->line(i)->vertex(1));
                }
                cell_dofs_and_orientations_and_points.emplace_back(new_item);
              }
              cell->face(face)->line(i)->set_user_flag();
            }
          }
          for (auto item: face_set) {
            DofIndexAndOrientationAndPosition new_item;
            new_item.index = item;
            new_item.position = cell->face(face)->center();
            new_item.orientation = get_orientation(cell->face(face)->vertex(0), cell->face(face)->vertex(1));
            cell_dofs_and_orientations_and_points.emplace_back(new_item);
          }
          for (auto item: cell_dofs_and_orientations_and_points) {
            ret.push_back(item);
          }
        }
      }
    }
  }
  ret.shrink_to_fit();
  std::sort(ret.begin(), ret.end(), compareDofBaseDataAndOrientation);
  return ret;
}

std::vector<DofIndexAndOrientationAndPosition> PMLSurface::get_dof_association() {
    return get_dof_association_by_boundary_id(b_id);
}

DofCount PMLSurface::get_dof_count_by_boundary_id(BoundaryId in_bid) {
    return (get_dof_association_by_boundary_id(in_bid)).size();
}

double PMLSurface::fraction_of_pml_direction(Position in_p) {
    double delta = 0;
    if(b_id == 0 || b_id == 1) {
        delta = std::abs(in_p[0]-additional_coordinate) / GlobalParams.PML_thickness;
    }
    if(b_id == 2 || b_id == 3) {
        delta = std::abs(in_p[1]-additional_coordinate) / GlobalParams.PML_thickness;
    }
    if(b_id == 4 || b_id == 5) {
        delta = std::abs(in_p[2]-additional_coordinate) / GlobalParams.PML_thickness;
    }
    return delta;
}

dealii::Tensor<2,3,ComplexNumber> PMLSurface::get_pml_tensor_epsilon(Position in_p) {
    dealii::Tensor<2,3,ComplexNumber> ret = get_pml_tensor(in_p);
    ret *= (Geometry.math_coordinate_in_waveguide(in_p))? GlobalParams.Epsilon_R_in_waveguide : GlobalParams.Epsilon_R_outside_waveguide;
    return ret;
}

dealii::Tensor<2,3,ComplexNumber> PMLSurface::get_pml_tensor_mu(Position in_p) {
    dealii::Tensor<2,3,ComplexNumber> ret = invert(get_pml_tensor(in_p));
    return ret;
}

dealii::Tensor<2,3,ComplexNumber> PMLSurface::get_pml_tensor(Position in_p) {
    dealii::Tensor<2,3,ComplexNumber> ret;
    double fraction = fraction_of_pml_direction(in_p);
    ComplexNumber part_a = {1 , std::pow(fraction, GlobalParams.PML_skaling_order) * GlobalParams.PML_Sigma_Max};
    for(unsigned int i = 0; i < 3; i++) {
        for(unsigned int j = 0; j < 3; j++) {
            ret[i][j] = 0;
        }
    }
    for(unsigned int i = 0; i < 3; i++) {
        if(i == b_id/2) {
            ret[i][i] = 1.0 / part_a;
        } else {
            ret[i][i] = part_a;
        }
    }
    return ret;
}

void PMLSurface::make_inner_constraints() {
    IndexSet is(dof_counter);
    is.add_range(0, dof_counter);
    constraints.reinit(is);
    dealii::DoFTools::make_zero_boundary_constraints(dof_h_nedelec, outer_boundary_id, constraints);
    constraints_made = true;
}

void PMLSurface::copy_constraints(dealii::AffineConstraints<ComplexNumber> * in_constraints, unsigned int shift) {
    constraints.shift(shift);
    in_constraints->merge(constraints, dealii::AffineConstraints<ComplexNumber>::MergeConflictBehavior::left_object_wins, true);
    constraints.shift(-shift);
}

struct CellwiseAssemblyDataPML {
  QGauss<3> quadrature_formula; 
  FEValues<3> fe_values;
  std::vector<Position> quadrature_points;
  const unsigned int dofs_per_cell;
  const unsigned int n_q_points;
  FullMatrix<ComplexNumber> cell_matrix;
  Vector<ComplexNumber> cell_rhs;
  std::vector<DofNumber> local_dof_indices;
  DofHandler3D::active_cell_iterator cell;
  DofHandler3D::active_cell_iterator end_cell;
  const FEValuesExtractors::Vector fe_field;
  CellwiseAssemblyDataPML(dealii::FE_NedelecSZ<3> * fe, DofHandler3D * dof_handler):
  quadrature_formula(GlobalParams.Nedelec_element_order + 2),
  fe_values(*fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values |
                            update_quadrature_points),
  dofs_per_cell(fe->dofs_per_cell),
  n_q_points(quadrature_formula.size()),
  cell_matrix(dofs_per_cell,dofs_per_cell),
  cell_rhs(dofs_per_cell),
  local_dof_indices(dofs_per_cell),
  fe_field(0)
  {
    cell_rhs = 0;
    cell = dof_handler->begin_active();
    end_cell = dof_handler->end();
  };

  Position get_position_for_q_index(unsigned int q_index) {
      return quadrature_points[q_index];
  }

  void prepare_for_current_q_index(unsigned int q_index, dealii::Tensor<2, 3, ComplexNumber> epsilon, dealii::Tensor<2,3,ComplexNumber> mu_inverse) {
    
    const double JxW = fe_values.JxW(q_index);
    for (unsigned int i = 0; i < dofs_per_cell; i++) {
      Tensor<1, 3, ComplexNumber> I_Curl;
      Tensor<1, 3, ComplexNumber> I_Val;
      I_Curl = fe_values[fe_field].curl(i, q_index);
      I_Val = fe_values[fe_field].value(i, q_index);

      for (unsigned int j = 0; j < dofs_per_cell; j++) {
        Tensor<1, 3, ComplexNumber> J_Curl;
        Tensor<1, 3, ComplexNumber> J_Val;
        J_Curl = fe_values[fe_field].curl(j, q_index);
        J_Val = fe_values[fe_field].value(j, q_index);

        cell_matrix[i][j] += I_Curl * (mu_inverse * Conjugate_Vector(J_Curl))* JxW
            - ( ( epsilon *  I_Val * Conjugate_Vector(J_Val)) * JxW);
      }
    }
  }

  Tensor<1, 3, ComplexNumber> Conjugate_Vector(
      Tensor<1, 3, ComplexNumber> input) {
    Tensor<1, 3, ComplexNumber> ret;

    for (int i = 0; i < 3; i++) {
      ret[i].real(input[i].real());
      ret[i].imag(-input[i].imag());
    }
    return ret;
  }

};

void PMLSurface::fill_sparsity_pattern(dealii::DynamicSparsityPattern *in_dsp, DofNumber shift, dealii::AffineConstraints<ComplexNumber> *constraints) {
    DynamicSparsityPattern dsp (dof_counter);
    DoFTools::make_sparsity_pattern (dof_h_nedelec, dsp);
    constraints->condense(dsp);
    for(auto it = dsp.begin(); it != dsp.end(); it++) {
        in_dsp->add(it->row() + shift, it->column()+ shift);
    }
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed* rhs, dealii::IndexSet in_is,  std::array<bool, 6> surfaces_hsie,  dealii::AffineConstraints<ComplexNumber> *in_constraints){
    const unsigned int shift = in_is.nth_index_in_set(0);
    fill_matrix(matrix, rhs, shift, surfaces_hsie, in_constraints);
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* , dealii::IndexSet , std::array<bool, 6>, dealii::AffineConstraints<ComplexNumber> *){
    // NOT IMPLEMENTED
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix*, dealii::PETScWrappers::SparseMatrix*, NumericVectorDistributed* , DofNumber , std::array<bool, 6> , dealii::AffineConstraints<ComplexNumber> *){
    // NOT IMPLEMENTED
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::SparseMatrix* matrix, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6>, dealii::AffineConstraints<ComplexNumber> *constraints){
    CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_h_nedelec);
    for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
        cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
        for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
            cell_data.local_dof_indices[i] += shift;
        }
        cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
        cell_data.fe_values.reinit(cell_data.cell);
        cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
        std::vector<types::global_dof_index> input_dofs(fe_nedelec.dofs_per_line);
        IndexSet input_dofs_local_set(fe_nedelec.dofs_per_cell);
        std::vector<Position> input_dof_centers(fe_nedelec.dofs_per_cell);
        std::vector<Tensor<1, 3, double>> input_dof_dirs(fe_nedelec.dofs_per_cell);
        cell_data.cell_matrix = 0;
        for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
            Position pos = cell_data.get_position_for_q_index(q_index);
            dealii::Tensor<2,3,ComplexNumber> epsilon = get_pml_tensor_epsilon(pos);
            dealii::Tensor<2,3,ComplexNumber> mu = get_pml_tensor_mu(pos);
            cell_data.prepare_for_current_q_index(q_index, epsilon, mu);
            constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, false);
        }
    }
    matrix->compress(dealii::VectorOperation::add);
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, dealii::IndexSet in_is, std::array<bool, 6> surfaces_hsie, dealii::AffineConstraints<ComplexNumber> *in_constraints){
    const unsigned int shift = in_is.nth_index_in_set(0);
    fill_matrix(matrix, rhs, shift, surfaces_hsie, in_constraints);
}

void PMLSurface::fill_matrix(dealii::PETScWrappers::MPI::SparseMatrix* matrix, NumericVectorDistributed* rhs, DofNumber shift, std::array<bool, 6>, dealii::AffineConstraints<ComplexNumber> *constraints){
    CellwiseAssemblyDataPML cell_data(&fe_nedelec, &dof_h_nedelec);
    for (; cell_data.cell != cell_data.end_cell; ++cell_data.cell) {
        cell_data.cell->get_dof_indices(cell_data.local_dof_indices);
        for (unsigned int i = 0; i < cell_data.local_dof_indices.size(); i++) {
            cell_data.local_dof_indices[i] += shift;
        }
        cell_data.cell_rhs.reinit(cell_data.dofs_per_cell, false);
        cell_data.fe_values.reinit(cell_data.cell);
        cell_data.quadrature_points = cell_data.fe_values.get_quadrature_points();
        std::vector<types::global_dof_index> input_dofs(fe_nedelec.dofs_per_line);
        IndexSet input_dofs_local_set(fe_nedelec.dofs_per_cell);
        std::vector<Position> input_dof_centers(fe_nedelec.dofs_per_cell);
        std::vector<Tensor<1, 3, double>> input_dof_dirs(fe_nedelec.dofs_per_cell);
        cell_data.cell_matrix = 0;
        for (unsigned int q_index = 0; q_index < cell_data.n_q_points; ++q_index) {
            Position pos = cell_data.get_position_for_q_index(q_index);
            dealii::Tensor<2,3,ComplexNumber> epsilon = get_pml_tensor_epsilon(pos);
            dealii::Tensor<2,3,ComplexNumber> mu = get_pml_tensor_mu(pos);
            cell_data.prepare_for_current_q_index(q_index, epsilon, mu);
            constraints->distribute_local_to_global(cell_data.cell_matrix, cell_data.cell_rhs, cell_data.local_dof_indices,*matrix, *rhs, false);
        }
    }
    matrix->compress(dealii::VectorOperation::add);
}
