//
// Created by pascal on 03.02.20.
//

#include "LocalProblem.h"
#include "../HSIEPreconditioner/HSIESurface.h"

LocalProblem::LocalProblem(unsigned int, unsigned int global_level) :
    HierarchicalProblem(0, global_level), base_problem() {
  base_problem.make_grid();

}

LocalProblem::~LocalProblem() {

}

void LocalProblem::solve() {}

void LocalProblem::initialize() {
  dealii::Triangulation<2, 3> temp_triangulation;
  dealii::Triangulation<2> surf_tria;
  std::complex<double> k0;
  std::map<dealii::Triangulation<2, 3>::cell_iterator,
      dealii::Triangulation<3, 3>::face_iterator> association;
  for (unsigned int side = 0; side < 6; side++) {
    dealii::Triangulation<3> tria;
    tria.copy_triangulation(base_problem.triangulation);
    std::set<unsigned int> b_ids;
    b_ids.insert(side);
    switch (side) {
    case 0:
      dealii::GridTools::transform(Transform_0_to_5, tria);
      break;
    case 1:
      dealii::GridTools::transform(Transform_1_to_5, tria);
      break;
    case 2:
      dealii::GridTools::transform(Transform_2_to_5, tria);
      break;
    case 3:
      dealii::GridTools::transform(Transform_3_to_5, tria);
      break;
    case 4:
      dealii::GridTools::transform(Transform_4_to_5, tria);
      break;
    default:
      break;
    }
    association = dealii::GridGenerator::extract_boundary_mesh(tria,
        temp_triangulation, b_ids);
    dealii::GridGenerator::flatten_triangulation(temp_triangulation, surf_tria);
    switch (side) {
    case 0:
      surface_0 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 1:
      surface_1 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 2:
      surface_2 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 3:
      surface_3 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 4:
      surface_4 = new HSIESurface(5, std::ref(surf_tria), side, 0,
          GlobalParams.So_ElementOrder, k0, std::ref(association));
      break;
    case 5:
      surface_5 = new HSIESurface(5, surf_tria, side, 0,
          GlobalParams.So_ElementOrder, k0, association);
      break;
    default:
      break;
    }
  }
}

void LocalProblem::generate_sparsity_pattern() {
}

unsigned int LocalProblem::compute_own_dofs() {
  // Todo this is the number of inner dofs + HSIE dofs for every direction.
  return 0;
}

void LocalProblem::assemble() {

}

void LocalProblem::initialize_index_sets() {

}

void LocalProblem::apply_sweep(
    dealii::LinearAlgebra::distributed::Vector<double>) {

}

unsigned int LocalProblem::compute_lower_interface_dof_count() {
  // For local problems there are not interfaces.
  return 0;
}

unsigned int LocalProblem::compute_upper_interface_dof_count() {
  // For local problems there are not interfaces.
  return 0;
}

dealii::IndexSet LocalProblem::get_owned_dofs_for_level(unsigned int level) {
  unsigned int n_owned_dofs = compute_own_dofs();
  if (level >= 2) {
    n_owned_dofs += 1;  // TODO: implement this.
  }
  if (level >= 1) {
    n_owned_dofs += 1; // TODO: implement this.
  }
  return dealii::IndexSet(n_owned_dofs);
}

LocalProblem* LocalProblem::get_local_problem() {
  return this;
}
