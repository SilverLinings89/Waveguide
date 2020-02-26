//
// Created by kraft on 16.07.19.
//

#ifndef WAVEGUIDEPROBLEM_DOFMANAGER_H
#define WAVEGUIDEPROBLEM_DOFMANAGER_H

#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <vector>
#include <deal.II/base/index_set.h>
#include "../HSIEPreconditioner/HSIESurface.h"

struct LevelDofOwnershipData {
  unsigned int global_dofs;
  unsigned int owned_dofs;
  dealii::IndexSet locally_owned_dofs;
  dealii::IndexSet input_dofs;
  dealii::IndexSet output_dofs;
  dealii::IndexSet locally_relevant_dofs;

  LevelDofOwnershipData() {
    global_dofs = 0;
    owned_dofs = 0;
    locally_owned_dofs.clear();
    input_dofs.clear();
    output_dofs.clear();
    locally_relevant_dofs.clear();
  }

  LevelDofOwnershipData(unsigned int in_global) {
    global_dofs = in_global;
    owned_dofs = 0;
    locally_owned_dofs.clear();
    locally_owned_dofs.set_size(in_global);
    input_dofs.clear();
    input_dofs.set_size(in_global);
    output_dofs.clear();
    output_dofs.set_size(in_global);
    locally_relevant_dofs.clear();
    locally_relevant_dofs.set_size(in_global);
  }
};

class DOFManager {
 public:
  unsigned int global_level;
  std::vector<dealii::IndexSet> own_dofs_per_process;  // via MPI_Build...
  dealii::IndexSet local_dofs;                         // via MPI_Build
  const unsigned int dofs_per_edge;                    // constructor
  const unsigned int dofs_per_face;                    // constructor
  const unsigned int dofs_per_cell;                    // constructor
  unsigned int n_global_dofs;                          // via MPI_Build
  unsigned int n_locally_owned_dofs;                   // via MPI_Build
  dealii::Triangulation<3, 3> *triangulation;
  dealii::DoFHandler<3, 3> *dof_handler;
  const dealii::FiniteElement<3, 3> *fe;
  bool computed_n_global;
  LevelDofOwnershipData * level_dofs;
  HSIESurface<5> ** hsie_surfaces;

  DOFManager(unsigned int, unsigned int, unsigned int,
             dealii::DoFHandler<3, 3> *, dealii::Triangulation<3, 3> *,
             const dealii::FiniteElement<3, 3> *, unsigned int);

  void compute_level_dofs();
  LevelDofOwnershipData compute_local_level_dofs();
  LevelDofOwnershipData compute_higher_level_dofs(unsigned int level);

  void init();
  unsigned int compute_n_own_dofs();
  dealii::IndexSet own_dofs_for_level(unsigned int local_level, unsigned int global_level);
  dealii::IndexSet input_dofs_for_level(unsigned int local_level, unsigned int global_level);
  dealii::IndexSet output_dofs_for_level(unsigned int local_level, unsigned int global_level);
  void MPI_build_global_index_set_vector();
  void compute_and_communicate_edge_dofs();
  void compute_and_communicate_face_dofs();
  void SortDofs();

  void compute_and_send_x_dofs();
  void compute_and_send_y_dofs();
  void compute_and_send_z_dofs();
  void receive_x_dofs();
  void receive_y_dofs();
  void receive_z_dofs();
  dealii::IndexSet get_dofs_for_boundary_id(dealii::types::boundary_id in_bid);
  dealii::IndexSet get_non_owned_dofs();
  void shift_own_to_final_dof_numbers();

  void update_interface_dofs_with_IndexSet(dealii::IndexSet in_new_indices,
                                           dealii::types::boundary_id in_bid);

  static int testValue();
  // TODO: There should be some functions checking for neighboring processes if
  // the communicated IndexSets are the same.
};

#endif  // WAVEGUIDEPROBLEM_DOFMANAGER_H
