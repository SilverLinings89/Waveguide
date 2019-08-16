//
// Created by kraft on 16.08.19.
//

#ifndef WAVEGUIDEPROBLEM_HSIESURFACE_H
#define WAVEGUIDEPROBLEM_HSIESURFACE_H

#include <deal.II/grid/tria.h>

template<unsigned int ORDER>
class HSIESurface {
    unsigned int n_total_hsie_dofs;
    dealii::Triangulation<3,3> * main_triangulation;
    dealii::Triangulation<2,3> surface_triangulation;
    unsigned int n_dofs_shared_with_interior;
    unsigned int n_pure_hsie_dofs;
    const unsigned int b_id;
    unsigned int level;

public:
    HSIESurface(dealii::Triangulation<3,3> * in_main_triangulation, unsigned int in_boundary_id, unsigned int in_level);
    void prepare_surface_triangulation();
    void compute_dof_numbers();
    void fill_matrix(dealii::SparseMatrix<double>* , dealii::IndexSet);
    unsigned int get_n_own_dofs();

};


#endif //WAVEGUIDEPROBLEM_HSIESURFACE_H
