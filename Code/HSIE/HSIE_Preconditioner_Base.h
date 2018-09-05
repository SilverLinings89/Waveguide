#ifndef HSIEPRECONDITIONERBASE_H
#define HSIEPRECONDITIONERBASE_H

#include <complex.h>
#include <deal.II/base/point.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

class HSIEPreconditionerBase {
  int HSIE_dofs_type_1;  // HSIE paper p.13 1.
  int HSIE_dofs_type_2;  // HSIE paper p.13 3.
  int HSIE_dofs_type_3;  // HSIE paper p.13 4a).
  int surface_vertices;
  int surface_edges;
  int surface_faces;
  int HSIE_degree;

 public:
  HSIEPreconditionerBase(
      dealii::parallel::distributed::Triangulation<3> *in_tria);
  ~HSIEPreconditionerBase();
  void compute_number_of_dofs(int hsie_degree);
  void assemble_blocks();
  void vmult(dealii::TrilinosWrappers::MPI::BlockVector &dst,
             const dealii::TrilinosWrappers::MPI::BlockVector &src) const;
};

#endif
