#ifndef HSIEPRECONDITIONERBASE_H
#define HSIEPRECONDITIONERBASE_H

#include <deal.II/base/config.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/vector.h>
#include "./HSIEDofType.h"

template <int hsie_order>
class HSIEPreconditionerBase : dealii::TrilinosWrappers::PreconditionBase {
  using dealii::TrilinosWrappers::PreconditionBase::vmult;
  dealii::parallel::distributed::Triangulation<2, 3> surf_tria;
  std::map<dealii::parallel::distributed::Triangulation<2, 3>::cell_iterator,
           dealii::parallel::distributed::Triangulation<3, 3>::face_iterator>
      association;
  int HSIE_dofs_type_1;  // HSIE paper p.13 1.
  int HSIE_dofs_type_2;  // HSIE paper p.13 3.
  int HSIE_dofs_type_3;  // HSIE paper p.13 4a).
  int surface_vertices;
  int surface_edges;
  int surface_faces;
  int HSIE_degree;
  dealii::DoFHandler hsie_dof_handler;
  dealii::FE_Nedelec<2> fe_nedelec;
  dealii::FE_Q<2> fe_q;

 public:
  HSIEPreconditionerBase(
      const dealii::parallel::distributed::Triangulation<3> *in_tria, double z);
  ~HSIEPreconditionerBase();
  unsigned int compute_number_of_dofs();
  void assemble_blocks();

  std::complex<double> a(HSIE_Dof_Type<hsie_order> u,
                         HSIE_Dof_Type<hsie_order> v, bool use_curl_fomulation,
                         dealii::Point<2, double> x);
  std::complex<double> A(HSIE_Dof_Type<hsie_order> u,
                         HSIE_Dof_Type<hsie_order> v,
                         dealii::Tensor<2, 3, double> G,
                         bool use_curl_fomulation);
};

#endif
