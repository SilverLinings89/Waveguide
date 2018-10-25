#ifndef HSIEPRECONDITIONERBASE_H
#define HSIEPRECONDITIONERBASE_H

#include <deal.II/base/config.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include "./HSIEDofType.h"

template <int hsie_order>
class HSIEPreconditionerBase : dealii::TrilinosWrappers::PreconditionBase {
  using dealii::TrilinosWrappers::PreconditionBase::vmult;
  dealii::parallel::distributed::Triangulation<2, 3> surf_tria;
  std::map<dealii::parallel::distributed::Triangulation<2, 3>::cell_iterator,
           dealii::parallel::distributed::Triangulation<3, 3>::face_iterator>
      association;
  int HSIE_dofs_type_1_factor;  // HSIE paper p.13 1.
  int HSIE_dofs_type_2_factor;  // HSIE paper p.13 3.
  int HSIE_dofs_type_3_factor;  // HSIE paper p.13 4a).
  int surface_vertices;
  int surface_edges;
  int surface_faces;
  int HSIE_degree;
  dealii::DoFHandler hsie_dof_handler;
  dealii::FE_Nedelec<2> fe_nedelec;
  dealii::FE_Q<2> fe_q;
  dealii::QGauss<3> quadrature_formula;

 public:
  /**
   * This constructor of a HSIE Predonditioner works as follows. It takes a
   * given Triangulation and cuts it at a specified z-coordinate and applies
   * infinite elements to the surface generated in this way. This class will be
   * renamed appropriately later since it is not really a Preconditioner as much
   * as it is a general implementation of the HSIE method which can also be
   * employed as part of a preconditioner.
   * \param in_tria A handle to the triangulation which contains the surface.
   * \param in_z the z-coordinate at which to attach the infinite Elements.
   */
  HSIEPreconditionerBase(
      const dealii::parallel::distributed::Triangulation<3> *in_tria,
      double in_z);
  ~HSIEPreconditionerBase();

  /**
   * The order of the Hardy-Space polynomials was given as a template argument
   * to an object of this type, so no arguments are required for this function.
   * Eventually a version of this function will be added which can deal with
   * higher then lowest order elements in the interior.
   */
  unsigned int n_dofs();

  /**
   * Similar to the function n_dofs() this function doesn't currently need any
   * arguments. This one however returns the number of HSIE-Dofs per face, which
   * will act as a replacement of dealii's similar functions dofs_per_face etc.
   */
  unsigned int n_dofs_per_face();

  /**
   * This function brings the Object-member
   * HSIEPreconditionerBase::system_matrix which is currently supposed to hold
   * all HSIE-dofs (including the Nedelec-elements of the surface elements).
   * This makes coupling the dofs to the interior via dof-constraints necessary.
   */
  void assemble_block();

  /**
   * This must be called before assemble block since it initializes the
   * HSIEPreconditionerBase::system_matrix.
   */
  void setup_system();

  /**
   * This function implements the following equation:
   * \f[
   * a(U,V) := \frac{-2 \operatorname{i} \kappa_0}{2 \pi} \int_{S_0}
   * U(z)v(\bar{z})|\mathrm{d}z|
   * \f]
   * which is called from A() which in turn is required to build the
   * HSIEPreconditionerBase::system_matrix. \param u is an
   * object describing the first type of dof, \param v describes the second dof,
   * \param use_curl_formulation simplifies the usage of this function: It
   * causes the function not to use $u_1$ etc. but the terms in A()'s argument
   * list (see equation (30) in the paper High order Curl-conforming Hardy space
   * infinite elements for exterior Maxwell problems.) \param x is the Point in
   * the surface triangulation where to evaluate the bilinear form.
   */
  std::complex<double> a(HSIE_Dof_Type<hsie_order> u,
                         HSIE_Dof_Type<hsie_order> v, bool use_curl_fomulation,
                         dealii::Point<2, double> x);
  /**
   * This function implements the following equation:
   * \f[
   * A(U,V) := \int_T \sum_{i,j=1}^{3}g_{ij}(\hat{x})a(U_i(\cdot,
   * \hat{x}),V_j(\cdot, \hat{x}))\mathrm{d} \hat{x}
   * \f]
   * where T is a Triangulation of the surface. For more details see the
   * publication mentioned in the description of a().
   * \param u is an object describing the first type of dof,
   * \param v describes the second dof,
   * \param G is the $3 \times 3$ matrix consisting of $g_{ij}$
   * \param use_curl_formulation has the same purpose as in a().
   */
  std::complex<double> A(HSIE_Dof_Type<hsie_order> u,
                         HSIE_Dof_Type<hsie_order> v,
                         dealii::Tensor<2, 3, double> G,
                         bool use_curl_fomulation);
  /**
   * This matrix contains the couplings between all HSIE-dofs. This also
   * contains the Nedelec-Elements of the surface triangulation.
   */
  dealii::TrilinosWrappers::SparseMatrix system_matrix;
};

#endif
