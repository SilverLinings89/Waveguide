/*
 * JacobianForCell.h
 *
 *  Created on: Jun 16, 2020
 *      Author: kraft
 */

#ifndef CODE_HSIEPRECONDITIONER_JACOBIANFORCELL_H_
#define CODE_HSIEPRECONDITIONER_JACOBIANFORCELL_H_
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/differentiation/sd/symengine_number_types.h>

class JacobianForCell {
public:
  JacobianForCell();
  virtual ~JacobianForCell();
  bool is_x_tilted;
  bool is_y_tilted;
  dealii::Tensor<2, 3> J;
  void reinit_for_cell(dealii::DoFHandler<2>::active_cell_iterator,
      std::vector<bool> b_ids_have_hsie);
  void reset();
  double x0;
  double y0;
  double Lx;
  double Ly;
  dealii::Tensor<1, 3, dealii::Differentiation::SD::Expression> F;

  double compute_L_x_for_cell(dealii::DoFHandler<2>::active_cell_iterator cell);
  double compute_L_y_for_cell(dealii::DoFHandler<2>::active_cell_iterator cell);
  bool is_cell_x_tilted(dealii::DoFHandler<2>::active_cell_iterator cell);
  bool is_cell_at_x_interface(dealii::DoFHandler<2>::active_cell_iterator cell);
  bool is_cell_y_tilted(dealii::DoFHandler<2>::active_cell_iterator cell);
  bool is_cell_at_y_interface(dealii::DoFHandler<2>::active_cell_iterator cell);
};

#endif /* CODE_HSIEPRECONDITIONER_JACOBIANFORCELL_H_ */
