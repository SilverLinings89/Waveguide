#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/differentiation/sd/symengine_number_types.h>
#include "../Core/Types.h"

class JacobianForCell {
public:
  JacobianForCell(const Position &in_V0, const BoundaryId& b_id, double additional_component);
  dealii::Differentiation::SD::types::substitution_map surface_wide_substitution_map;
  std::pair<Position2D,double> V0;
  BoundaryId boundary_id;
  double additional_component;
  virtual ~JacobianForCell() = default;
  void reinit_for_cell(CellIterator2D);
  std::vector<bool> b_ids_have_hsie;
  void reinit(const Position &in_V0, const BoundaryId& b_id, double additional_component);
  auto get_C_G_and_J(Position2D) -> JacobianAndTensorData;
  MathExpression x;
  MathExpression y;
  MathExpression z;
  MathExpression V0x;
  MathExpression V0y;
  MathExpression V0z;
  MathExpression z0;
  
  dealii::Tensor<1, 3, MathExpression> F;
  dealii::Tensor<2, 3, MathExpression> J;

  static bool is_line_in_x_direction(dealii::internal::DoFHandlerImplementation::Iterators<dealii::DoFHandler<2, 2>, false>::line_iterator line);
  static bool is_line_in_y_direction(dealii::internal::DoFHandlerImplementation::Iterators<dealii::DoFHandler<2, 2>, false>::line_iterator line);
  dealii::Tensor<2,3,double> get_J_hat_for_position(const Position2D &) const;  
  auto transform_to_3D_space(Position2D) -> Position;
  auto split_into_triangulation_and_external_part(Position) -> std::pair<Position2D,double>;
};
