#pragma once
/**
 * @file JacobianForCell.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/differentiation/sd/symengine_number_types.h>
#include "../Core/Types.h"

/**
 * @brief This class is only for internal use.
 * The jacobian it represents is used in the HSIESurface to represent the transformation of the cell onto a cuboid. If the external direction is chosen axis-parallel, this is an identity transformation.
 * 
 */
class JacobianForCell {
public:
  
  dealii::Differentiation::SD::types::substitution_map surface_wide_substitution_map;
  BoundaryId boundary_id;
  double additional_component;
  std::vector<bool> b_ids_have_hsie;
  MathExpression x;
  MathExpression y;
  MathExpression z;
  MathExpression z0;
  dealii::Tensor<1, 3, MathExpression> F;
  dealii::Tensor<2, 3, MathExpression> J;

  /**
   * @brief Construct a new Jacobian For Cell object
   * 
   * @param in_fad denotes which faces are angled (45 degrees) and which are not.
   * @param b_id the boundary id of the surface the cell belongs to.
   * @param additional_component orthogonal surface coordinate.
   */
  JacobianForCell(FaceAngelingData & in_fad, const BoundaryId& b_id, double additional_component);

  virtual ~JacobianForCell() = default;

  /**
   * @brief Builds the base data for the provided cell
   * 
   */
  void reinit_for_cell(CellIterator2D);

  /**
   * @brief Does the same as the constructor.
   * 
   * @param in_fad denotes which faces are angled (45 degrees) and which are not.
   * @param b_id the boundary id of the surface the cell belongs to.
   * @param additional_component orthogonal surface coordinate.
   */
  void reinit(FaceAngelingData & in_fad, const BoundaryId& b_id, double additional_component);

  /**
   * @brief Get the C G and J tensors used in the HSIE formulation.
   * @see HSIESurface
   * 
   * @return JacobianAndTensorData 
   */
  auto get_C_G_and_J(Position2D) -> JacobianAndTensorData;

  /**
   * @brief For a given Cordinate in 3D, this identifies its position on the surface and the orthogonal part.
   * 
   * @param in_point The position in 3D
   * @return std::pair<Position2D,double> Th cordinate in 2D and the orthogonal part
   */
  std::pair<Position2D,double> split_into_triangulation_and_external_part(const Position in_point);

  /**
   * @brief Checks if a edge on the HSIESurface points in the x or y direction
   * 
   * @param line An iterator pointing to a line in a surface triangualtion.
   * @return true the line points in the x-direction
   * @return false the line does not point in the x-direction
   */
  static bool is_line_in_x_direction(dealii::internal::DoFHandlerImplementation::Iterators<2, 2, false>::line_iterator line);

  /**
   * @brief Checks if a edge on the HSIESurface points in the x or y direction
   * 
   * @param line An iterator pointing to a line in a surface triangualtion.
   * @return true the line points in the y-direction
   * @return false the line does not point in the y-direction
   */
  static bool is_line_in_y_direction(dealii::internal::DoFHandlerImplementation::Iterators<2, 2, false>::line_iterator line);

  /**
   * @brief Evaluates the Jacobian at the given position 
   * 
   * @param position 2D coordinate to evaluate the jacobian at.
   * @return dealii::Tensor<2,3,double> 
   */

  dealii::Tensor<2,3,double> get_J_hat_for_position(const Position2D & position) const;  

  /**
   * @brief Takes a position on the surface and provides the 3D coordinate.
   * 
   * @param position location on the surface
   * @return Position 
   */
  auto transform_to_3D_space(Position2D position) -> Position;
};
