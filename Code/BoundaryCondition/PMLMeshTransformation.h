#pragma once
/**
 * @file PMLMeshTransformation.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <utility>
#include "../Core/Types.h"

/**
 * \class PMLMeshTransformation
 * 
 * \brief Generating the basic mesh for a PML domain is simple because it is an axis parallel cuboid. This functions shifts and stretches the domain to the correct proportions.
 * 
 * Specifically, the implementation is done in the operator() function. Choosing this nomenclature, the function is compatible with deal.IIs interface for a coordinate transformation and an object of this type can be used directy in the GridTools::transform function.
 */ 

struct PMLMeshTransformation {
    std::pair<double, double> default_x_range;
    std::pair<double, double> default_y_range;
    std::pair<double, double> default_z_range;
    double base_coordinate_for_transformed_direction;
    unsigned int outward_direction;
    std::array<bool, 6> transform_coordinate;
    
    PMLMeshTransformation();
    PMLMeshTransformation(std::pair<double, double> in_x_range, std::pair<double, double> in_y_range, std::pair<double, double> in_z_range, double in_base_coordinate, unsigned int in_outward_direction, std::array<bool, 6> in_transform_coordinate);

    /**
     * @brief Transforms a coordinate of the unit cube onto the actual sizes provided in the constructor of this object.
     *
     * 
     * @param in_p The coordinate to be transformed.
     * @return Position The transformed coordinated.
     */
    Position operator()(const Position &in_p) const;

    /**
     * @brief Inverse operation of operator().
     * 
     * @param in_p The coordinate on which to undo the transformation
     * @return Position The coordinate before operator() was applied to it.
     */
    Position undo_transform(const Position &in_p);
};