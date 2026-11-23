#pragma once

/**
 * @file PMLMeshTransformation.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief Coordinate transformation for PML domains.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <array>
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
    /** @brief Lower and upper x coordinate of the cuboid before the transformation. */
    std::pair<double, double> default_x_range;
    /** @brief Lower and upper y coordinate of the cuboid before the transformation. */
    std::pair<double, double> default_y_range;
    /** @brief Lower and upper z coordinate of the cuboid before the transformation. */
    std::pair<double, double> default_z_range;
    /** @brief Coordinate in the outward direction at which the transformation is the identity (the surface of the inner domain). */
    double base_coordinate_for_transformed_direction;
    /** @brief Coordinate direction (0 = x, 1 = y, 2 = z) in which the PML extends away from the inner domain. */
    unsigned int outward_direction;
    /** @brief Indexed by boundary id (0 to 5). If the entry is true, the points on that side of the cuboid are moved outward by their distance to the base coordinate. */
    std::array<bool, 6> transform_coordinate;
    
    /**
     * @brief Creates a transformation with empty ranges and all flags set to false. operator() is the identity for it.
     */
    PMLMeshTransformation();

    /**
     * @brief Stores the description of the cuboid and of the sides that are to be moved.
     *
     * @param in_x_range Lower and upper x coordinate of the cuboid.
     * @param in_y_range Lower and upper y coordinate of the cuboid.
     * @param in_z_range Lower and upper z coordinate of the cuboid.
     * @param in_base_coordinate Coordinate in the outward direction at which points are not moved.
     * @param in_outward_direction Coordinate direction (0, 1 or 2) in which the PML extends.
     * @param in_transform_coordinate For every boundary id: whether the points on that side of the cuboid are moved.
     */
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