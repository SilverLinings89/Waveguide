#pragma once

#include <utility>
#include "../Core/Types.h"

struct PMLMeshTransformation {
    std::pair<double, double> default_x_range;
    std::pair<double, double> default_y_range;
    std::pair<double, double> default_z_range;
    double base_coordinate_for_transformed_direction;
    unsigned int outward_direction;
    std::array<bool, 6> transform_coordinate;
    
    PMLMeshTransformation();
    PMLMeshTransformation(std::pair<double, double> in_x_range, std::pair<double, double> in_y_range, std::pair<double, double> in_z_range, double in_base_coordinate, unsigned int in_outward_direction, std::array<bool, 6> in_transform_coordinate);

    Position operator()(const Position &in_p) const;
    Position undo_transform(const Position &in_p);
};