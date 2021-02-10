#pragma once

#include <utility>
#include "../Core/Types.h"

struct PMLMeshTransformation {
    inline static std::pair<double, double> default_x_range;
    inline static std::pair<double, double> default_y_range;
    inline static std::pair<double, double> default_z_range;
    inline static double base_coordinate_for_transformed_direction;
    inline static unsigned int outward_direction;
    inline static std::array<bool, 6> transform_coordinate;
    
    PMLMeshTransformation();
    static void set(std::pair<double, double> in_x_range, std::pair<double, double> in_y_range, std::pair<double, double> in_z_range, double in_base_coordinate, unsigned int in_outward_direction, std::array<bool, 6> in_transform_coordinate);
    static Position transform(const Position &in_p);
    static Position undo_transform(const Position &in_p);
};