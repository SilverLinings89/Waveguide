#include <utility>
#include "./PMLMeshTransformation.h"
#include "../Core/Types.h"

PMLMeshTransformation::PMLMeshTransformation() {

}

void PMLMeshTransformation::set(std::pair<double, double> in_x_range, std::pair<double, double> in_y_range, std::pair<double, double> in_z_range, double in_base_coordinate, unsigned int in_outward_direction, std::array<bool, 6> in_transform_coordinate) {
    PMLMeshTransformation::default_x_range = in_x_range;
    PMLMeshTransformation::default_y_range = in_y_range;
    PMLMeshTransformation::default_z_range = in_z_range;
    PMLMeshTransformation::base_coordinate_for_transformed_direction = in_base_coordinate;
    PMLMeshTransformation::outward_direction = in_outward_direction;
    PMLMeshTransformation::transform_coordinate = in_transform_coordinate;
}

Position PMLMeshTransformation::transform(const Position &in_p) {
    Position ret = in_p;
    double extension_factor = std::abs(in_p[outward_direction] - PMLMeshTransformation::base_coordinate_for_transformed_direction);
    if(PMLMeshTransformation::outward_direction != 0) {
        if(std::abs(in_p[0] - PMLMeshTransformation::default_x_range.first ) < 0.001 && PMLMeshTransformation::transform_coordinate[0]) ret[0] -= extension_factor;
        if(std::abs(in_p[0] - PMLMeshTransformation::default_x_range.second) < 0.001 && PMLMeshTransformation::transform_coordinate[1]) ret[0] += extension_factor;
    }
    if(PMLMeshTransformation::outward_direction != 1) {
        if(std::abs(in_p[1] - PMLMeshTransformation::default_y_range.first ) < 0.001 && PMLMeshTransformation::transform_coordinate[2]) ret[1] -= extension_factor;
        if(std::abs(in_p[1] - PMLMeshTransformation::default_y_range.second) < 0.001 && PMLMeshTransformation::transform_coordinate[3]) ret[1] += extension_factor;
    }
    if(PMLMeshTransformation::outward_direction != 2) {
        if(std::abs(in_p[2] - PMLMeshTransformation::default_z_range.first ) < 0.001 && PMLMeshTransformation::transform_coordinate[4]) ret[2] -= extension_factor;
        if(std::abs(in_p[2] - PMLMeshTransformation::default_z_range.second) < 0.001 && PMLMeshTransformation::transform_coordinate[5]) ret[2] += extension_factor;
    }
    return ret;
}

Position PMLMeshTransformation::undo_transform(const Position &in_p) {
    Position ret = in_p;
    if(in_p[0] < PMLMeshTransformation::default_x_range.first)  ret[0] = PMLMeshTransformation::default_x_range.first;
    if(in_p[0] > PMLMeshTransformation::default_x_range.second) ret[0] = PMLMeshTransformation::default_x_range.second;
    if(in_p[1] < PMLMeshTransformation::default_y_range.first)  ret[1] = PMLMeshTransformation::default_y_range.first;
    if(in_p[1] > PMLMeshTransformation::default_y_range.second) ret[1] = PMLMeshTransformation::default_y_range.second;
    if(in_p[2] < PMLMeshTransformation::default_z_range.first)  ret[2] = PMLMeshTransformation::default_z_range.first;
    if(in_p[2] > PMLMeshTransformation::default_z_range.second) ret[2] = PMLMeshTransformation::default_z_range.second;
    return ret;
}