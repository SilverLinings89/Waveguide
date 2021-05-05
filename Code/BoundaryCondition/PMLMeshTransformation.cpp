#include <utility>
#include "./PMLMeshTransformation.h"
#include "../Core/Types.h"

PMLMeshTransformation::PMLMeshTransformation():
    default_x_range(std::pair<double,double>(0,0)),
    default_y_range(std::pair<double,double>(0,0)),
    default_z_range(std::pair<double,double>(0,0)),
    base_coordinate_for_transformed_direction(0),
    outward_direction(0),
    transform_coordinate(std::array<bool, 6>())
 {

}

PMLMeshTransformation::PMLMeshTransformation(std::pair<double, double> in_x_range, std::pair<double, double> in_y_range, std::pair<double, double> in_z_range, double in_base_coordinate, unsigned int in_outward_direction, std::array<bool, 6> in_transform_coordinate):
    default_x_range(in_x_range),
    default_y_range(in_y_range),
    default_z_range(in_z_range),
    base_coordinate_for_transformed_direction(in_base_coordinate),
    outward_direction(in_outward_direction),
    transform_coordinate(in_transform_coordinate)
{

}

Position PMLMeshTransformation::operator()(const Position &in_p) const {
    Position ret = in_p;
    double extension_factor = std::abs(in_p[outward_direction] - base_coordinate_for_transformed_direction);
    if(outward_direction != 0) {
        if(std::abs(in_p[0] - default_x_range.first ) < FLOATING_PRECISION && transform_coordinate[0]) ret[0] -= extension_factor;
        if(std::abs(in_p[0] - default_x_range.second) < FLOATING_PRECISION && transform_coordinate[1]) ret[0] += extension_factor;
    }
    if(outward_direction != 1) {
        if(std::abs(in_p[1] - default_y_range.first ) < FLOATING_PRECISION && transform_coordinate[2]) ret[1] -= extension_factor;
        if(std::abs(in_p[1] - default_y_range.second) < FLOATING_PRECISION && transform_coordinate[3]) ret[1] += extension_factor;
    }
    if(outward_direction != 2) {
        if(std::abs(in_p[2] - default_z_range.first ) < FLOATING_PRECISION && transform_coordinate[4]) ret[2] -= extension_factor;
        if(std::abs(in_p[2] - default_z_range.second) < FLOATING_PRECISION && transform_coordinate[5]) ret[2] += extension_factor;
    }
    return ret;
}

Position PMLMeshTransformation::undo_transform(const Position &in_p) {
    Position ret = in_p;
    if(in_p[0] < default_x_range.first)  ret[0] = default_x_range.first;
    if(in_p[0] > default_x_range.second) ret[0] = default_x_range.second;
    if(in_p[1] < default_y_range.first)  ret[1] = default_y_range.first;
    if(in_p[1] > default_y_range.second) ret[1] = default_y_range.second;
    if(in_p[2] < default_z_range.first)  ret[2] = default_z_range.first;
    if(in_p[2] > default_z_range.second) ret[2] = default_z_range.second;
    return ret;
}