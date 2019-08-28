/*
 * GemoetryManager.h
 * An object of this type handles all the geometric information of the run.
 * It can be used by the Mesh-Generator to retrieve data, or by the Simulation
 * to retrieve neighboring-process information as well as surface-types
 * (boundary condition, dirichlet surface etc.)
 * An object of this type is available as a static variable.
 * Created on: Jun 19, 2019
 * Author: Pascal Kraft
 */

#ifndef CODE_HELPERS_GEOMETRY_H_
#define CODE_HELPERS_GEOMETRY_H_

#include <utility>
#include <deal.II/base/point.h>
#include "Parameters.h"


enum Direction {
    MinusX = 0,
    PlusX = 1,
    MinusY = 2,
    PlusY = 3,
    MinusZ = 4,
    PlusZ = 5
};

class GeometryManager {
public:
    GeometryManager();

    virtual ~GeometryManager();

    void initialize(Parameters & parameters);

    std::pair<double, double> x_range;
    std::pair<double, double> y_range;
    std::pair<double, double> z_range;

    std::pair<double, double> compute_x_range(Parameters & parameters);

    std::pair<double, double> compute_y_range(Parameters & parameters);

    std::pair<double, double> compute_z_range(Parameters & parameters);

    void set_x_range(std::pair<double, double>);

    void set_y_range(std::pair<double, double>);

    void set_z_range(std::pair<double, double>);

    // This function returns false in the first return value if the neighbour is
    // not a process but an outside boundary. Otherwise it returns the MPI Rank of
    // the neighboring process in the that direction.
    std::pair<bool, unsigned int> get_neighbor_for_interface(Direction);

    bool math_coordinate_in_waveguide(dealii::Point<3, double>) const;

};

#endif /* CODE_HELPERS_GEOMETRY_H_ */
