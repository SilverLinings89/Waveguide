/*
 * ShapeDescription.h
 *
 *  Created on: Feb 9, 2018
 *      Author: kraft
 */

#ifndef CODE_HELPERS_SHAPEDESCRIPTION_H_
#define CODE_HELPERS_SHAPEDESCRIPTION_H_

#include <string>
#include <vector>

class ShapeDescription {
public:
    ShapeDescription();

    ~ShapeDescription();

    void SetByString(std::string);

    int Sectors;
    std::vector<double> m, v, z;
};

#endif /* CODE_HELPERS_SHAPEDESCRIPTION_H_ */
