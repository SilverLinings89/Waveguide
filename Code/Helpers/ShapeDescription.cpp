/*
 * ShapeDescription.cpp
 *
 *  Created on: Feb 9, 2018
 *      Author: kraft
 */

#include "ShapeDescription.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

ShapeDescription::ShapeDescription() {}

ShapeDescription::~ShapeDescription() {}

void ShapeDescription::SetByString(std::string str) {
    std::vector<std::string> ret;
    std::istringstream iss(str);
    std::string token;
    std::getline(iss, token, ',');
    Sectors = std::stoi(token);
    int i = 0;
    for (i = 0; i < Sectors; i++) {
        std::getline(iss, token, ',');
        z.push_back(std::stod(token));
    }
    for (i = 0; i < Sectors; i++) {
        std::getline(iss, token, ',');
        m.push_back(std::stod(token));
    }
    for (i = 0; i < Sectors; i++) {
        std::getline(iss, token, ',');
        v.push_back(std::stod(token));
    }
}
