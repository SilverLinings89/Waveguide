#include "ShapeDescription.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "../GlobalObjects/GlobalObjects.h"

ShapeDescription::ShapeDescription() {}

ShapeDescription::~ShapeDescription() {}

void ShapeDescription::SetStraight() {
  z.push_back(GlobalParams.global_z_shift);
  z.push_back(GlobalParams.Geometry_Size_Z + GlobalParams.global_z_shift);
  m.push_back(0);
  m.push_back(0);
}

void ShapeDescription::SetByString(std::string str) {
  std::istringstream iss(str);
  std::string token;
  std::getline(iss, token, ',');
  Sectors = std::stoi(token);
  double z_0 = 0.0;
  int i = 0;
  for (i = 0; i < Sectors; i++) {
    std::getline(iss, token, ',');
    if(i == 0) {
      z_0 = std::stod(token);
      z.push_back(0);
    } else {
      z.push_back(std::stod(token) - z_0);
    }
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
