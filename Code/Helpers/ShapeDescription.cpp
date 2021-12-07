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
#include "../GlobalObjects/GlobalObjects.h"

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
  GlobalParams.Sector_thickness = (z[z.size() - 1] - z[0]) / (Sectors - 1);
  std::cout << "Sector thickness: " << GlobalParams.Sector_thickness << std::endl;
  /**
  if(GlobalParams.MPI_Rank == 0) {
    std::cout << "I found " << std::to_string(Sectors) << " sectors and initialized them:" << std::endl;;
    for(unsigned int i = 0; i < Sectors; i++) {
      std::cout << "  Sector " << i  << " has m: " << m[i] << " v: " << v[i] << " and z:" << z[i] << std::endl;
    }
  }
  **/
}
