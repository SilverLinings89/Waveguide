#pragma once

/**
 * @file ShapeDescription.h
 * @author your name (you@domain.com)
 * @brief An object used to wrap the description of the prescribed waveguide shapes.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <string>
#include <vector>

class ShapeDescription {
 public:
  ShapeDescription();

  ~ShapeDescription();

  void SetByString(std::string);

  void SetStraight();

  int Sectors;
  
  std::vector<double> m, v, z;
};
