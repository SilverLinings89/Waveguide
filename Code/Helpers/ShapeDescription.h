#pragma once

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
