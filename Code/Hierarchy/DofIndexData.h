#pragma once

#include <deal.II/base/index_set.h>
#include "LevelDofIndexData.h"

class DofIndexData {
public:
  bool *isSurfaceNeighbor;
  std::vector<LevelDofIndexData> indexCountsByLevel;

  DofIndexData();
  ~DofIndexData();

  void communicateSurfaceDofs();
  void initialize();
  void initialize_level(unsigned int level);
};
