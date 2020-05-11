/*
 * DofIndexData.h
 *
 *  Created on: Apr 24, 2020
 *      Author: kraft
 */

#ifndef CODE_HIERARCHY_DOFINDEXDATA_H_
#define CODE_HIERARCHY_DOFINDEXDATA_H_

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

#endif /* CODE_HIERARCHY_DOFINDEXDATA_H_ */
