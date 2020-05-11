/*
 * LevelDofIndexData.h
 *
 *  Created on: Apr 29, 2020
 *      Author: kraft
 */

#ifndef CODE_HIERARCHY_LEVELDOFINDEXDATA_H_
#define CODE_HIERARCHY_LEVELDOFINDEXDATA_H_

class LevelDofIndexData {
  unsigned int **boundary_dofs;
  bool **has_hsie_part;
public:
  LevelDofIndexData();
  virtual ~LevelDofIndexData();
};

#endif /* CODE_HIERARCHY_LEVELDOFINDEXDATA_H_ */
