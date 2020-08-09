#pragma once
class LevelDofIndexData {
  unsigned int **boundary_dofs;
  bool **has_hsie_part;
public:
  LevelDofIndexData();
  virtual ~LevelDofIndexData();
};
