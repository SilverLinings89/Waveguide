#pragma once
/**
 * @file LevelDofIndexData.h
 * @author Pascal Kraft
 * @brief Contains the class LevelDofIndexData. Legacy code that is not used by the solver.
 */

/**
 * @brief Legacy container of per-level dof data, only used as a member of DofIndexData.
 *
 * It allocates two 6 x 6 arrays (six being the number of sides of the domain) and initializes them. Nothing writes to or reads
 * from them afterwards.
 */
class LevelDofIndexData {
  /** @brief 6 x 6 array, initialized to 0 and never changed. */
  unsigned int **boundary_dofs;
  /** @brief 6 x 6 array, initialized to false and never changed. */
  bool **has_hsie_part;
public:
  /** @brief Allocates and initializes both arrays. */
  LevelDofIndexData();
  /** @brief Frees both arrays. */
  virtual ~LevelDofIndexData();
};
