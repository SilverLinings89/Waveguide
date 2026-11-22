#pragma once

/**
 * @file DofIndexData.h
 * @author Pascal Kraft
 * @brief Contains the class DofIndexData. Legacy code that is not used by the solver.
 */
#include <deal.II/base/index_set.h>
#include "LevelDofIndexData.h"

/**
 * @brief Legacy container for dof index data of all levels. No object of this type is created anywhere.
 *
 * The dof numbering is handled by GeometryManager, InnerDomain and the boundary condition classes. Of this class
 * only the constructor has an implementation with an effect.
 */
class DofIndexData {
public:
  /** @brief Array of length six, indexed by boundary id. True if there is a neighboring process behind that surface of the local domain. */
  bool *isSurfaceNeighbor;
  /** @brief One entry per sweeping level (GlobalParams.Sweeping_Level entries). The entries are only initialized, never filled. */
  std::vector<LevelDofIndexData> indexCountsByLevel;

  /** @brief Creates the per-level entries and computes isSurfaceNeighbor from the block indices of this process. */
  DofIndexData();
  /** @brief Frees isSurfaceNeighbor. */
  ~DofIndexData();

  /** @brief Empty implementation. */
  void communicateSurfaceDofs();
  /** @brief Empty implementation. */
  void initialize();
  /**
   * @brief Empty implementation.
   *
   * @param level Not used.
   */
  void initialize_level(unsigned int level);
};
