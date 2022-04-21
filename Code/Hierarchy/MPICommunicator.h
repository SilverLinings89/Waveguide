#pragma once
/**
 * @file MPICommunicator.h
 * @author Pascal Kraft
 * @brief This class stores the implementation of the MPICommunicator type.
 * @version 0.1
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <mpi.h>
#include <vector>
#include "../Core/Enums.h"

/**
 * @brief Utility class that provides additional information about the MPI setup on the level.
 * This object wraps all information about communicators on all levels, i.e.which MPI_COMM to use on which level, ranks of this process on all levels and provides some useful functions like computing the neightbor MPI ranks by interface id.
 */
class MPICommunicator {
public:
  MPICommunicator();
  ~MPICommunicator();
  std::vector<MPI_Comm> communicators_by_level;
  std::vector<unsigned int> rank_on_level;

  /**
   * @brief Get the neighbor for interface
   * For the provided surface, this function computes the MPI rank of the neighbor and if it exists.
   * 
   * @param in_direction The direction to check in.
   * @return std::pair<bool, unsigned int> First is true, if there is a neighbor in this direction. Second is the global MPI_rank of the neighbor.
   */
  std::pair<bool, unsigned int> get_neighbor_for_interface(Direction in_direction);

  /**
   * @brief Initializes this object by computing the level communicators.
   * 
   */
  void initialize();

  /**
   * @brief This is used to free up some space and is just in general a good practice.
   * 
   */
  void destroy_comms();
};
