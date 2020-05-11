/*
 * MPICommunicator.h
 *
 *  Created on: Apr 27, 2020
 *      Author: kraft
 */

#ifndef CODE_HIERARCHY_MPICOMMUNICATOR_H_
#define CODE_HIERARCHY_MPICOMMUNICATOR_H_

#include <mpi.h>
#include <vector>
#include "../Helpers/Enums.h"


class MPICommunicator {
public:
  MPICommunicator();
  virtual ~MPICommunicator();
  std::vector<MPI_Comm> communicators_by_level;
  std::pair<bool, unsigned int> get_neighbor_for_interface(
      Direction in_direction);
  void initialize();
};

#endif /* CODE_HIERARCHY_MPICOMMUNICATOR_H_ */
