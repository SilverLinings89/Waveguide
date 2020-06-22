#pragma once

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
