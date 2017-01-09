#ifndef SpaceTransformation_CPP
#define SpaceTransformation_CPP

#include "SpaceTransformation.h"

std::pair<int, double> SpaceTransformation::Z_to_Sector_and_local_z(double in_z) {
  std::pair<int, double> ret(0,0.0);
  if(in_z <= GlobalParams.M_R_ZLength/2.0) {
    ret.first = 0;
    ret.second = 0.0;
  } else if(abs(in_z) < GlobalParams.M_R_ZLength/2.0) {
    ret.first = floor((in_z + GlobalParams.M_R_ZLength/2.0)/ (GlobalParams.LayerThickness));
    ret.second = (in_z + GlobalParams.M_R_ZLength/2.0 - (ret.first*GlobalParams.LayerThickness))/ (GlobalParams.LayerThickness);
  } else {
    ret.first = GlobalParams.NumberProcesses -1;
    ret.second = 1.0;
  }
}

SpaceTransformation::SpaceTransformation(int in_dofs_per_layer) :
    dofs_per_layer(in_dofs_per_layer),
    boundary_dofs_in(in_dofs_per_layer),
    boundary_dofs_out(in_dofs_per_layer),
    epsilon_M(GlobalParams.M_W_epsilonout),
    epsilon_K(GlobalParams.M_W_epsilonin),
    sectors(GlobalParams.M_W_Sectors),
    deltaY(GlobalParams.M_W_Delta)
    {
  InitialQuality = 0;
}

#endif
