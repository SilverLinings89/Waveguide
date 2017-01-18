#ifndef SpaceTransformation_CPP
#define SpaceTransformation_CPP

#include "SpaceTransformation.h"

std::pair<int, double> SpaceTransformation::Z_to_Sector_and_local_z(double in_z) {
  std::pair<int, double> ret(0,0.0);
  if(in_z <= -GlobalParams.M_R_ZLength/2.0) {
    ret.first = 0;
    ret.second = 0.0;
  } else if(abs(in_z) < GlobalParams.M_R_ZLength/2.0) {
    ret.first = floor((in_z + GlobalParams.M_R_ZLength/2.0)/ (GlobalParams.LayerThickness));
    ret.second = (in_z + GlobalParams.M_R_ZLength/2.0 - (ret.first*GlobalParams.LayerThickness))/ (GlobalParams.LayerThickness);
  } else {
    ret.first = sectors - 1;
    ret.second = 1.0;
  }
  return ret;
}

SpaceTransformation::SpaceTransformation(int in_dofs_per_layer) :
    dofs_per_layer(in_dofs_per_layer),
    boundary_dofs_in(in_dofs_per_layer),
    boundary_dofs_out(in_dofs_per_layer),
    epsilon_K(GlobalParams.M_W_epsilonin),
    epsilon_M(GlobalParams.M_W_epsilonout),
    sectors(GlobalParams.M_W_Sectors),
    deltaY(GlobalParams.M_W_Delta)
    {
  InitialQuality = 0;
}

double SpaceTransformation::Sector_Length() {
  return GlobalParams.M_R_ZLength / (double)GlobalParams.M_W_Sectors;
}

int SpaceTransformation::Z_to_Layer(double in_z) {
  return floor((in_z + (GlobalParams.M_R_ZLength/2.0))/GlobalParams.LayerThickness);
}

bool SpaceTransformation::is_identity(Point<3, double> coord) {
  double sum =0.0;
  Point<3,double> temp = math_to_phys(coord);
  for(unsigned int i = 0; i < 3; i++) {
    sum += std::abs(temp[i] - coord[i]);
  }
  return sum < 0.0001;
}

#endif
