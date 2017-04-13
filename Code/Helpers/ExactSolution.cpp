
#ifndef ExactSolutionFlag_CPP
#define ExactSolutionFlag_CPP

#include "ExactSolution.h"


double ExactSolution::value (const Point<3> &p , const unsigned int component) const
{
  bool zero = false;
  if(p[0] > GlobalParams.M_R_XLength/2.0 - GlobalParams.M_BC_XPlus) zero = true;
  if(p[0] < -GlobalParams.M_R_XLength/2.0 + GlobalParams.M_BC_XMinus) zero = true;
  if(p[1] > GlobalParams.M_R_YLength/2.0 - GlobalParams.M_BC_YPlus) zero = true;
  if(p[1] < -GlobalParams.M_R_YLength/2.0 + GlobalParams.M_BC_YMinus) zero = true;
  if(p[2] > GlobalParams.M_R_ZLength/2.0) zero = true;
  if(zero){
    return 0;
  } else {
    return ModeMan.get_input_component( component, p, 0);
  }
	//return 0.0;
}


void ExactSolution::vector_value (const Point<3> &p,	Vector<double> &values) const
{
  bool zero = false;
  if(p[0] > GlobalParams.M_R_XLength/2.0 - GlobalParams.M_BC_XPlus) zero = true;
  if(p[0] < -GlobalParams.M_R_XLength/2.0 + GlobalParams.M_BC_XMinus) zero = true;
  if(p[1] > GlobalParams.M_R_YLength/2.0 - GlobalParams.M_BC_YPlus) zero = true;
  if(p[1] < -GlobalParams.M_R_YLength/2.0 + GlobalParams.M_BC_YMinus) zero = true;
  if(p[2] > GlobalParams.M_R_ZLength/2.0) zero = true;
  if(zero) {
    for (unsigned int c=0; c<6; ++c) values[c] = 0.0;
  } else {
    for (unsigned int c=0; c<6; ++c) values[c] = ModeMan.get_input_component( c, p, 0);
  }
}

ExactSolution::ExactSolution(): Function<3>(6) {

}

#endif
