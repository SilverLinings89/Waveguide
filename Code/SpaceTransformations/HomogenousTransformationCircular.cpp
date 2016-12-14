#ifndef HomogenousTransformationCircular_CPP
#define HomogenousTransformationCircular_CPP

using namespace dealii;

HomogenousTransformationCircular::HomogenousTransformationCircular ():
  XMinus( -(GlobalParams.M_R_XLength *0.5 - GlobalParams.M_BC_XMinus)),
  XPlus( GlobalParams.M_R_XLength *0.5 - GlobalParams.M_BC_XPlus),
  YMinus( -(GlobalParams.M_R_YLength *0.5 - GlobalParams.M_BC_YMinus)),
  YPlus( GlobalParams.M_R_YLength *0.5 - GlobalParams.M_BC_YPlus),
  ZMinus( - GlobalParams.M_R_ZLength *0.5 ),
  ZPlus( GlobalParams.M_R_ZLength *0.5 )
{


}


bool HomogenousTransformationCircular::PML_in_X(Point<3> &p) {
  return p(0) < XMinus ||p(0) > XPlus;
}

bool HomogenousTransformationCircular::PML_in_Y(Point<3> &p) {
  return p(1) < YMinus ||p(1) > YPlus;
}

bool HomogenousTransformationCircular::PML_in_Z(Point<3> &p) {
  return p(2) < ZMinus ||p(2) > ZPlus;
}

bool HomogenousTransformationCircular::Preconditioner_PML_in_Z(Point<3> &p, unsigned int block) {
  double l = structure->Layer_Length();
  double width = l * 1.0;
  if( block == GlobalParams.NumberProcesses-2) return false;
  if ( block == GlobalParams.MPI_Rank-1){
    return true;
  } else {
    return false;
  }
}

double HomogenousTransformationCircular::Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block ){
  double l = structure->Layer_Length();
  double width = l * 1.0;

  return p(2) +GlobalParams.M_R_ZLength/2.0 - ((double)block +1)*l;

}

double HomogenousTransformationCircular::PML_X_Distance(Point<3> &p){
  if(p(0) >0){
    return p(0) - XPlus ;
  } else {
    return -p(0) - XMinus;
  }
}

double HomogenousTransformationCircular::PML_Y_Distance(Point<3> &p){
  if(p(1) >0){
    return p(1) - YMinus;
  } else {
    return -p(1) - YPlus;
  }
}

double HomogenousTransformationCircular::PML_Z_Distance(Point<3> &p){
  if(p(3) < 0) {
    return - (p(2) + (GlobalParams.M_R_ZLength / 2.0));
  } else {
    return p(2) - (GlobalParams.M_R_ZLength / 2.0);
  }
}

#endif
