
#ifndef ExactSolutionFlag_CPP
#define ExactSolutionFlag_CPP

#include "ExactSolution.h"
#include "PointVal.cpp"

double ExactSolution::value (const Point<3> &p , const unsigned int component) const
{
  if(is_rectangular) {

  }else {
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
  }

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

std::vector<std::string> ExactSolution::split(std::string str,std::string sep) const{
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    std::vector<std::string> arr;
    current=strtok(cstr,sep.c_str());
    while(current!=NULL){
        arr.push_back(current);
        current=strtok(NULL,sep.c_str());
    }
    return arr;
}

double scientific_string_to_double(std::string inp) {
  std::istringstream os(inp);
  double d;
  os >> d;
  return d;
}

ExactSolution::ExactSolution(bool in_rectangular): Function<3>(6) {
  is_rectangular = in_rectangular;
  if(is_rectangular) {
    std::ifstream input( "filename.ext" );
    std::string line;
    getline( input, line );
    getline( input, line );
    float l_val =  3.0;
    for( ; getline( input, line ); )
    {
      std::vector<std::string> ls = split(line, "\t");
      float x = std::stof(ls[2]);
      if(x < l_val) {
        mesh_points.push_back(x);
        l_val = x;
      }
    }
    mesh_points.shrink_to_fit();
    unsigned int cnt  = mesh_points.size();
    PointVal ** vals = new PointVal*[cnt];
    for(int i = 0; i < cnt; i++) {
      vals[i] = new PointVal[cnt];
    }

    std::ifstream input2( "filename.ext" );
    std::string line2;
    getline( input2, line2 );
    getline( input2, line2 );
    for (int i = 0; i < cnt ; ++ i) {
      for (int j = 0; j < cnt ; ++ j) {
        getline( input2, line2 );
        std::vector<std::string> ls = split(line, "\t");
        vals[j][i].set(scientific_string_to_double(ls[5]),scientific_string_to_double(ls[4]),scientific_string_to_double(ls[3]),scientific_string_to_double(ls[8]),scientific_string_to_double(ls[7]),scientific_string_to_double(ls[6]));
      }
    }

  }
}

#endif
